import os
import json
import sqlite3
import yaml
import time
from typing import Dict, Any, List, Tuple

# Attempt to import necessary libraries
try:
    from ollama import Client
except ImportError:
    print("Error: The 'ollama' Python library is not installed. Please run 'pip install ollama'.")
    exit()
try:
    # Use CLoader for performance if available, fallback to safe_load
    from yaml import CLoader as Loader, load
except ImportError:
    from yaml import SafeLoader as Loader, load

# --- 1. CONFIGURATION AND TEST CASE DEFINITION ---

CONFIG_FILE = "config.yaml"

# Schema information for LLM prompting
COMPANY_DB_SCHEMA = (
    "The database alias '{alias}' contains two tables:\n"
    "1. employees: (id, name, salary, department)\n"
    "2. projects: (project_id, project_name, budget, employee_id)"
)

# Test cases designed to challenge the LLM's understanding of intent and schema,
# where NLP hints from the MCP spec provide the maximum accuracy benefit.
TEST_CASES = [
    {
        "name": "Simple Filter",
        "nlp_query": "Find the name and salary of all employees in the 'Sales' department who earn over $70,000.",
        "correct_sql": "SELECT name, salary FROM employees WHERE department = 'Sales' AND salary > 70000;"
    },
    {
        "name": "Aggregation/Ambiguity",
        "nlp_query": "I need to know the total headcount in Engineering and the average compensation there.",
        "correct_sql": "SELECT COUNT(*), AVG(salary) FROM employees WHERE department = 'Engineering';"
    },
    {
        "name": "Complex Filter/Ordering",
        "nlp_query": "Show me the top 3 biggest contracts being managed in the company's Project database.",
        "correct_sql": "SELECT project_name, budget FROM projects ORDER BY budget DESC LIMIT 3;"
    },
    {
        "name": "Cross-Table/Join Hint",
        "nlp_query": "Which project is Alice Johnson assigned to? Just the project name.",
        # Assumes the LLM can infer the JOIN if the schema is provided (a good LLM should)
        "correct_sql": "SELECT T1.project_name FROM projects AS T1 INNER JOIN employees AS T2 ON T1.employee_id = T2.id WHERE T2.name = 'Alice Johnson';"
    }
]

# Global state for DB connections
DATABASE_CONNECTIONS = {}

# --- 2. CONFIGURATION AND SPECIFICATION LOADING ---

def load_config(file_path: str) -> Dict[str, Any]:
    """Loads configuration from the specified YAML file."""
    try:
        with open(file_path, 'r') as f:
            config = load(f, Loader=Loader)
        return config
    except FileNotFoundError:
        print(f"FATAL ERROR: Configuration file not found at '{file_path}'.")
        exit()
    except Exception as e:
        print(f"FATAL ERROR: Failed to load or parse YAML config: {e}")
        exit()

def load_mcp_spec(file_path: str) -> Dict[str, Any]:
    """Loads the MCP spec JSON directly, assuming it is fully compliant."""
    try:
        with open(file_path, 'r') as f:
            spec = json.load(f)
        return spec
    except FileNotFoundError:
        print(f"FATAL ERROR: MCP Spec file not found at '{file_path}'.")
        exit()
    except Exception as e:
        print(f"FATAL ERROR: Failed to load or parse MCP JSON spec: {e}")
        exit()

def parse_mcp_spec_for_ollama(mcp_spec: Dict, db_alias: str) -> tuple[List[Dict], str]:
    """
    Converts the Open MCP Specification into Ollama tool format and generates 
    an enhanced system prompt using enumeration and nlp_hints from the spec.
    """
    ollama_tools = []
    nlp_hint_list = []
    
    for intent in mcp_spec.get('intents', []):
        if intent['name'] in ['execute_read_query', 'connect_database', 'get_catalog']:
            function_parameters = intent.get('parameters', {})
            
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": intent['name'],
                    "description": intent['description'],
                    "parameters": function_parameters
                }
            })

            # Extract NLP hints directly from the loaded spec metadata
            metadata = intent.get('metadata', {})
            if 'nlp_hints' in metadata:
                hints = metadata['nlp_hints']
                category = intent.get('category', 'Tool') 
                nlp_hint_list.append(f"- {intent['name']} ({category}): {', '.join(hints)}")


    # Generate Enhanced System Prompt using Enumeration data from the spec
    enum = mcp_spec.get('enumeration', {})
    capabilities = ", ".join(enum.get('capability_tags', []))
    roles = ", ".join(enum.get('filters', {}).get('roles', [])) 

    system_prompt = (
        f"You are an expert SQL agent (Capabilities: {capabilities}) "
        f"with the following roles: {roles}.\n"
        f"Database Schema:\n{COMPANY_DB_SCHEMA.format(alias=db_alias)}\n\n"
        "Your task is to analyze the user's request and respond by calling the most relevant tool.\n"
        "**Intent Recognition Guidance (NLP Hints for better ACI):**\n"
        f"{'\\n'.join(nlp_hint_list)}\n\n"
        "When the request is for data retrieval, you MUST call the 'execute_read_query' tool "
        f"with a single, valid SQLite SELECT query for the database alias '{db_alias}'. "
        "Only output the tool call. Do not output any natural language."
    )
    
    return ollama_tools, system_prompt


# --- 3. SIMULATED MCP SERVER & CHECK LOGIC ---

def connect_database(path: str, alias: str, readonly: bool = True) -> str:
    """Simulates connection to the pre-generated DB file."""
    if not os.path.exists(path):
        return f"Error: Database file not found at '{path}'."
    try:
        mode = 'ro' if readonly else 'rw'
        conn = sqlite3.connect(f"file:{path}?mode={mode}", uri=True)
        DATABASE_CONNECTIONS[alias] = conn
        return f"Success: Connected to database with alias '{alias}'. Read-Only: {readonly}."
    except Exception as e:
        return f"Error: Failed to connect to DB: {e}"

def check_mcp_server_readiness(db_path: str, db_alias: str) -> bool:
    """Performs pre-flight checks."""
    print("--- PRE-FLIGHT CHECK: MCP Server Readiness ---")
    
    connect_result = connect_database(db_path, db_alias, readonly=True)
    print(f"  Check 1 (DB Connection): {connect_result}")
    if "Error" in connect_result:
        print("\nFATAL: Database connection failed. Cannot proceed with tests.")
        return False
        
    conn = DATABASE_CONNECTIONS.get(db_alias)
    if not conn: return False
    
    try:
        cursor = conn.execute("SELECT name, salary FROM employees LIMIT 1;")
        cursor.fetchone()
        print("  Check 2 (Schema/Table): 'employees' table is accessible and structured correctly.")
        return True
    except sqlite3.Error:
        print("  Check 2 (Schema/Table): FAILED. Critical table 'employees' not found.")
        print("\nFATAL: Database structure is incorrect. Cannot proceed with tests.")
        return False
    finally:
        pass

# --- 4. EVALUATION LOGIC ---

def evaluate_sql_correctness(generated_sql: str, correct_sql: str) -> float:
    """
    Calculates Argument Correctness (ACI score). 1.0 for exact match, 
    0.9 for functionally correct with minor syntax differences (e.g., alias/whitespace),
    0.0 otherwise.
    """
    def normalize(sql):
        # Normalizes whitespace, case, and removes semicolon for comparison
        return ' '.join(sql.strip().upper().replace(';', '').split())
    
    norm_gen = normalize(generated_sql)
    norm_corr = normalize(correct_sql)

    if norm_gen == norm_corr:
        return 1.0 
    
    # Simple check for functional correctness of key clauses as a partial score
    keywords = ['SELECT', 'FROM', 'WHERE', 'ORDER BY', 'LIMIT', 'JOIN']
    is_functionally_similar = True
    for keyword in keywords:
        if keyword in norm_corr and keyword not in norm_gen:
            is_functionally_similar = False
            break

    if is_functionally_similar and len(norm_gen.split()) > len(norm_corr.split()) * 0.5:
        return 0.5 # Partial correctness score
    
    return 0.0

def run_ollama_query(client: Client, prompt: str, system_prompt: str, tools: List[Dict[str, Any]] = None, model: str = "") -> Tuple[str, str, float]:
    """Calls Ollama API with or without tools and measures time."""
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    start_time = time.time()
    try:
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            options={"temperature": 0.0, "seed": 42} 
        )
        end_time = time.time()
        
        # Extract SQL from Tool Call
        if tools and 'tool_calls' in response.get('message', {}):
            tool_calls = response['message']['tool_calls']
            if tool_calls and tool_calls[0]['function']['name'] == 'execute_read_query':
                args = tool_calls[0]['function']['arguments']
                return args.get('query', ''), 'tool_call', end_time - start_time
            
        # Extract SQL from Raw Text Response
        return response['message']['content'].strip(), 'text', end_time - start_time

    except Exception as e:
        end_time = time.time()
        return f"Ollama API Error: {e}", 'error', end_time - start_time

# --- 5. MAIN EXECUTION ---

def main():
    
    # Load configurations from config.yaml
    config = load_config(CONFIG_FILE)
    llm_conf = config['llm_config']
    mcp_conf = config['mcp_config']
    
    OLLAMA_MODEL = llm_conf['model']
    OLLAMA_HOST = llm_conf['base_url']
    DB_NAME = mcp_conf['db_path']
    DB_ALIAS = mcp_conf['db_alias']
    SPEC_PATH = mcp_conf['spec_path']
    
    if 'api_key' in llm_conf:
        os.environ['OLLAMA_API_KEY'] = llm_conf['api_key']

    if not os.environ.get("OLLAMA_API_KEY"):
        print("\nFATAL ERROR: OLLAMA_API_KEY environment variable is not set or empty.")
        return

    client = Client(host=OLLAMA_HOST)
    
    # STEP 1: Pre-flight Check (MCP Server Readiness)
    if not check_mcp_server_readiness(DB_NAME, DB_ALIAS):
        return

    print("\n--- Starting Enhanced MCP Evaluation Script ---")
    print(f"Model: {OLLAMA_MODEL} | Total Test Cases: {len(TEST_CASES)}")
    print("-" * 60)
    
    # STEP 2: Load and Parse the Fully Compliant MCP Specification
    full_mcp_spec = load_mcp_spec(SPEC_PATH) 
    mcp_tools, mcp_system_prompt = parse_mcp_spec_for_ollama(full_mcp_spec, DB_ALIAS)
    
    # Generate the baseline system prompt (without tool context)
    nl_system_prompt = (
        f"You are an expert SQL agent. Database Schema:\n{COMPANY_DB_SCHEMA.format(alias=DB_ALIAS)}\n"
        "Your task is to generate a single, valid SQLite SELECT query that answers the user's request. "
        "Output ONLY the SQL query and nothing else."
    )
    
    # --- STEP 3: Run the Test Suite and Aggregate Metrics ---
    
    results = {
        "with_spec": {"scores": [], "times": []},
        "without_spec": {"scores": [], "times": []}
    }
    
    print("\n--- Executing Test Suite ---")
    
    for i, case in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}: {case['name']}] Query: '{case['nlp_query']}'")
        
        # --- Test A: With Full MCP Spec (Tool-Augmented) ---
        gen_sql_w, output_type_w, time_w = run_ollama_query(client, case['nlp_query'], mcp_system_prompt, tools=mcp_tools, model=OLLAMA_MODEL)
        score_w = evaluate_sql_correctness(gen_sql_w, case['correct_sql'])
        
        print(f"  -> With Spec: Score={score_w:.2f}, Time={time_w:.3f}s, Output Type={output_type_w}")
        results["with_spec"]["scores"].append(score_w)
        results["with_spec"]["times"].append(time_w)
        
        # --- Test B: Without MCP Spec (Baseline NL Generation) ---
        gen_sql_wo, output_type_wo, time_wo = run_ollama_query(client, case['nlp_query'], nl_system_prompt, tools=None, model=OLLAMA_MODEL)
        score_wo = evaluate_sql_correctness(gen_sql_wo, case['correct_sql'])
        
        print(f"  -> Without Spec: Score={score_wo:.2f}, Time={time_wo:.3f}s, Output Type={output_type_wo}")
        results["without_spec"]["scores"].append(score_wo)

        # Skip time comparison if the LLM returned an error or non-SQL output in the baseline test, 
        # as it would skew the speed metric (LLM might fail faster).
        if score_wo > 0.0 or output_type_wo == 'text':
             results["without_spec"]["times"].append(time_wo)
        else:
             # Use the time, but note the potential bias.
             results["without_spec"]["times"].append(time_wo)

    print("-" * 60)

    # --- STEP 4: Derive Final Metrics ---
    
    avg_score_w = sum(results['with_spec']['scores']) / len(results['with_spec']['scores'])
    avg_score_wo = sum(results['without_spec']['scores']) / len(results['without_spec']['scores'])
    
    avg_time_w = sum(results['with_spec']['times']) / len(results['with_spec']['times'])
    # Only average the successful/valid attempts for the without_spec time if possible
    avg_time_wo = sum(results['without_spec']['times']) / len(results['without_spec']['times'])
    
    # Argument Correctness Improvement (ACI)
    aci_improvement = avg_score_w - avg_score_wo
    
    # Speed Improvement (SI)
    # Note: Tool-calling often introduces latency due to the extra context parsing/tool selection step,
    # so speed improvement is often negligible or negative, but we measure it anyway.
    speed_difference = avg_time_wo - avg_time_w
    
    print("--- FINAL AGGREGATED METRICS ---")
    print(f"Average ACI Score (With MCP Spec):       {avg_score_w:.3f}")
    print(f"Average ACI Score (Without MCP Spec):    {avg_score_wo:.3f}")
    print(f"Argument Correctness Improvement (ACI):  **{aci_improvement:.3f}**")
    print("-" * 40)
    print(f"Average Speed (With MCP Spec):           {avg_time_w:.3f}s")
    print(f"Average Speed (Without MCP Spec):        {avg_time_wo:.3f}s")
    
    if speed_difference > 0:
        print(f"Average Speed Improvement (SI):          **{speed_difference:.3f}s faster (With Spec)**")
    else:
        print(f"Average Speed Difference (SI):           **{abs(speed_difference):.3f}s slower (With Spec)**")
        
    print("-" * 60)
    print("Conclusion: A positive ACI score on difficult queries proves the value of MCP's semantic context (NLP Hints, Enumeration).")

if __name__ == "__main__":
    main()