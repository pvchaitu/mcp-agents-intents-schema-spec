import os
import json
import sqlite3
import yaml
from typing import Dict, Any, List

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

# --- 1. TEST CASE CONSTANTS (Define the test being run) ---

# This specific test case query and expected SQL must remain hardcoded, 
# as they define the metric being calculated, not the tool configuration.
TEST_NLP_HINT = "Find the name and salary of all employees in the 'Sales' department who earn over $70,000."
CORRECT_SQL = "SELECT name, salary FROM employees WHERE department = 'Sales' AND salary > 70000;"

# Dictionary to hold the state of connected databases (Simulated MCP Server state)
DATABASE_CONNECTIONS = {}

# Schema information for LLM prompting
COMPANY_DB_SCHEMA = (
    "The database alias '{alias}' contains two tables:\n"
    "1. employees: (id, name, salary, department)\n"
    "2. projects: (project_id, project_name, budget, employee_id)"
)

CONFIG_FILE = "config.yaml"

# --- 2. CONFIGURATION AND SPECIFICATION LOADING ---

def load_config(file_path: str) -> Dict[str, Any]:
    """Loads configuration from the specified YAML file."""
    try:
        with open(file_path, 'r') as f:
            config = load(f, Loader=Loader)
        print(f"Configuration loaded from {file_path}.")
        return config
    except FileNotFoundError:
        print(f"FATAL ERROR: Configuration file not found at '{file_path}'.")
        exit()
    except Exception as e:
        print(f"FATAL ERROR: Failed to load or parse YAML config: {e}")
        exit()

def load_mcp_spec(file_path: str) -> Dict[str, Any]:
    """
    Loads the MCP spec JSON directly without injecting hardcoded data.
    The spec is assumed to be fully compliant (containing enumeration and metadata).
    """
    try:
        with open(file_path, 'r') as f:
            spec = json.load(f)
        
        # NOTE: The provided spec is already in the array format and includes metadata.
        print(f"MCP Specification loaded directly from {file_path}. No enhancement needed.")
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
    
    # Extract Tool Definitions
    # The intents array already contains the required metadata
    for intent in mcp_spec.get('intents', []):
        # We focus on the core intents for this test
        if intent['name'] in ['execute_read_query', 'connect_database', 'get_catalog']:
            
            # The spec uses JSON Schema parameters, which align with Ollama/OpenAI
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
                # Use intent category if available
                category = intent.get('category', 'Tool') 
                nlp_hint_list.append(f"- {intent['name']} ({category}): {', '.join(hints)}")


    # Generate Enhanced System Prompt using Enumeration data from the spec
    enum = mcp_spec.get('enumeration', {})
    capabilities = ", ".join(enum.get('capability_tags', []))
    # Filter roles to include only those relevant to the agent's LLM context
    roles = ", ".join(enum.get('filters', {}).get('roles', [])) 

    system_prompt = (
        f"You are an expert SQL agent (Capabilities: {capabilities}) "
        f"with the following roles: {roles}.\n"
        f"Database Schema:\n{COMPANY_DB_SCHEMA.format(alias=db_alias)}\n\n"
        "Your task is to analyze the user's request and respond by calling the most relevant tool.\n"
        "**Intent Recognition Guidance (NLP Hints for better ACI):**\n"
        f"{'\\n'.join(nlp_hint_list)}\n\n"
        f"When the request is for data retrieval (e.g., '{TEST_NLP_HINT}'), you MUST call the 'execute_read_query' tool "
        f"with a single, valid SQLite SELECT query for the database alias '{db_alias}'. "
        "Only output the tool call. Do not output any natural language."
    )
    
    return ollama_tools, system_prompt


# --- 3. SIMULATED MCP SERVER & EVALUATION LOGIC ---

def connect_database(path: str, alias: str, readonly: bool = True) -> str:
    """Simulates connection to the pre-generated DB file."""
    if not os.path.exists(path):
        return f"Error: Database file not found at '{path}'."
    try:
        # Use URI mode for read-only enforcement
        mode = 'ro' if readonly else 'rw'
        conn = sqlite3.connect(f"file:{path}?mode={mode}", uri=True)
        DATABASE_CONNECTIONS[alias] = conn
        return f"Success: Connected to database with alias '{alias}'. Read-Only: {readonly}."
    except Exception as e:
        return f"Error: Failed to connect to DB: {e}"

def check_mcp_server_readiness(db_path: str, db_alias: str) -> bool:
    """
    Performs pre-flight checks to ensure the SQLite DB is loadable and structured.
    This simulates the MCP server being 'up and running' and connected to the context.
    """
    print("--- PRE-FLIGHT CHECK: MCP Server Readiness ---")
    
    # 1. Check database connection
    connect_result = connect_database(db_path, db_alias, readonly=True)
    print(f"  Check 1 (DB Connection): {connect_result}")
    if "Error" in connect_result:
        print("\nFATAL: Database connection failed. Cannot proceed with tests.")
        print("Please ensure you have run 'gen-company-db.py' to create the database file.")
        return False
        
    # 2. Check essential table structure (simulates server confirming context)
    conn = DATABASE_CONNECTIONS.get(db_alias)
    if not conn: return False
    
    try:
        cursor = conn.execute("SELECT name, salary FROM employees LIMIT 1;")
        cursor.fetchone()
        print("  Check 2 (Schema/Table): 'employees' table is accessible and structured correctly.")
        return True
    except sqlite3.Error as e:
        print(f"  Check 2 (Schema/Table): FAILED. Critical table 'employees' not found or inaccessible.")
        print(f"  SQLite Error: {e}")
        print("\nFATAL: Database structure is incorrect. Cannot proceed with tests.")
        return False
    finally:
        pass # Keep connection for the global dict

def evaluate_sql_correctness(generated_sql: str, correct_sql: str) -> float:
    """Simplified Argument Correctness Metric."""
    def normalize(sql):
        return ' '.join(sql.strip().upper().replace(';', '').split())
    norm_gen = normalize(generated_sql)
    norm_corr = normalize(correct_sql)

    if norm_gen == norm_corr:
        return 1.0 
    
    # Check for argument correctness of key clauses 
    if ('SELECT NAME, SALARY' in norm_gen) and \
       ('FROM EMPLOYEES' in norm_gen) and \
       ("DEPARTMENT = 'SALES'" in norm_gen) and \
       ("SALARY > 70000" in norm_gen):
        return 0.9 

    return 0.0

def run_ollama_query(client: Client, prompt: str, system_prompt: str, tools: List[Dict[str, Any]] = None, model: str = "") -> tuple[str, str]:
    """Calls Ollama API with or without tools."""
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    try:
        print("  ... Sending request to Ollama API ...")
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            options={"temperature": 0.0, "seed": 42} 
        )
        
        # Extract SQL from Tool Call
        if tools and 'tool_calls' in response.get('message', {}):
            tool_calls = response['message']['tool_calls']
            if tool_calls and tool_calls[0]['function']['name'] == 'execute_read_query':
                args = tool_calls[0]['function']['arguments']
                return args.get('query', ''), 'tool_call'
            
        # Extract SQL from Raw Text Response
        return response['message']['content'].strip(), 'text'

    except Exception as e:
        return f"Ollama API Error: {e}", 'error'

# --- 4. MAIN EXECUTION ---

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
    
    # Set the OLLAMA_API_KEY from config to environment variable
    if 'api_key' in llm_conf:
        os.environ['OLLAMA_API_KEY'] = llm_conf['api_key']

    if not os.environ.get("OLLAMA_API_KEY"):
        print("\nFATAL ERROR: OLLAMA_API_KEY environment variable is not set or empty.")
        return

    client = Client(host=OLLAMA_HOST)
    
    # STEP 1: Pre-flight Check (MCP Server Readiness)
    if not check_mcp_server_readiness(DB_NAME, DB_ALIAS):
        return

    print("\n--- Starting MCP SQLite Evaluation Script ---")
    print(f"Model: {OLLAMA_MODEL} | DB Alias: {DB_ALIAS} | Spec: {SPEC_PATH}")
    print("-" * 60)
    
    # STEP 2: Load the Fully Compliant MCP Specification
    print("STEP 2: Loading and Parsing Fully Compliant Open MCP Spec...")
    # This now loads the spec directly, assuming it contains all necessary metadata
    full_mcp_spec = load_mcp_spec(SPEC_PATH) 
    mcp_tools, mcp_system_prompt = parse_mcp_spec_for_ollama(full_mcp_spec, DB_ALIAS)
    print(f"Successfully extracted {len(mcp_tools)} tools and generated ACI-optimized system prompt (Metadata read directly from spec).")
    
    print("-" * 60)

    # --- Test 1: With Full MCP Spec (Enhanced ACI) ---
    print("STEP 3: Test 1 - With Full MCP Spec (Tool-Augmented Evaluation)")
    
    gen_sql_with_tools, output_type = run_ollama_query(client, TEST_NLP_HINT, mcp_system_prompt, tools=mcp_tools, model=OLLAMA_MODEL)

    if output_type == 'tool_call':
        print(f"  Result: LLM successfully generated a Tool Call.")
        print(f"  Generated SQL Argument: {gen_sql_with_tools}")
    else:
        print(f"  Raw Output: {gen_sql_with_tools}")
        
    score_with_tools = evaluate_sql_correctness(gen_sql_with_tools, CORRECT_SQL)
    print(f"  -> Argument Correctness Score (With Enhanced MCP Spec): {score_with_tools:.2f}")
    
    print("-" * 60)

    # --- Test 2: Without MCP Spec (Baseline ACI) ---
    print("STEP 4: Test 2 - Without MCP Spec (Standard NL Generation Baseline)")
    
    # Minimal system prompt (no tools provided)
    nl_system_prompt = (
        f"You are an expert SQL agent. Database Schema:\n{COMPANY_DB_SCHEMA.format(alias=DB_ALIAS)}\n"
        "Your task is to generate a single, valid SQLite SELECT query that answers the user's request. "
        "Output ONLY the SQL query and nothing else."
    )
    
    gen_sql_without_tools, output_type = run_ollama_query(client, TEST_NLP_HINT, nl_system_prompt, tools=None, model=OLLAMA_MODEL)
    
    if output_type == 'text':
        print(f"  Result: LLM successfully generated a raw text query.")
        print(f"  Generated SQL Raw Output: {gen_sql_without_tools}")
    else:
        print(f"  Raw Output: {gen_sql_without_tools}")

    score_without_tools = evaluate_sql_correctness(gen_sql_without_tools, CORRECT_SQL)
    print(f"  -> Argument Correctness Score (Without MCP Spec Baseline): {score_without_tools:.2f}")
    
    print("-" * 60)

    # --- STEP 5: Derive the Metric ---
    print("STEP 5: Deriving Argument Correctness Improvement Metric")
    
    argument_correctness_improvement_metric = score_with_tools - score_without_tools
    
    print(f"Score with Enhanced MCP Tools: {score_with_tools:.2f}")
    print(f"Score without MCP Baseline:    {score_without_tools:.2f}")
    print(f"\nArgument Correctness Improvement (ACI): **{argument_correctness_improvement_metric:.2f}**")
    
    print("-" * 60)
    print("Conclusion: A positive ACI score demonstrates the quantifiable benefit of providing the LLM with "
          "structured tool definitions and rich context (NLP Hints, Enumeration metadata) from the Open MCP specification.")

if __name__ == "__main__":
    main()