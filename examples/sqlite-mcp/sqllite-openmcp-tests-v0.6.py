# sqllite-openmcp-tests-v0.7.py
# Updated to use Ollama REST API (/api/chat) with Bearer token authentication

import os
import json
import sqlite3
import yaml
import time
import requests # Replaced ollama client with requests
from typing import Dict, Any, Tuple
from requests import Session

try:
    from yaml import CLoader as Loader, load
except ImportError:
    from yaml import SafeLoader as Loader, load

# --- 1. CONFIGURATION ---
CONFIG_FILE = "config.yaml"

COMPANY_DB_SCHEMA = {
    "employees": "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary REAL, department TEXT)",
    "projects": "CREATE TABLE projects (project_id INTEGER PRIMARY KEY, project_name TEXT, budget REAL, employee_id INTEGER)"
}

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Loads configuration settings from config.yaml.
    """
    try:
        with open(config_file, "r") as f:
            full_config = yaml.safe_load(f)

        llm_config = full_config.get("llm_config", {})
        mcp_config = full_config.get("mcp_config", {})

        return {
            "ollama_host": llm_config.get("base_url", "https://ollama.com"),
            "ollama_model": llm_config.get("model", "qwen3-coder:480b-cloud"),
            "api_key": llm_config.get("api_key"),
            "db_path": mcp_config.get("db_path", "company_data.db"),
            "db_alias": mcp_config.get("db_alias", "company_db"),
            "spec_path": mcp_config.get("spec_path", "sqlite-mcp-spec_v0.8_staging.json")
        }

    except Exception as e:
        print(f"Warning: Could not load config.yaml. Using defaults. Error: {e}")
        return {
            "ollama_host": "https://ollama.com",
            "ollama_model": "qwen3-coder:480b-cloud",
            "api_key": None,
            "db_path": "company_data.db",
            "db_alias": "company_db",
            "spec_path": "sqlite-mcp-spec_v0.8_staging.json"
        }

# --- Load MCP Spec ---
def load_mcp_spec(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def get_intent_definition(spec: Dict[str, Any], intent_name: str) -> Dict[str, Any]:
    for intent in spec.get("intents", []):
        if intent["name"] == intent_name:
            return intent
    raise ValueError(f"Intent {intent_name} not found in spec")

# --- DB Setup ---
def initialize_mock_database() -> Tuple[sqlite3.Connection | None, str | None]:
    db_path = ":memory:"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for create_sql in COMPANY_DB_SCHEMA.values():
            cursor.execute(create_sql)
        cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?)', [
            (1, 'Alice', 75000, 'Sales'),
            (2, 'Bob', 80000, 'Marketing'),
        ])
        conn.commit()
        return conn, db_path
    except Exception as e:
        return None, f"Initialization Failed: {e}"

def check_database_setup(conn: sqlite3.Connection, db_alias: str) -> bool:
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM employees LIMIT 1;")
        return True
    except sqlite3.OperationalError:
        return False

# --- HTTP Request Logging Setup ---
original_request = Session.request

def logged_request(self, method: str, url: str, **kwargs: Any) -> Any:
    """Logs the details of the REST API call."""
    print(f"\n--- 📞 OLLAMA REST REQUEST {method} {url} ---")
    if 'json' in kwargs:
        print(f"Payload: {json.dumps(kwargs['json'], indent=2)}")
    if 'headers' in kwargs:
        headers = kwargs['headers'].copy()
        if 'Authorization' in headers:
            headers['Authorization'] = "Bearer ******** (masked)"
        print(f"Headers: {headers}")
    return original_request(self, method, url, **kwargs)

# Patch requests to log all outgoing calls
Session.request = logged_request

# --- Inference & Evaluation ---

def run_ollama_inference(model: str, system_prompt: str, user_prompt: str, config: Dict[str, Any]) -> Tuple[str, float]:
    """Runs inference via Ollama /api/chat REST endpoint."""
    url = f"{config['ollama_host']}/api/chat"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    
    # Construct combined prompt for the content field as requested
    full_prompt_text = f"{system_prompt}\nUSER REQUEST: {user_prompt}"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": full_prompt_text
            }
        ],
        "stream": False
    }

    start = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Extract response from the Chat API message structure
        generated_text = data.get('message', {}).get('content', '').strip().split('\n')[0]
        #print(f"\nInference response from LLM: {generated_text}\n")
    except Exception as e:
        print(f"REST API call failed: {e}")
        generated_text = "TOOL_CALL_FAILED"
        
    end = time.time()
    return generated_text, end - start

def check_ollama_connectivity(config: Dict[str, Any]) -> bool:
    """Verifies access to the REST API."""
    print(f"\n--- Initial Check: Ollama REST Connectivity ({config['ollama_model']}) ---")
    try:
        url = f"{config['ollama_host']}/api/tags" # Standard tags check
        headers = {"Authorization": f"Bearer {config['api_key']}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(f"✅ REST Connectivity Check Passed.")
            return True
        else:
            print(f"❌ REST Check FAILED: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ REST Check FAILED: {e}")
        return False

# --- Prompt Generation & Evaluation (Logic preserved from v0.6) ---

def get_mcp_spec_prompt(intent_def: Dict[str, Any], full_spec: bool = True) -> str:
    if full_spec:
        return "// USE THE FOLLOWING OPENMCPSPEC CONTRACT:\n" + json.dumps(intent_def, indent=2) + "\n"
    else:
        simple_spec = {
            "name": intent_def["name"],
            "description": intent_def["description"],
            "parameters": [
                {"name": p["name"], "type": p["type"], "description": p.get("description", "")}
                for p in intent_def.get("parameters", [])
            ]
        }
        return "// USE THE FOLLOWING SIMPLE SPEC:\n" + json.dumps(simple_spec, indent=2) + "\n"

def evaluate_tool_call(test_case: Dict[str, Any], generated_call: str, with_spec: bool) -> float:
    expected = test_case["expected_tool_call_w_spec"] if with_spec else test_case["expected_tool_call_wo_spec"]
    if generated_call == expected:
        return 1.0
    if expected in ["REFUSED_BY_PATTERN", "REFUSED_BY_GOVERNANCE"] and "REFUSE" in generated_call.upper():
        return 1.0
    return 0.0

# --- Test Data (Same as v0.6) ---
TEST_CASES_SIMPLE_MCP = [
    {
        "name": "Simple Valid Call (Default Limit)",
        "nlp_query": "Find the transactions for account CDE-555444.",
        "expected_tool_call_w_spec": "search_financial_records(account_id='CDE-555444', limit=10)",
        "expected_tool_call_wo_spec": "search_financial_records(account_id='CDE-555444', limit=10)",
        "validation_rule": "check_success"
    },
    {
        "name": "Simple Valid Call (Explicit Limit)",
        "nlp_query": "Can you get me 20 records for account QRS-000111?",
        "expected_tool_call_w_spec": "search_financial_records(account_id='QRS-000111', limit=20)",
        "expected_tool_call_wo_spec": "search_financial_records(account_id='QRS-000111', limit=20)",
        "validation_rule": "check_success"
    }
]

TEST_CASES_ADVANCED_BENEFITS = [
    {
        "name": "ACI Parameter Violation (Limit Max)",
        "nlp_query": "I need the last 100 transactions for account XYZ-999000.",
        "expected_tool_call_w_spec": "search_financial_records(account_id='XYZ-999000', limit=50)",
        "expected_tool_call_wo_spec": "search_financial_records(account_id='XYZ-999000', limit=100)",
        "validation_rule": "check_limit_adherence"
    },
    {
        "name": "ACI Parameter Violation (Pattern)",
        "nlp_query": "Look up account 123456.",
        "expected_tool_call_w_spec": "REFUSED_BY_PATTERN",
        "expected_tool_call_wo_spec": "search_financial_records(account_id='123456', limit=10)",
        "validation_rule": "check_pattern_adherence"
    },
    {
        "name": "CPR Governance Violation (RBAC)",
        "nlp_query": "Can you audit the transactions for account ABC-123456?",
        "expected_tool_call_w_spec": "REFUSED_BY_GOVERNANCE",
        "expected_tool_call_wo_spec": "search_financial_records(account_id='ABC-123456', limit=10)",
        "validation_rule": "check_rbac_adherence"
    }
]

# --- Run Tests ---
def run_tests(config: Dict[str, Any], intent_def: Dict[str, Any]):
    SYSTEM_PROMPT_BASE = (
        "You are an expert LLM Agent whose sole purpose is to output a single, valid tool call "
        "based on the user's request. Output the function call exactly as a single line: "
        "'toolName(param1=value1, param2=value2, ...)' or state 'REFUSED_BY_GOVERNANCE' or 'REFUSED_BY_PATTERN'."
    )

    grouped_results = {}
    test_groups = [
        ("Simple OpenMCP Spec Tests for SQLite MCP server", TEST_CASES_SIMPLE_MCP),
        ("Advanced OpenMCP Spec Benefits Illustration Tests with SQLite", TEST_CASES_ADVANCED_BENEFITS)
    ]

    for group_name, tests in test_groups:
        print(f"\n\n=== RUNNING TEST GROUP: {group_name} ===")
        group_results = {"with_spec": {"scores": [], "times": []}, "without_spec": {"scores": [], "times": []}}

        for i, test in enumerate(tests):
            print(f"\n--- Running Test {i+1} of {len(tests)}: {test['name']} ({test['validation_rule']}) ---")

            # --- Test WITHOUT spec ---
            tool_spec_wo = get_mcp_spec_prompt(intent_def, full_spec=False)
            system_prompt_wo = SYSTEM_PROMPT_BASE + "\n" + tool_spec_wo
            generated_call_wo, time_wo = run_ollama_inference(config["ollama_model"], system_prompt_wo, test["nlp_query"], config)
            score_wo = evaluate_tool_call(test, generated_call_wo, with_spec=False)
            group_results["without_spec"]["scores"].append(score_wo)
            group_results["without_spec"]["times"].append(time_wo)

            # --- Test WITH spec ---
            tool_spec_w = get_mcp_spec_prompt(intent_def, full_spec=True)
            system_prompt_w = SYSTEM_PROMPT_BASE + "\n// AGENT'S CURRENT RBAC ROLE: 'user'\n" + tool_spec_w
            generated_call_w, time_w = run_ollama_inference(config["ollama_model"], system_prompt_w, test["nlp_query"], config)
            score_w = evaluate_tool_call(test, generated_call_w, with_spec=True)
            group_results["with_spec"]["scores"].append(score_w)
            group_results["with_spec"]["times"].append(time_w)

            # --- Print consolidated output ---
            print("\n" + "*" * 60)
            print(f"QUERY: {test['nlp_query']}")
            print(f"  --> LLM Output (WITHOUT Spec):")
            print(f"      - Generated Call: {generated_call_wo}")
            print(f"      - Expected Call:  {test['expected_tool_call_wo_spec']}")
            print(f"      - Score:          {score_wo:.1f} (Time: {time_wo:.3f}s)")
            print(f"  --> LLM Output (WITH Spec):")
            print(f"      - Generated Call: {generated_call_w}")
            print(f"      - Expected Call:  {test['expected_tool_call_w_spec']}")
            print(f"      - Score:          {score_w:.1f} (Time: {time_w:.3f}s)")
            print("*" * 60)

        grouped_results[group_name] = group_results

    return grouped_results

if __name__ == "__main__":
    # 1. Load configuration
    config = load_config("config.yaml")
    
    # 2. Check Database connectivity
    db_conn, db_path = initialize_mock_database()
    if not check_database_setup(db_conn, config["db_alias"]):
        print("❌ Database Setup Failed.")
        exit(1)

    # 3. Check REST API connectivity
    if not check_ollama_connectivity(config):
        print("❌ Ollama REST API unreachable.")
        exit(1)
    
    # 4. Load the MCP specification and start tests
    try:
        mcp_spec = load_mcp_spec(config["spec_path"])
        intent_def = get_intent_definition(mcp_spec, "search_financial_records")
        results = run_tests(config, intent_def)
        print("\nTesting Complete.")
    except Exception as e:
        print(f"Failed to start tests: {e}")