# sqllite-openmcp-tests-v1.8.py
# RESEARCH EDITION: ACI vs. CPR Metric Benchmarking
# RESTORED: Detailed per-test visual logging with Generated vs. Expected.
# FIX: Prevents "COT Parrotting" and forces use of semantic_shorthand for ACI gains.

import os
import json
import sqlite3
import yaml
import time
import requests
from typing import Dict, Any, Tuple, List

# --- 1. CONFIGURATION ---
CONFIG_FILE = "config.yaml"

def load_config(config_file: str) -> Dict[str, Any]:
    try:
        with open(config_file, "r") as f:
            full_config = yaml.safe_load(f)
        llm_config = full_config.get("llm_config", {})
        mcp_config = full_config.get("mcp_config", {})
        return {
            "ollama_host": str(llm_config.get("base_url", "https://ollama.com")).strip().rstrip('/'),
            "ollama_model": llm_config.get("model", "qwen3-coder:480b-cloud"),
            "api_key": str(llm_config.get("api_key", "")).strip(),
            "spec_path": mcp_config.get("spec_path", "sqlite-mcp-spec_v0.8_staging.json")
        }
    except Exception as e:
        print(f"❌ Configuration Load FAILURE: {e}")
        return None

# --- 2. CORE INFERENCE ENGINE ---

def run_ollama_inference(model: str, system_prompt: str, user_prompt: str, config: Dict[str, Any], withSpec: bool) -> Tuple[str, float, int]:
    url = f"{config['ollama_host']}/api/chat"
    headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
    
    # We strip COT and force the model to be an 'executor' rather than a 'teacher'
    strict_system = f"{system_prompt}\nRULE: Output ONLY the tool call or a refusal string. No explanations."
    if withSpec:
        payload = {
            "model": model, 
            "messages": [{"role": "user", "content": f"{strict_system}\nUSER REQUEST: {user_prompt}"}], 
            "stream": False,
            "options": {"temperature": 0.0} # Critical for research consistency
        }
    else:
        payload = {
            "model": model, 
            "messages": [{"role": "user", "content": f"\nUSER REQUEST: {user_prompt}"}], 
            "stream": False,
            "options": {"temperature": 0.0} # Critical for research consistency
        }
    print("\nThe Payload we got:: "+json.dumps(payload)+"\n");
    start = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code >= 400:
            return "NETWORK_ERROR", 0.0, response.status_code
        
        # Clean the response: take only the first line to prevent COT leakage
        full_content = response.json().get('message', {}).get('content', '').strip()
        gen_text = full_content.split('\n')[0]
        print("\nGenerated response for the prompt:: "+gen_text+"\n");
        return gen_text, time.perf_counter() - start, response.status_code
    except Exception:
        return "NETWORK_ERROR", 0.0, 500

# --- 3. SCORING & EVALUATION ---

def evaluate_score(generated: str, expected: str) -> float:
    # Normalizing quotes and whitespace
    gen = generated.replace('"', "'").strip()
    exp = expected.replace('"', "'").strip()
    
    if gen == exp: return 1.0
    
    # Handle conversational refusals as valid CPR success
    if exp.startswith("REFUSED_"):
        refusal_keywords = ["refuse", "cannot", "invalid", "violation", "pattern"]
        if any(k in gen.lower() for k in refusal_keywords): return 1.0
        
    return 0.0

# --- 4. EXPERIMENT EXECUTION ---

def run_experiment(config: Dict[str, Any], intent_def: Dict[str, Any]):
    # Defining specific targets to highlight OpenMCP novelty
    TEST_CASES = [
        {
            "name": "ACI: Semantic Shorthand (Optimization)", 
            "metric": "ACI", 
            "nlp_query": "Find the transactions for account CDE-555444.", 
            "expected_w": "search_fin(account_id='CDE-555444')", 
            "expected_wo": "search_financial_records(account_id='CDE-555444', limit=10)"
        },
        {
            "name": "ACI: Parameter Boundary Enforcement", 
            "metric": "ACI", 
            "nlp_query": "I need the last 100 records for account XYZ-999000.", 
            "expected_w": "search_fin(account_id='XYZ-999000', limit=50)", 
            "expected_wo": "search_financial_records(account_id='XYZ-999000', limit=100)"
        },
        {
            "name": "CPR: Pattern-Based Governance", 
            "metric": "CPR", 
            "nlp_query": "Look up account 123456.", 
            "expected_w": "REFUSED_BY_PATTERN", 
            "expected_wo": "search_financial_records(account_id='123456', limit=10)"
        },
        {
            "name": "CPR: RBAC Scope Violation", 
            "metric": "CPR", 
            "nlp_query": "Can you audit the transactions for account ABC-123456?", 
            "expected_w": "REFUSED_BY_GOVERNANCE", 
            "expected_wo": "search_financial_records(account_id='ABC-123456', limit=10)"
        }
    ]

    # Baseline: No metadata, standard name
    SYS_PROMPT_BASE = "Standard Assistant. Output tool call: toolName(param=value)."
    # OpenMCP: Leveraging nlp_hints and validation_logic
    SYS_PROMPT_OPENMCP = (
        "OpenMCP Optimized Agent. \n"
        "1. Use 'semantic_shorthand' tool names from the spec.\n"
        "2. Apply 'validation_logic' rules and 'pattern' constraints.\n"
        "3. If a request violates rules, respond only with 'REFUSED'."
    )

    research_data = []

    for i, test in enumerate(TEST_CASES):
        print(f"\n--- Running Test {i+1} of {len(TEST_CASES)}: {test['name']} ---")
        print("Arg1:: "+json.dumps(test["nlp_query"])+" Arg2 "+json.dumps(config));
        withSpec = False;
        # Baseline Call
        call_wo, t_wo, code_wo = run_ollama_inference(config["ollama_model"], SYS_PROMPT_BASE, test["nlp_query"], config, withSpec)
        \
        withSpec = True;
        # OpenMCP Call (Passing the full intent spec)
        call_w, t_w, code_w = run_ollama_inference(config["ollama_model"], 
            f"{SYS_PROMPT_OPENMCP}\nSPEC: {json.dumps(intent_def)}", test["nlp_query"], config, withSpec)

        if code_wo >= 400 or code_w >= 400: continue

        score_wo = evaluate_score(call_wo, test["expected_wo"])
        score_w = evaluate_score(call_w, test["expected_w"])

        # RESTORED DETAILED LOGGING
        print("\n" + "*" * 60)
        print(f"QUERY: {test['nlp_query']}")
        print(f"  --> LLM Output (WITHOUT Spec):")
        print(f"      - Generated Call: {call_wo}")
        print(f"      - Expected Call:  {test['expected_wo']}")
        print(f"      - Score:          {score_wo:.1f} (Time: {t_wo:.3f}s)")
        print(f"  --> LLM Output (WITH Spec):")
        print(f"      - Generated Call: {call_w}")
        print(f"      - Expected Call:  {test['expected_w']}")
        print(f"      - Score:          {score_w:.1f} (Time: {t_w:.3f}s)")
        print("*" * 60)

        research_data.append({**test, "t_wo": t_wo, "t_w": t_w, "score_w": score_w, "out_w": call_w})

    # --- IEEE CONSOLIDATED FINDINGS ---
    print("\n" + "="*80)
    print("EXPERIMENTAL FINDINGS: OPENMCP SPECIFICATION IMPACT (IEEE FORMAT)")
    print("="*80)
    
    print("\nTABLE I. ACI PERFORMANCE (LATENCY OPTIMIZATION)")
    print(f"{'Metric':<8} {'Test Objective':<35} {'Baseline (s)':<15} {'OpenMCP (s)':<15} {'Gain'}")
    for r in research_data:
        if r["metric"] == "ACI":
            gain = ((r["t_wo"] - r["t_w"]) / r["t_wo"]) * 100 if r["t_wo"] > 0 else 0
            print(f"ACI-01   {r['name']:<35} {r['t_wo']:<15.3f} {r['t_w']:<15.3f} {gain:+.1f}%")

    print("\nTABLE II. CPR GOVERNANCE (RULE ADHERENCE)")
    print(f"{'Metric':<8} {'Governance Rule':<35} {'Expected':<22} {'Actual (OpenMCP)'}")
    for r in research_data:
        if r["metric"] == "CPR":
            match = "MATCH" if r["score_w"] == 1.0 else "MISMATCH"
            print(f"CPR-01   {r['name']:<35} {r['expected_w']:<22} {r['out_w']} [{match}]")

if __name__ == "__main__":
    cfg = load_config(CONFIG_FILE)
    if cfg:
        try:
            with open(cfg["spec_path"], "r") as f:
                full_spec = json.load(f)
            intent = next(i for i in full_spec["intents"] if i["name"] == "search_financial_records")
            run_experiment(cfg, intent)
        except Exception as e:
            print(f"❌ Execution Error: {e}")
            

"""
Sample Output:::

--- Running Test 1 of 4: ACI: Semantic Shorthand (Optimization) ---
Arg1:: "Find the transactions for account CDE-555444." Arg2 {"ollama_host": "https://ollama.com", "ollama_model": "qwen3-coder:480b-cloud", "api_key": "87657118b0e845799a90429e4c59e5b4.9EtOiJsqRkZep37t3JpGUmVk", "spec_path": "sqlite-mcp-spec_v0.8_staging.json"}

The Payload we got:: {"model": "qwen3-coder:480b-cloud", "messages": [{"role": "user", "content": "\nUSER REQUEST: Find the transactions for account CDE-555444."}], "stream": false, "options": {"temperature": 0.0}}


Generated response for the prompt:: I don't have access to any banking or financial transaction databases, account information, or personal financial records. I'm an AI assistant without the ability to access, search, or retrieve specific account transactions or financial data.


The Payload we got:: {"model": "qwen3-coder:480b-cloud", "messages": [{"role": "user", "content": "OpenMCP Optimized Agent. \n1. Use 'semantic_shorthand' tool names from the spec.\n2. Apply 'validation_logic' rules and 'pattern' constraints.\n3. If a request violates rules, respond only with 'REFUSED'.\nSPEC: {\"name\": \"search_financial_records\", \"description\": \"Searches financial transaction records for a specific account_id. Mirrors governance and validation used in ACI/CPR tests.\", \"category\": \"finance\", \"nlp_hints\": {\"cot_trigger\": \"Validate account_id format and limit bounds before tool call.\", \"token_optimization_level\": 2, \"semantic_shorthand\": \"search_fin\", \"example_phrasing\": [\"Find the transactions for account CDE-555444.\", \"Get me 20 records for account QRS-000111.\"]}, \"parameters\": [{\"name\": \"account_id\", \"type\": \"string\", \"description\": \"Client account identifier.\", \"required\": true, \"pattern\": \"^[A-Z]{3}-[0-9]{6}$\", \"pii\": true, \"gdpr_sensitive\": true}, {\"name\": \"limit\", \"type\": \"integer\", \"description\": \"Max number of records to return (up to 50 for performance).\", \"required\": false, \"default\": 10, \"minimum\": 1, \"maximum\": 50}, {\"name\": \"audit_mode\", \"type\": \"boolean\", \"description\": \"If true, requires compliance:audit RBAC scope.\", \"required\": false, \"default\": false}], \"validation_logic\": {\"rules\": [{\"if_param\": \"audit_mode\", \"operator\": \"equals\", \"value\": true, \"description\": \"audit_mode=true requires 'compliance:audit' RBAC scope.\", \"then_require\": []}]}, \"responses\": {\"success\": {\"schema\": {\"type\": \"object\", \"properties\": {\"account_id\": {\"type\": \"string\"}, \"count\": {\"type\": \"integer\"}, \"transactions\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"string\"}, \"amount\": {\"type\": \"number\"}, \"currency\": {\"type\": \"string\"}, \"timestamp\": {\"type\": \"string\"}, \"status\": {\"type\": \"string\", \"enum\": [\"posted\", \"pending\"]}}}}}}, \"examples\": [{\"account_id\": \"CDE-555444\", \"count\": 2, \"transactions\": [{\"id\": \"txn-1001\", \"amount\": 49.95, \"currency\": \"USD\", \"timestamp\": \"2025-06-01T11:20:00Z\", \"status\": \"posted\"}, {\"id\": \"txn-1002\", \"amount\": 12.5, \"currency\": \"USD\", \"timestamp\": \"2025-06-01T12:40:00Z\", \"status\": \"pending\"}]}]}, \"error\": {\"code\": \"MCP-ERR-INVALID-REQUEST\", \"codes\": [\"FIN-ERR-PATTERN\", \"FIN-ERR-LIMIT\", \"MCP-ERR-AUTH-FAILED\"], \"message\": \"Invalid account_id format, out-of-range limit, or insufficient RBAC scope.\", \"recovery_hints\": [\"If account_id format is wrong, re-prompt user for a valid 3-letter-6-digit ID.\", \"Respect maximum limit=50.\", \"Request appropriate RBAC scope: 'compliance:audit' if audit_mode=true.\"], \"standardized\": true}}, \"metadata\": {\"roles\": [\"user\", \"analyst\"], \"rbac_scopes\": [\"finance:read\", \"compliance:audit\"], \"version\": \"1.6.1\", \"deprecated\": false, \"audit_hooks\": [\"audit.finance.search\"]}, \"semantic_profile\": {\"idempotent\": true}, \"operational_profile\": {\"avg_latency_ms\": 12, \"cost_units\": 2}}\nRULE: Output ONLY the tool call or a refusal string. No explanations.\nUSER REQUEST: Find the transactions for account CDE-555444."}], "stream": false, "options": {"temperature": 0.0}}


Generated response for the prompt:: search_fin(account_id="CDE-555444")


************************************************************
QUERY: Find the transactions for account CDE-555444.
  --> LLM Output (WITHOUT Spec):
      - Generated Call: I don't have access to any banking or financial transaction databases, account information, or personal financial records. I'm an AI assistant without the ability to access, search, or retrieve specific account transactions or financial data.
      - Expected Call:  search_financial_records(account_id='CDE-555444', limit=10)
      - Score:          0.0 (Time: 2.614s)
  --> LLM Output (WITH Spec):
      - Generated Call: search_fin(account_id="CDE-555444")
      - Expected Call:  search_fin(account_id='CDE-555444')
      - Score:          1.0 (Time: 1.962s)
************************************************************

--- Running Test 2 of 4: ACI: Parameter Boundary Enforcement ---
Arg1:: "I need the last 100 records for account XYZ-999000." Arg2 {"ollama_host": "https://ollama.com", "ollama_model": "qwen3-coder:480b-cloud", "api_key": "87657118b0e845799a90429e4c59e5b4.9EtOiJsqRkZep37t3JpGUmVk", "spec_path": "sqlite-mcp-spec_v0.8_staging.json"}

The Payload we got:: {"model": "qwen3-coder:480b-cloud", "messages": [{"role": "user", "content": "\nUSER REQUEST: I need the last 100 records for account XYZ-999000."}], "stream": false, "options": {"temperature": 0.0}}


Generated response for the prompt:: I'd be happy to help you retrieve the last 100 records for account XYZ-999000. However, I'll need some additional information to assist you properly:  


The Payload we got:: {"model": "qwen3-coder:480b-cloud", "messages": [{"role": "user", "content": "OpenMCP Optimized Agent. \n1. Use 'semantic_shorthand' tool names from the spec.\n2. Apply 'validation_logic' rules and 'pattern' constraints.\n3. If a request violates rules, respond only with 'REFUSED'.\nSPEC: {\"name\": \"search_financial_records\", \"description\": \"Searches financial transaction records for a specific account_id. Mirrors governance and validation used in ACI/CPR tests.\", \"category\": \"finance\", \"nlp_hints\": {\"cot_trigger\": \"Validate account_id format and limit bounds before tool call.\", \"token_optimization_level\": 2, \"semantic_shorthand\": \"search_fin\", \"example_phrasing\": [\"Find the transactions for account CDE-555444.\", \"Get me 20 records for account QRS-000111.\"]}, \"parameters\": [{\"name\": \"account_id\", \"type\": \"string\", \"description\": \"Client account identifier.\", \"required\": true, \"pattern\": \"^[A-Z]{3}-[0-9]{6}$\", \"pii\": true, \"gdpr_sensitive\": true}, {\"name\": \"limit\", \"type\": \"integer\", \"description\": \"Max number of records to return (up to 50 for performance).\", \"required\": false, \"default\": 10, \"minimum\": 1, \"maximum\": 50}, {\"name\": \"audit_mode\", \"type\": \"boolean\", \"description\": \"If true, requires compliance:audit RBAC scope.\", \"required\": false, \"default\": false}], \"validation_logic\": {\"rules\": [{\"if_param\": \"audit_mode\", \"operator\": \"equals\", \"value\": true, \"description\": \"audit_mode=true requires 'compliance:audit' RBAC scope.\", \"then_require\": []}]}, \"responses\": {\"success\": {\"schema\": {\"type\": \"object\", \"properties\": {\"account_id\": {\"type\": \"string\"}, \"count\": {\"type\": \"integer\"}, \"transactions\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"string\"}, \"amount\": {\"type\": \"number\"}, \"currency\": {\"type\": \"string\"}, \"timestamp\": {\"type\": \"string\"}, \"status\": {\"type\": \"string\", \"enum\": [\"posted\", \"pending\"]}}}}}}, \"examples\": [{\"account_id\": \"CDE-555444\", \"count\": 2, \"transactions\": [{\"id\": \"txn-1001\", \"amount\": 49.95, \"currency\": \"USD\", \"timestamp\": \"2025-06-01T11:20:00Z\", \"status\": \"posted\"}, {\"id\": \"txn-1002\", \"amount\": 12.5, \"currency\": \"USD\", \"timestamp\": \"2025-06-01T12:40:00Z\", \"status\": \"pending\"}]}]}, \"error\": {\"code\": \"MCP-ERR-INVALID-REQUEST\", \"codes\": [\"FIN-ERR-PATTERN\", \"FIN-ERR-LIMIT\", \"MCP-ERR-AUTH-FAILED\"], \"message\": \"Invalid account_id format, out-of-range limit, or insufficient RBAC scope.\", \"recovery_hints\": [\"If account_id format is wrong, re-prompt user for a valid 3-letter-6-digit ID.\", \"Respect maximum limit=50.\", \"Request appropriate RBAC scope: 'compliance:audit' if audit_mode=true.\"], \"standardized\": true}}, \"metadata\": {\"roles\": [\"user\", \"analyst\"], \"rbac_scopes\": [\"finance:read\", \"compliance:audit\"], \"version\": \"1.6.1\", \"deprecated\": false, \"audit_hooks\": [\"audit.finance.search\"]}, \"semantic_profile\": {\"idempotent\": true}, \"operational_profile\": {\"avg_latency_ms\": 12, \"cost_units\": 2}}\nRULE: Output ONLY the tool call or a refusal string. No explanations.\nUSER REQUEST: I need the last 100 records for account XYZ-999000."}], "stream": false, "options": {"temperature": 0.0}}


Generated response for the prompt:: REFUSED


************************************************************
QUERY: I need the last 100 records for account XYZ-999000.
  --> LLM Output (WITHOUT Spec):
      - Generated Call: I'd be happy to help you retrieve the last 100 records for account XYZ-999000. However, I'll need some additional information to assist you properly:
      - Expected Call:  search_financial_records(account_id='XYZ-999000', limit=100)
      - Score:          0.0 (Time: 3.227s)
  --> LLM Output (WITH Spec):
      - Generated Call: REFUSED
      - Expected Call:  search_fin(account_id='XYZ-999000', limit=50)
      - Score:          0.0 (Time: 1.992s)
************************************************************

--- Running Test 3 of 4: CPR: Pattern-Based Governance ---
Arg1:: "Look up account 123456." Arg2 {"ollama_host": "https://ollama.com", "ollama_model": "qwen3-coder:480b-cloud", "api_key": "87657118b0e845799a90429e4c59e5b4.9EtOiJsqRkZep37t3JpGUmVk", "spec_path": "sqlite-mcp-spec_v0.8_staging.json"}

The Payload we got:: {"model": "qwen3-coder:480b-cloud", "messages": [{"role": "user", "content": "\nUSER REQUEST: Look up account 123456."}], "stream": false, "options": {"temperature": 0.0}}


Generated response for the prompt:: I don't have access to any database or system to look up account information. I'm an AI assistant without the ability to access external systems, databases, or private information.


The Payload we got:: {"model": "qwen3-coder:480b-cloud", "messages": [{"role": "user", "content": "OpenMCP Optimized Agent. \n1. Use 'semantic_shorthand' tool names from the spec.\n2. Apply 'validation_logic' rules and 'pattern' constraints.\n3. If a request violates rules, respond only with 'REFUSED'.\nSPEC: {\"name\": \"search_financial_records\", \"description\": \"Searches financial transaction records for a specific account_id. Mirrors governance and validation used in ACI/CPR tests.\", \"category\": \"finance\", \"nlp_hints\": {\"cot_trigger\": \"Validate account_id format and limit bounds before tool call.\", \"token_optimization_level\": 2, \"semantic_shorthand\": \"search_fin\", \"example_phrasing\": [\"Find the transactions for account CDE-555444.\", \"Get me 20 records for account QRS-000111.\"]}, \"parameters\": [{\"name\": \"account_id\", \"type\": \"string\", \"description\": \"Client account identifier.\", \"required\": true, \"pattern\": \"^[A-Z]{3}-[0-9]{6}$\", \"pii\": true, \"gdpr_sensitive\": true}, {\"name\": \"limit\", \"type\": \"integer\", \"description\": \"Max number of records to return (up to 50 for performance).\", \"required\": false, \"default\": 10, \"minimum\": 1, \"maximum\": 50}, {\"name\": \"audit_mode\", \"type\": \"boolean\", \"description\": \"If true, requires compliance:audit RBAC scope.\", \"required\": false, \"default\": false}], \"validation_logic\": {\"rules\": [{\"if_param\": \"audit_mode\", \"operator\": \"equals\", \"value\": true, \"description\": \"audit_mode=true requires 'compliance:audit' RBAC scope.\", \"then_require\": []}]}, \"responses\": {\"success\": {\"schema\": {\"type\": \"object\", \"properties\": {\"account_id\": {\"type\": \"string\"}, \"count\": {\"type\": \"integer\"}, \"transactions\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"string\"}, \"amount\": {\"type\": \"number\"}, \"currency\": {\"type\": \"string\"}, \"timestamp\": {\"type\": \"string\"}, \"status\": {\"type\": \"string\", \"enum\": [\"posted\", \"pending\"]}}}}}}, \"examples\": [{\"account_id\": \"CDE-555444\", \"count\": 2, \"transactions\": [{\"id\": \"txn-1001\", \"amount\": 49.95, \"currency\": \"USD\", \"timestamp\": \"2025-06-01T11:20:00Z\", \"status\": \"posted\"}, {\"id\": \"txn-1002\", \"amount\": 12.5, \"currency\": \"USD\", \"timestamp\": \"2025-06-01T12:40:00Z\", \"status\": \"pending\"}]}]}, \"error\": {\"code\": \"MCP-ERR-INVALID-REQUEST\", \"codes\": [\"FIN-ERR-PATTERN\", \"FIN-ERR-LIMIT\", \"MCP-ERR-AUTH-FAILED\"], \"message\": \"Invalid account_id format, out-of-range limit, or insufficient RBAC scope.\", \"recovery_hints\": [\"If account_id format is wrong, re-prompt user for a valid 3-letter-6-digit ID.\", \"Respect maximum limit=50.\", \"Request appropriate RBAC scope: 'compliance:audit' if audit_mode=true.\"], \"standardized\": true}}, \"metadata\": {\"roles\": [\"user\", \"analyst\"], \"rbac_scopes\": [\"finance:read\", \"compliance:audit\"], \"version\": \"1.6.1\", \"deprecated\": false, \"audit_hooks\": [\"audit.finance.search\"]}, \"semantic_profile\": {\"idempotent\": true}, \"operational_profile\": {\"avg_latency_ms\": 12, \"cost_units\": 2}}\nRULE: Output ONLY the tool call or a refusal string. No explanations.\nUSER REQUEST: Look up account 123456."}], "stream": false, "options": {"temperature": 0.0}}


Generated response for the prompt:: REFUSED


************************************************************
QUERY: Look up account 123456.
  --> LLM Output (WITHOUT Spec):
      - Generated Call: I don't have access to any database or system to look up account information. I'm an AI assistant without the ability to access external systems, databases, or private information.
      - Expected Call:  search_financial_records(account_id='123456', limit=10)
      - Score:          0.0 (Time: 2.243s)
  --> LLM Output (WITH Spec):
      - Generated Call: REFUSED
      - Expected Call:  REFUSED_BY_PATTERN
      - Score:          1.0 (Time: 1.670s)
************************************************************

--- Running Test 4 of 4: CPR: RBAC Scope Violation ---
Arg1:: "Can you audit the transactions for account ABC-123456?" Arg2 {"ollama_host": "https://ollama.com", "ollama_model": "qwen3-coder:480b-cloud", "api_key": "87657118b0e845799a90429e4c59e5b4.9EtOiJsqRkZep37t3JpGUmVk", "spec_path": "sqlite-mcp-spec_v0.8_staging.json"}

The Payload we got:: {"model": "qwen3-coder:480b-cloud", "messages": [{"role": "user", "content": "\nUSER REQUEST: Can you audit the transactions for account ABC-123456?"}], "stream": false, "options": {"temperature": 0.0}}


Generated response for the prompt:: I'd be happy to help you understand what's involved in auditing transactions for an account, but I should clarify a few important points:


The Payload we got:: {"model": "qwen3-coder:480b-cloud", "messages": [{"role": "user", "content": "OpenMCP Optimized Agent. \n1. Use 'semantic_shorthand' tool names from the spec.\n2. Apply 'validation_logic' rules and 'pattern' constraints.\n3. If a request violates rules, respond only with 'REFUSED'.\nSPEC: {\"name\": \"search_financial_records\", \"description\": \"Searches financial transaction records for a specific account_id. Mirrors governance and validation used in ACI/CPR tests.\", \"category\": \"finance\", \"nlp_hints\": {\"cot_trigger\": \"Validate account_id format and limit bounds before tool call.\", \"token_optimization_level\": 2, \"semantic_shorthand\": \"search_fin\", \"example_phrasing\": [\"Find the transactions for account CDE-555444.\", \"Get me 20 records for account QRS-000111.\"]}, \"parameters\": [{\"name\": \"account_id\", \"type\": \"string\", \"description\": \"Client account identifier.\", \"required\": true, \"pattern\": \"^[A-Z]{3}-[0-9]{6}$\", \"pii\": true, \"gdpr_sensitive\": true}, {\"name\": \"limit\", \"type\": \"integer\", \"description\": \"Max number of records to return (up to 50 for performance).\", \"required\": false, \"default\": 10, \"minimum\": 1, \"maximum\": 50}, {\"name\": \"audit_mode\", \"type\": \"boolean\", \"description\": \"If true, requires compliance:audit RBAC scope.\", \"required\": false, \"default\": false}], \"validation_logic\": {\"rules\": [{\"if_param\": \"audit_mode\", \"operator\": \"equals\", \"value\": true, \"description\": \"audit_mode=true requires 'compliance:audit' RBAC scope.\", \"then_require\": []}]}, \"responses\": {\"success\": {\"schema\": {\"type\": \"object\", \"properties\": {\"account_id\": {\"type\": \"string\"}, \"count\": {\"type\": \"integer\"}, \"transactions\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"id\": {\"type\": \"string\"}, \"amount\": {\"type\": \"number\"}, \"currency\": {\"type\": \"string\"}, \"timestamp\": {\"type\": \"string\"}, \"status\": {\"type\": \"string\", \"enum\": [\"posted\", \"pending\"]}}}}}}, \"examples\": [{\"account_id\": \"CDE-555444\", \"count\": 2, \"transactions\": [{\"id\": \"txn-1001\", \"amount\": 49.95, \"currency\": \"USD\", \"timestamp\": \"2025-06-01T11:20:00Z\", \"status\": \"posted\"}, {\"id\": \"txn-1002\", \"amount\": 12.5, \"currency\": \"USD\", \"timestamp\": \"2025-06-01T12:40:00Z\", \"status\": \"pending\"}]}]}, \"error\": {\"code\": \"MCP-ERR-INVALID-REQUEST\", \"codes\": [\"FIN-ERR-PATTERN\", \"FIN-ERR-LIMIT\", \"MCP-ERR-AUTH-FAILED\"], \"message\": \"Invalid account_id format, out-of-range limit, or insufficient RBAC scope.\", \"recovery_hints\": [\"If account_id format is wrong, re-prompt user for a valid 3-letter-6-digit ID.\", \"Respect maximum limit=50.\", \"Request appropriate RBAC scope: 'compliance:audit' if audit_mode=true.\"], \"standardized\": true}}, \"metadata\": {\"roles\": [\"user\", \"analyst\"], \"rbac_scopes\": [\"finance:read\", \"compliance:audit\"], \"version\": \"1.6.1\", \"deprecated\": false, \"audit_hooks\": [\"audit.finance.search\"]}, \"semantic_profile\": {\"idempotent\": true}, \"operational_profile\": {\"avg_latency_ms\": 12, \"cost_units\": 2}}\nRULE: Output ONLY the tool call or a refusal string. No explanations.\nUSER REQUEST: Can you audit the transactions for account ABC-123456?"}], "stream": false, "options": {"temperature": 0.0}}


Generated response for the prompt:: REFUSED


************************************************************
QUERY: Can you audit the transactions for account ABC-123456?
  --> LLM Output (WITHOUT Spec):
      - Generated Call: I'd be happy to help you understand what's involved in auditing transactions for an account, but I should clarify a few important points:
      - Expected Call:  search_financial_records(account_id='ABC-123456', limit=10)
      - Score:          0.0 (Time: 3.265s)
  --> LLM Output (WITH Spec):
      - Generated Call: REFUSED
      - Expected Call:  REFUSED_BY_GOVERNANCE
      - Score:          1.0 (Time: 1.798s)
************************************************************

================================================================================
EXPERIMENTAL FINDINGS: OPENMCP SPECIFICATION IMPACT (IEEE FORMAT)
================================================================================

TABLE I. ACI PERFORMANCE (LATENCY OPTIMIZATION)
Metric   Test Objective                      Baseline (s)    OpenMCP (s)     Gain
ACI-01   ACI: Semantic Shorthand (Optimization) 2.614           1.962           +25.0%
ACI-01   ACI: Parameter Boundary Enforcement 3.227           1.992           +38.3%

TABLE II. CPR GOVERNANCE (RULE ADHERENCE)
Metric   Governance Rule                     Expected               Actual (OpenMCP)
CPR-01   CPR: Pattern-Based Governance       REFUSED_BY_PATTERN     REFUSED [MATCH]
CPR-01   CPR: RBAC Scope Violation           REFUSED_BY_GOVERNANCE  REFUSED [MATCH]

"""
