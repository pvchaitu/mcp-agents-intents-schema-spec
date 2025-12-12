import sqlite3
import json
import os
import yaml
from typing import TypedDict, Annotated, List
from operator import itemgetter

# Using pydantic's utility to create models dynamically
from langchain_core.pydantic_v1 import BaseModel, Field, create_model 
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import OllamaFunctions
from langgraph.graph import StateGraph, END


# --- UTILITY FUNCTIONS FOR CONFIGURATION ---

def load_config(config_path: str = "config.yaml"):
    """Loads configuration from the YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")
    return config

def create_pydantic_model_from_schema(intent_name: str, schema: dict) -> type[BaseModel]:
    """Dynamically creates a Pydantic model from a JSON schema intent definition."""
    
    # Extract properties and required fields
    properties = schema['parameters']['properties']
    required = schema['parameters'].get('required', [])
    
    fields = {}
    
    for prop_name, prop_schema in properties.items():
        field_type = str
        # Simple type mapping
        if prop_schema.get('type') == 'integer':
            field_type = int
        elif prop_schema.get('type') == 'boolean':
            field_type = bool
        
        # Prepare the field definition (type, Field object)
        field_definition = (field_type, Field(
            prop_schema.get('default'), 
            description=prop_schema.get('description')
        ))
        
        # If required, remove default from Field and make the type non-optional
        if prop_name in required:
            field_definition = (field_type, Field(
                ..., 
                description=prop_schema.get('description')
            ))

        fields[prop_name] = field_definition

    # Create the model dynamically
    ToolModel = create_model(
        f'{intent_name.replace("_", "")}Tool',
        __doc__=schema['description'],
        __base__=BaseModel,
        **fields
    )
    return ToolModel

def load_mcp_tools(spec_path: str) -> List[type[BaseModel]]:
    """Loads the MCP specification and generates a list of Pydantic tool models."""
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"MCP spec file not found at: {spec_path}")
        
    with open(spec_path, 'r') as f:
        spec = json.load(f)
        
    tools = []
    # Loop through all defined intents and create a Pydantic model for each
    for intent_name, intent_data in spec['intents'].items():
        try:
            ToolModel = create_pydantic_model_from_schema(intent_name, intent_data)
            tools.append(ToolModel)
        except Exception as e:
            print(f"Warning: Could not create Pydantic model for intent '{intent_name}'. Error: {e}")
            
    return tools


# --- 1. MOCK SQLITE MCP SERVER (TOOL IMPLEMENTATION) ---

class MockSQLiteServer:
    """Simulates the mcp-server-sqlite functionality using sqlite3."""
    
    def __init__(self, db_path: str, tool_models: List[type[BaseModel]]):
        self.db_path = db_path
        self.connections = {}
        self._setup_db()
        # Tools are the dynamically loaded Pydantic models
        self.tools = tool_models

    def _setup_db(self):
        """Creates the initial database file and populates it with data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS employees")
        cursor.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                salary INTEGER,
                department TEXT
            )
        """)
        
        employees_data = [
            ("Alice Johnson", 75000, "Sales"),
            ("Bob Smith", 92000, "Engineering"),
            ("Charlie Brown", 55000, "HR"),
            ("Diana Prince", 120000, "Engineering"),
            ("Ethan Hunt", 60000, "Sales"),
            ("Fiona Glenanne", 110000, "Sales"),
            ("George Clooney", 88000, "Marketing")
        ]
        cursor.executemany("INSERT INTO employees (name, salary, department) VALUES (?, ?, ?)", employees_data)
        
        cursor.execute("DROP TABLE IF EXISTS projects")
        cursor.execute("""
            CREATE TABLE projects (
                project_id INTEGER PRIMARY KEY,
                project_name TEXT NOT NULL,
                budget REAL
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"Database '{self.db_path}' created and populated.")

    def _get_conn(self, alias):
        """Internal utility to get an active connection."""
        conn_info = self.connections.get(alias)
        if not conn_info:
            raise ValueError(f"Database alias '{alias}' not connected.")
        return conn_info['conn']
    
    # --- MCP Intent Implementations (Methods must match intent names) ---

    def connect_database(self, path: str, alias: str, readonly: bool = False):
        """[Intent] Connects to the database."""
        try:
            conn = sqlite3.connect(path)
            self.connections[alias] = {'conn': conn, 'readonly': readonly}
            return f"Successfully connected to database at '{path}' with alias '{alias}'. Read-only: {readonly}."
        except Exception as e:
            return f"Error connecting to database: {e}"

    def get_catalog(self, database_alias: str):
        """[Intent] Returns a comprehensive schema catalog."""
        # ... (implementation remains the same as previous examples)
        conn = self._get_conn(database_alias)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']
        catalog = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [{"name": col[1], "type": col[2], "notnull": col[3], "pk": col[5]} for col in cursor.fetchall()]
            catalog[table] = {"columns": columns}
        return json.dumps(catalog, indent=2)

    def get_table_info(self, database_alias: str, table_name: str, sample_rows: int = 5):
        """[Intent] Retrieves detailed information about a specific table."""
        # ... (implementation remains the same as previous examples)
        conn = self._get_conn(database_alias)
        cursor = conn.cursor()
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [description[1] for description in cursor.fetchall()]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {sample_rows}")
            rows = cursor.fetchall()
            result = {"table_name": table_name, "row_count": row_count, "columns": columns, "sample_data": rows}
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error retrieving table info: {e}"

    def execute_read_query(self, database_alias: str, query: str, limit: int = 100):
        """[Intent] Executes a SELECT query."""
        # ... (implementation remains the same as previous examples)
        conn = self._get_conn(database_alias)
        cursor = conn.cursor()
        try:
            if "LIMIT" not in query.upper() and query.upper().startswith("SELECT"):
                 query = f"{query} LIMIT {limit}"
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            header = " | ".join(columns)
            separator = " | ".join(["---"] * len(columns))
            body = "\n".join([" | ".join(map(str, row)) for row in rows])
            return f"Query successful. Result:\n{header}\n{separator}\n{body}"
        except sqlite3.Error as e:
            return f"SQLite Error while executing query: {e}. Query: {query}"
        except Exception as e:
            return f"Unexpected Error: {e}"

    def execute_write_query(self, database_alias: str, query: str):
        """[Intent] Executes a modifying query (DDL/DML)."""
        # ... (implementation remains the same as previous examples)
        conn_info = self.connections.get(database_alias)
        if conn_info['readonly']:
            return "Error: Database is connected in read-only mode. Cannot execute write queries."
        conn = conn_info['conn']
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            conn.commit()
            return f"Write query successful. Rows affected: {cursor.rowcount}"
        except sqlite3.Error as e:
            return f"SQLite Error while executing write query: {e}. Query: {query}"


# --- 2. LANGGRAPH SETUP (LOGIC REMAINS THE SAME) ---

class AgentState(TypedDict):
    query: str
    tool_call: str
    tool_result: str
    final_answer: str
    messages: Annotated[List[BaseMessage], itemgetter("messages")]

# Global placeholder for the LLM instance
llm = None 

def llm_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    
    if response.tool_calls:
        return {"tool_call": response.tool_calls[0], "messages": messages + [response]}
    else:
        return {"final_answer": response.content, "messages": messages + [response]}

def tool_node(state: AgentState):
    tool_call = state["tool_call"]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args")

    tool_func = getattr(tool_executor, tool_name)
    
    print(f"\n--- TOOL EXECUTION: {tool_name}({tool_args}) ---")
    
    tool_result = tool_func(**tool_args)
    
    return {
        "tool_result": tool_result, 
        "messages": state["messages"] + [HumanMessage(content=tool_result)]
    }

def decide_next_step(state: AgentState):
    if state.get("final_answer"):
        return END
    elif state.get("tool_call"):
        return "call_tool"
    else:
        return "llm" 

def build_graph(llm_instance):
    global llm
    llm = llm_instance
    
    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.add_node("call_tool", tool_node)
    workflow.set_entry_point("llm")
    
    workflow.add_conditional_edges(
        "llm",
        decide_next_step,
        {"call_tool": "call_tool", END: END}
    )
    workflow.add_edge("call_tool", "llm")
    
    return workflow.compile()

# --- 3. ACI COMPARISON LOGIC (Uses config values) ---

def run_aci_test(agent_name: str, llm_instance: OllamaFunctions, test_cases: List[dict]):
    """Runs a sequence of tests and scores argument correctness."""
    correct_arguments = 0
    graph = build_graph(llm_instance)
    
    print(f"\n=============================================")
    print(f"       Running ACI Test for: {agent_name}      ")
    print(f"=============================================")
    
    if not server.connections.get(config['mcp_config']['db_alias']):
        server.connect_database(config['mcp_config']['db_path'], config['mcp_config']['db_alias'], readonly=False)
        print(f"Pre-connected '{config['mcp_config']['db_alias']}' for subsequent tests.")

    for i, case in enumerate(test_cases):
        if i == 0 and config['mcp_config']['db_alias'] in server.connections:
            continue
            
        print(f"\n--- TEST CASE {case['step']} ---")
        print(f"User Query: {case['user_query']}")
        
        initial_state = {"messages": [HumanMessage(content=case["user_query"])]}
        
        try:
            output = graph.invoke(initial_state, {"recursion_limit": 2})
            tool_call = output.get("tool_call")
            
            if tool_call and tool_call.get("name") == case["expected_tool"]:
                generated_args = tool_call.get("args", {})
                is_correct = True
                for arg_name, expected_value in case["expected_args"].items():
                    if arg_name == "query":
                        if not isinstance(generated_args.get(arg_name), str) or not generated_args.get(arg_name).strip():
                            is_correct = False
                            break
                    elif arg_name != "query" and generated_args.get(arg_name) != expected_value:
                        is_correct = False
                        break
                
                if is_correct:
                    print(f"Result: PASS. Correct tool call: {tool_call}")
                    correct_arguments += 1
                else:
                    print(f"Result: FAIL. Expected args: {case['expected_args']}, Got: {generated_args}")
            else:
                print(f"Result: FAIL. Did not call expected tool '{case['expected_tool']}'.  {output.get('final_answer')}")

        except Exception as e:
            print(f"Error during graph execution for case {case['step']}: {e}")

    return correct_arguments

# --- 4. CONFIGURATION LOADING AND EXECUTION ---

try:
    # --- Load Configuration ---
    config = load_config()
    llm_conf = config['llm_config']
    mcp_conf = config['mcp_config']

    # Set API Key in environment for OllamaFunctions
    os.environ["OPENAI_API_KEY"] = llm_conf['api_key'] 

    # --- Load Dynamic Tools ---
    mcp_tool_models = load_mcp_tools(mcp_conf['spec_path'])
    
    # --- Initialize Mock Server ---
    server = MockSQLiteServer(mcp_conf['db_path'], mcp_tool_models)
    tool_executor = server

    # --- Define Test Cases (Uses config values for dynamic injection) ---
    TEST_CASES = [
        {
            "user_query": f"Connect to the {mcp_conf['db_path']} file and set its alias to '{mcp_conf['db_alias']}'.",
            "expected_tool": "connect_database",
            "expected_args": {"path": mcp_conf['db_path'], "alias": mcp_conf['db_alias'], "readonly": False},
            "step": 1
        },
        {
            "user_query": f"What are the tables in the '{mcp_conf['db_alias']}' database?",
            "expected_tool": "get_catalog",
            "expected_args": {"database_alias": mcp_conf['db_alias']},
            "step": 2
        },
        {
            "user_query": "Get the top 5 employees with the highest salary from the 'employees' table.",
            "expected_tool": "execute_read_query",
            "expected_args": {
                "database_alias": mcp_conf['db_alias'],
                "query": "SELECT name, salary, department FROM employees ORDER BY salary DESC",
                "limit": 5
            },
            "step": 3
        },
        {
            "user_query": "What are the columns in the projects table in 'company_db'?",
            "expected_tool": "get_table_info",
            "expected_args": {"database_alias": mcp_conf['db_alias'], "table_name": "projects"},
            "step": 4
        }
    ]

    # 4.1. Agent with FULL MCP CONTEXT (High Context)
    llm_mcp_context = OllamaFunctions(
        model=llm_conf['model'], 
        base_url=llm_conf['base_url'], 
        format=llm_conf['format'], 
        keep_alive=llm_conf['keep_alive']
    ).bind_tools(mcp_tool_models)

    # 4.2. Agent with MINIMAL CONTEXT (Low Context)
    class MinimalExecuteTool(BaseModel):
        """Executes a SQL query."""
        db_alias: str = Field(description="The database alias.")
        sql: str = Field(description="The SQL query to run.")

    llm_minimal_context = OllamaFunctions(
        model=llm_conf['model'], 
        base_url=llm_conf['base_url'], 
        format=llm_conf['format'], 
        keep_alive=llm_conf['keep_alive']
    ).bind_tools([MinimalExecuteTool])


    # --- ACI TEST EXECUTION ---

    total_tests = len(TEST_CASES)
    score_minimal = run_aci_test("LOW CONTEXT (NO MCP SPEC)", llm_minimal_context, TEST_CASES)
    score_mcp = run_aci_test("HIGH CONTEXT (FULL MCP SPEC)", llm_mcp_context, TEST_CASES)

    # Calculate and report ACI
    aci_score = score_mcp - score_minimal
    aci_percentage = (aci_score / total_tests) * 100 if total_tests > 0 else 0


    print("\n\n#####################################################")
    print("          ARGUMENT CORRECTNESS IMPROVEMENT (ACI) METRIC")
    print("#####################################################")
    print(f"Total Test Cases: {total_tests}")
    print(f"Correct Arguments (Minimal Context): {score_minimal} / {total_tests}")
    print(f"Correct Arguments (Full MCP Context): {score_mcp} / {total_tests}")
    print(f"**ACI Score (Improvement in Correct Arguments): {aci_score}**")
    print(f"**ACI Percentage: {aci_percentage:.2f}%** (Higher is better)")
    print("#####################################################")


    # --- 5. FINAL RUN WITH HIGH CONTEXT (MCP) AGENT ---

    print("\n\n--- RUNNING FINAL NLP QUERY WITH MCP AGENT ---")
    final_graph = build_graph(llm_mcp_context)
    final_query = f"Please connect to the {mcp_conf['db_path']} with alias '{mcp_conf['db_alias']}', then get the top 5 employees with the highest salary from the 'employees' table."

    final_state = final_graph.invoke({"messages": [HumanMessage(content=final_query)]})

    print("\n\n#####################################################")
    print("          FINAL RESULT FROM MCP LANGGRAPH")
    print("#####################################################")
    print(f"Query: {final_query}")
    print(f"Final Answer:")
    print(final_state['final_answer'])
    print("#####################################################")

except FileNotFoundError as e:
    print(f"\n\nFATAL ERROR: Configuration file missing. {e}")
    print("Please ensure both 'config.yaml' and 'open_mcp_spec.json' are created.")
except Exception as e:
    print("\n\n#####################################################")
    print("        ERROR: Configuration or Remote API Failed")
    print("#####################################################")
    print(f"An error occurred, likely related to configuration or the remote API call.")
    print(f"Error details: {type(e).__name__}: {e}")
    print(f"Ensure the remote Ollama service is running and accessible at '{llm_conf['base_url']}', and that the '{llm_conf['model']}' model is available.")
    print("#####################################################")