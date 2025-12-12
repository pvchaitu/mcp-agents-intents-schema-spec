import sqlite3
import os

# --- Configuration (Should match config.yaml) ---
DB_PATH = "company_data.db"

def create_company_database(db_path: str):
    """
    Creates and populates the SQLite database file with 'employees' and 'projects' tables.
    """
    
    # 1. Connect to the database file (creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Connecting to/Creating database: {db_path}")

    # --- Employees Table (Primary table for the NLP query) ---
    print("Creating 'employees' table...")
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
        ("George Clooney", 88000, "Marketing"),
        ("Hannah Baker", 130000, "Engineering"),
        ("Ian Malcolm", 95000, "Sales"),
        ("Jasmine Doe", 70000, "HR")
    ]
    cursor.executemany("INSERT INTO employees (name, salary, department) VALUES (?, ?, ?)", employees_data)
    
    # --- Projects Table (Used for schema introspection intent) ---
    print("Creating 'projects' table...")
    cursor.execute("DROP TABLE IF EXISTS projects")
    cursor.execute("""
        CREATE TABLE projects (
            project_id INTEGER PRIMARY KEY,
            project_name TEXT NOT NULL,
            budget REAL,
            employee_id INTEGER,
            FOREIGN KEY(employee_id) REFERENCES employees(id)
        )
    """)
    
    projects_data = [
        ("Apollo Launch", 500000.00, 4), # Diana Prince
        ("Q4 Sales Strategy", 75000.00, 1), # Alice Johnson
        ("Marketing Rebrand", 120000.00, 7) # George Clooney
    ]
    cursor.executemany("INSERT INTO projects (project_name, budget, employee_id) VALUES (?, ?, ?)", projects_data)

    # 3. Commit changes and close the connection
    conn.commit()
    conn.close()
    
    print(f"\nDatabase '{db_path}' successfully created with 2 tables and populated.")
    print("Schema Check:")
    print("  Table 'employees': id, name, salary, department (10 rows)")
    print("  Table 'projects': project_id, project_name, budget, employee_id (3 rows)")

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Existing file '{DB_PATH}' removed.")
        
    create_company_database(DB_PATH)