import sqlite3
import pandas as pd

DB_PATH = "data/northwind.sqlite"

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_schema():
    """Returns a simplified schema string for the LLM."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_str = ""
    for table in tables:
        table_name = table[0]
        # Skip internal sqlite tables
        if "sqlite" in table_name:
            continue
            
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        schema_str += f"Table: {table_name}\nColumns: {', '.join(col_names)}\n\n"
    
    conn.close()
    return schema_str

def execute_sql(query):
    """Executes SQL and returns results as a list of dicts or error string."""
    try:
        conn = get_db_connection()
        # Enable case-insensitive logic if needed, but standard SQL usually fine
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            return "Query executed successfully but returned 0 rows."
        return df.to_dict(orient="records")
    except Exception as e:
        return f"SQL Error: {str(e)}"