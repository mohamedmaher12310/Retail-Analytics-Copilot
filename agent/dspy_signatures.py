import dspy

class Router(dspy.Signature):
    """Classify the user question into one of three categories: 'sql' (requires database), 'rag' (requires documents), or 'hybrid' (requires both)."""
    question = dspy.InputField()
    classification = dspy.OutputField(desc="One of: sql, rag, hybrid")

class GenerateSQL(dspy.Signature):
    """Generate a SQLite query based on the question and schema. 
    Use 'Order Details' (with space) for line items.
    Revenue logic: SUM(UnitPrice * Quantity * (1 - Discount)).
    """
    question = dspy.InputField()
    schema = dspy.InputField()
    constraints = dspy.InputField(desc="Specific dates, categories, or logic from docs")
    sql_query = dspy.OutputField(desc="Valid SQLite query string. No markdown formatting.")

class SynthesizeAnswer(dspy.Signature):
    """Answer the question based on the provided context and SQL results. 
    Ensure the output matches the format_hint exactly.
    Include citations for DB tables used and Doc chunks used.
    """
    question = dspy.InputField()
    format_hint = dspy.InputField()
    sql_query = dspy.InputField()
    sql_result = dspy.InputField()
    doc_context = dspy.InputField()
    
    final_answer = dspy.OutputField(desc="The answer matching format_hint")
    explanation = dspy.OutputField(desc="Brief explanation of how the answer was derived")
    citations = dspy.OutputField(desc="List of strings: DB tables (e.g., 'Orders') and doc IDs (e.g., 'marketing::chunk1')")