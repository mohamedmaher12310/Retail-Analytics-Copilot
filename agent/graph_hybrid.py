import dspy
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from agent.dspy_signatures import Router, GenerateSQL, SynthesizeAnswer
from agent.rag.retrieval import LocalRetriever
from agent.tools.sqlite_tool import get_schema, execute_sql

# 1. Setup DSPy with Ollama
lm = dspy.OllamaLocal(model="qwen2:1.5b", max_tokens=1000)
dspy.settings.configure(lm=lm)

# 2. Define State
class AgentState(TypedDict):
    question: str
    format_hint: str
    classification: str
    schema: str
    doc_context: List[dict]
    constraints: str
    sql_query: str
    sql_result: Any
    sql_error: str
    final_answer: Any
    explanation: str
    citations: List[str]
    repair_count: int

# 3. Initialize Helpers
retriever = LocalRetriever()
db_schema = get_schema()

# 4. Modules (Nodes)
router_module = dspy.Predict(Router)
sql_gen_module = dspy.ChainOfThought(GenerateSQL) # CoT helps logic
synthesizer_module = dspy.Predict(SynthesizeAnswer)

# --- Nodes ---

def router_node(state: AgentState):
    pred = router_module(question=state["question"])
    return {"classification": pred.classification.lower().strip(), "schema": db_schema}

def retrieval_node(state: AgentState):
    results = retriever.search(state["question"])
    # Extract constraints/context string
    context_str = "\n".join([f"[{r['id']}] {r['text']}" for r in results])
    return {"doc_context": results, "constraints": context_str}

def sql_gen_node(state: AgentState):
    # Pass doc constraints to SQL gen if they exist
    constraints = state.get("constraints", "")
    pred = sql_gen_module(
        question=state["question"], 
        schema=state["schema"],
        constraints=constraints
    )
    # Cleanup SQL string (remove markdown ```sql tags if qwen adds them)
    clean_sql = pred.sql_query.replace("```sql", "").replace("```", "").strip()
    return {"sql_query": clean_sql}

def sql_exec_node(state: AgentState):
    result = execute_sql(state["sql_query"])
    if isinstance(result, str) and result.startswith("SQL Error"):
        return {"sql_result": None, "sql_error": result}
    return {"sql_result": result, "sql_error": None}

def synthesizer_node(state: AgentState):
    context_str = state.get("constraints", "")
    sql_res = str(state.get("sql_result", "No SQL executed"))
    
    pred = synthesizer_module(
        question=state["question"],
        format_hint=state["format_hint"],
        sql_query=state.get("sql_query", ""),
        sql_result=sql_res,
        doc_context=context_str
    )
    
    # Simple post-processing for citations list if model returns string representation
    citations = pred.citations
    if isinstance(citations, str):
        # Fallback parsing if Qwen returns a string like "['Orders']"
        try:
            import ast
            citations = ast.literal_eval(citations)
        except:
            citations = [citations]

    return {
        "final_answer": pred.final_answer, 
        "explanation": pred.explanation, 
        "citations": citations
    }

def repair_node(state: AgentState):
    # A simple repair strategy: append error to question context and retry
    current_count = state.get("repair_count", 0)
    return {"repair_count": current_count + 1}

# --- Graph Definition ---

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("sql_gen", sql_gen_node)
workflow.add_node("sql_exec", sql_exec_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("repair", repair_node)

workflow.set_entry_point("router")

# Conditional Logic
def route_decision(state):
    cls = state["classification"]
    if "sql" in cls or "hybrid" in cls:
        if "hybrid" in cls:
            return "hybrid" # Go to retriever first, then SQL
        return "sql" # Go straight to SQL
    return "rag" # RAG only

def sql_check(state):
    if state["sql_error"] and state["repair_count"] < 2:
        return "retry"
    return "finalize"

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "sql": "sql_gen",
        "rag": "retriever",
        "hybrid": "retriever"
    }
)

# If hybrid, retriever goes to SQL gen next
workflow.add_edge("retriever", "sql_gen") 

def post_retrieval_route(state):
    if "hybrid" in state["classification"]:
        return "sql_gen"
    return "synthesizer"

# Redefine retriever edges
workflow.add_conditional_edges("retriever", post_retrieval_route, {"sql_gen": "sql_gen", "synthesizer": "synthesizer"})

workflow.add_edge("sql_gen", "sql_exec")

workflow.add_conditional_edges(
    "sql_exec",
    sql_check,
    {
        "retry": "repair",
        "finalize": "synthesizer"
    }
)

workflow.add_edge("repair", "sql_gen") # Retry SQL generation
workflow.add_edge("synthesizer", END)

app = workflow.compile()