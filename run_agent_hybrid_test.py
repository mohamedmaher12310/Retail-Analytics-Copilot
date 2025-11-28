import click
import json
import traceback

# Mock the graph for testing (no Ollama dependency)
class MockApp:
    def invoke(self, state):
        """Mock invoke that returns sensible defaults."""
        question = state.get("question", "")
        format_hint = state.get("format_hint", "")
        
        # Determine classification
        if "sql" in question.lower() and "revenue" in question.lower():
            classification = "sql"
        elif "policy" in question.lower() or "rag" in question.lower():
            classification = "rag"
        else:
            classification = "hybrid"
        
        # Mock SQL for certain queries
        sql_query = ""
        sql_result = None
        if "top 3 products" in question.lower():
            sql_query = "SELECT ProductName, SUM(UnitPrice * Quantity * (1-Discount)) as revenue FROM Products JOIN OrderDetails USING(ProductID) GROUP BY ProductID ORDER BY revenue DESC LIMIT 3"
            sql_result = [
                {"product": "Queso Cabrales", "revenue": 5000.0},
                {"product": "Camembert Pierrot", "revenue": 4800.0},
                {"product": "Tofu", "revenue": 4500.0}
            ]
        
        # Mock answer based on format_hint
        final_answer = None
        if format_hint == "int":
            final_answer = 30
        elif format_hint == "float":
            final_answer = 12345.67
        elif format_hint == "{category:str, quantity:int}":
            final_answer = {"category": "Beverages", "quantity": 15000}
        elif "list" in format_hint.lower():
            final_answer = [
                {"product": "Queso Cabrales", "revenue": 5000.0},
                {"product": "Camembert Pierrot", "revenue": 4800.0},
                {"product": "Tofu", "revenue": 4500.0}
            ]
        else:
            final_answer = "Mock response for: " + question[:50]
        
        return {
            "question": state.get("question"),
            "format_hint": state.get("format_hint"),
            "classification": classification,
            "schema": "Mock schema",
            "doc_context": [],
            "constraints": "Mock constraints",
            "sql_query": sql_query,
            "sql_result": sql_result,
            "sql_error": None,
            "final_answer": final_answer,
            "explanation": "Mock explanation",
            "citations": ["mock_doc_1", "Orders", "Order Details"]
        }

app = MockApp()

@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch, out):
    """
    Main execution loop for hybrid agent evaluation (TEST VERSION).
    Uses mock responses instead of calling Ollama.
    """
    
    print("--- Starting Hybrid Agent Evaluation (TEST MODE) ---")
    results = []
    
    # Load questions from JSONL
    try:
        with open(batch, 'r') as f:
            questions = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(questions)} questions from {batch}")
    except Exception as e:
        print(f"ERROR: Failed to load questions: {e}")
        return
    
    # Process each question
    for idx, q in enumerate(questions, 1):
        question_id = q.get("id", f"q_{idx}")
        print(f"\n[{idx}/{len(questions)}] Processing: {question_id}")
        print(f"  Question: {q.get('question', '')[:80]}...")
        
        try:
            initial_state = {
                "question": q.get("question", ""),
                "format_hint": q.get("format_hint", ""),
                "repair_count": 0,
                "sql_error": None,
                "classification": "",
                "schema": "",
                "doc_context": [],
                "constraints": "",
                "sql_query": "",
                "sql_result": None,
                "final_answer": None,
                "explanation": "",
                "citations": []
            }
            
            # Run the mock workflow
            print(f"  Running workflow...")
            final_state = app.invoke(initial_state)
            
            # Extract and format output
            output = {
                "id": question_id,
                "question": q.get("question", ""),
                "format_hint": q.get("format_hint", ""),
                "final_answer": final_state.get("final_answer"),
                "sql": final_state.get("sql_query", ""),
                "sql_error": final_state.get("sql_error"),
                "confidence": 1.0 if not final_state.get("sql_error") else 0.5,
                "explanation": final_state.get("explanation", ""),
                "citations": final_state.get("citations", [])
            }
            
            print(f"  ✓ Success. Answer: {str(final_state.get('final_answer', 'N/A'))[:60]}")
            results.append(output)
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            
            # Still add result with error info
            output = {
                "id": question_id,
                "question": q.get("question", ""),
                "format_hint": q.get("format_hint", ""),
                "final_answer": None,
                "sql": "",
                "sql_error": str(e),
                "confidence": 0.0,
                "explanation": f"Error during processing: {str(e)}",
                "citations": []
            }
            results.append(output)
    
    # Write results to output file
    print(f"\n--- Writing Results ---")
    try:
        with open(out, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"✓ Wrote {len(results)} results to {out}")
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}")
    
    print(f"\n--- Summary ---")
    successful = sum(1 for r in results if r.get("sql_error") is None)
    print(f"Successful: {successful}/{len(results)}")
    print(f"Output file: {out}")

if __name__ == "__main__":
    main()
