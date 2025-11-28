import click
import json
import traceback
import ast
import re
import os
import sys

# Ensure project root is on sys.path so package imports like `rag` and `agent` work
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.graph_hybrid import app

def parse_format_hint(format_hint):
    """Parse format_hint to determine expected type."""
    format_hint = format_hint.strip()
    if format_hint == "int":
        return "int"
    elif format_hint == "float":
        return "float"
    elif format_hint.startswith("{") and format_hint.endswith("}"):
        return "dict"
    elif format_hint.startswith("list"):
        return "list"
    else:
        return "str"

def normalize_answer(final_answer, format_hint):
    """
    Normalize final_answer to match format_hint strictly.
    Returns (normalized_answer, is_valid).
    """
    expected_type = parse_format_hint(format_hint)
    
    if final_answer is None:
        return None, False
    
    try:
        # If it's a string representation, try to parse it
        if isinstance(final_answer, str):
            # Try to eval as Python literal
            try:
                final_answer = ast.literal_eval(final_answer)
            except:
                pass
        
        if expected_type == "int":
            if isinstance(final_answer, (int, float)):
                return int(final_answer), True
            elif isinstance(final_answer, str):
                try:
                    return int(float(final_answer)), True
                except:
                    return None, False
        
        elif expected_type == "float":
            if isinstance(final_answer, (int, float)):
                return round(float(final_answer), 2), True
            elif isinstance(final_answer, str):
                try:
                    return round(float(final_answer), 2), True
                except:
                    return None, False
        
        elif expected_type == "dict":
            if isinstance(final_answer, dict):
                return final_answer, True
            elif isinstance(final_answer, str):
                try:
                    parsed = ast.literal_eval(final_answer)
                    if isinstance(parsed, dict):
                        return parsed, True
                except:
                    pass
            return None, False
        
        elif expected_type == "list":
            if isinstance(final_answer, list):
                return final_answer, True
            elif isinstance(final_answer, str):
                try:
                    parsed = ast.literal_eval(final_answer)
                    if isinstance(parsed, list):
                        return parsed, True
                except:
                    pass
            return None, False
        
        else:  # str
            return str(final_answer), True
    
    except Exception:
        return None, False

def sanitize_citations(citations, sql_query):
    """
    Extract table names from SQL and merge with provided citations.
    Returns a list of citation strings in format: "TableName" or "doc_id::chunkN".
    """
    result = []
    
    # Extract table names from SQL if present
    if sql_query:
        # Simple regex to find table names (not perfect, but works for common cases)
        table_pattern = r'\b(?:FROM|JOIN|INTO|UPDATE|DELETE\s+FROM)\s+[`"]?(\w+)[`"]?'
        matches = re.findall(table_pattern, sql_query, re.IGNORECASE)
        result.extend(set(matches))
    
    # Add doc citations (already formatted)
    if isinstance(citations, list):
        for c in citations:
            if isinstance(c, str) and c not in result:
                result.append(c)
    elif isinstance(citations, str):
        if citations not in result:
            result.append(citations)
    
    return sorted(list(set(result)))

def truncate_explanation(explanation, max_chars=250):
    """Truncate explanation to fit within 2 sentences (~250 chars)."""
    if not explanation:
        return ""
    
    explanation = explanation.strip()
    
    # Try to cut at sentence boundary within max_chars
    sentences = re.split(r'(?<=[.!?])\s+', explanation)
    result = ""
    for sent in sentences:
        if len(result) + len(sent) + 1 <= max_chars:
            result += sent + " " if result else sent + " "
        else:
            break
    
    return result.strip()

@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch, out):
    """
    Main execution loop for hybrid agent evaluation.
    Enforces Output Contract: each result must have id, final_answer, sql, confidence, explanation, citations.
    """
    
    print("--- Starting Hybrid Agent Evaluation (with Output Contract Validation) ---")
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
        format_hint = q.get("format_hint", "")
        
        print(f"\n[{idx}/{len(questions)}] Processing: {question_id}")
        print(f"  Question: {q.get('question', '')[:70]}...")
        
        try:
            initial_state = {
                "question": q.get("question", ""),
                "format_hint": format_hint,
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
            
            # Run the LangGraph workflow
            print(f"  Running workflow...")
            final_state = app.invoke(initial_state)
            
            # Normalize and validate final_answer
            final_answer_raw = final_state.get("final_answer")
            final_answer, is_valid = normalize_answer(final_answer_raw, format_hint)
            
            # Extract SQL
            sql_query = final_state.get("sql_query", "").strip()
            
            # Sanitize citations
            citations = sanitize_citations(final_state.get("citations", []), sql_query)
            
            # Truncate explanation
            explanation = truncate_explanation(final_state.get("explanation", ""))
            
            # Determine confidence
            has_error = final_state.get("sql_error") is not None
            confidence = 1.0 if (not has_error and is_valid) else 0.5 if (not has_error) else 0.0
            
            # Build Output Contract-compliant result
            output = {
                "id": question_id,
                "final_answer": final_answer,
                "sql": sql_query,
                "confidence": confidence,
                "explanation": explanation,
                "citations": citations
            }
            
            status_msg = "✓ Valid" if is_valid else "⚠ Normalized"
            print(f"  {status_msg}. Answer: {str(final_answer)[:50]}")
            results.append(output)
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            print(f"     Traceback: {traceback.format_exc()}")
            
            # Still output contract-compliant result with error info
            output = {
                "id": question_id,
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Processing error: {str(e)[:100]}",
                "citations": []
            }
            results.append(output)
    
    # Write results to output file (Output Contract format)
    print(f"\n--- Writing Results (Output Contract Format) ---")
    try:
        with open(out, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"✓ Wrote {len(results)} results to {out}")
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}")
    
    print(f"\n--- Summary ---")
    valid_count = sum(1 for r in results if r.get("confidence") == 1.0)
    partial_count = sum(1 for r in results if r.get("confidence") == 0.5)
    error_count = sum(1 for r in results if r.get("confidence") == 0.0)
    print(f"Valid: {valid_count}/{len(results)} | Partial: {partial_count} | Error: {error_count}")
    print(f"Output file: {out}")

if __name__ == "__main__":
    main()