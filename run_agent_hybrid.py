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
    Normalize final_answer to match format_hint strictly per Output Contract.
    Returns (normalized_answer, is_valid).
    
    Output Contract requires final_answer to match format_hint exactly:
    - int: integer value (0 if None)
    - float: float value ±0.01 tolerance (0.0 if None)
    - object/dict: dict ({} if None)
    - list: list ([] if None)
    - str: string ("" if None)
    """
    expected_type = parse_format_hint(format_hint)
    
    # Default values for each type when answer is None or invalid
    defaults = {
        "int": 0,
        "float": 0.0,
        "dict": {},
        "list": [],
        "str": ""
    }
    
    if final_answer is None:
        return defaults.get(expected_type, None), False
    
    # Reject "not applicable" or similar rejection strings
    if isinstance(final_answer, str):
        if 'not applicable' in final_answer.lower() or final_answer.lower() in ['na', 'n/a', 'no answer', 'none']:
            return defaults.get(expected_type, None), False
    
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
                    return defaults["int"], False
        
        elif expected_type == "float":
            if isinstance(final_answer, (int, float)):
                return round(float(final_answer), 2), True
            elif isinstance(final_answer, str):
                try:
                    return round(float(final_answer), 2), True
                except:
                    return defaults["float"], False
        
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
            return defaults["dict"], False
        
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
            return defaults["list"], False
        
        else:  # str
            return str(final_answer), True
    
    except Exception:
        return defaults.get(expected_type, None), False

def is_valid_citation(c_str):
    """
    Return True if c_str is a valid DB table name or doc chunk ID per Output Contract.
    Valid: simple alphanumeric table names (e.g., Orders, ProductPolicy)
           or doc IDs in format filename::chunkN (e.g., marketing_calendar::chunk0)
    """
    # Valid doc chunk ID: filename::chunkN
    if '::chunk' in c_str:
        return True
    
    # Valid table name: alphanumeric + underscores, no spaces or parens, reasonable length
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', c_str) and len(c_str) < 50:
        return True
    
    return False

def sanitize_citations(citations, sql_query):
    """
    Extract table names from SQL and merge with provided citations.
    Returns a list of citation strings: "TableName" or "doc_id::chunkN".
    Per Output Contract: citations must include only DB tables and doc chunk IDs,
    filtering out any explanatory text or invalid entries.
    """
    result = []
    
    # Extract table names from SQL if present
    if sql_query:
        # Simple regex to find table names (not perfect, but works for common cases)
        table_pattern = r'\b(?:FROM|JOIN|INTO|UPDATE|DELETE\s+FROM)\s+[`"]?(\w+)[`"]?'
        matches = re.findall(table_pattern, sql_query, re.IGNORECASE)
        result.extend(set(matches))
    
    # Add doc citations (filter out explanatory text)
    if isinstance(citations, list):
        for c in citations:
            if isinstance(c, str):
                c_str = c.strip()
                if is_valid_citation(c_str) and c_str not in result:
                    result.append(c_str)
    elif isinstance(citations, str):
        c_str = citations.strip()
        if is_valid_citation(c_str) and c_str not in result:
            result.append(c_str)
    
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
    Enforces strict Output Contract per specification:
    - final_answer must match format_hint exactly (int, float, dict, list, or str)
    - citations must include DB tables and doc chunk IDs only
    - Repair loop: retry up to 2 times on SQL errors or format mismatch
    - Confidence: 1.0 (valid), 0.5 (partial), 0.0 (error)
    """
    
    print("--- Starting Hybrid Agent Evaluation (with Output Contract Validation & Repair) ---")
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
        question_text = q.get("question", "")
        
        print(f"\n[{idx}/{len(questions)}] Processing: {question_id}")
        print(f"  Format: {format_hint} | Question: {question_text[:60]}...")
        
        # Repair loop: attempt up to 3 times (initial + 2 repairs)
        final_answer = None
        sql_query = ""
        citations = []
        explanation = ""
        sql_error = None
        is_valid = False
        repair_count = 0
        max_repairs = 2
        
        while repair_count <= max_repairs:
            try:
                initial_state = {
                    "question": question_text,
                    "format_hint": format_hint,
                    "repair_count": repair_count,
                    "sql_error": sql_error,
                    "classification": "",
                    "schema": "",
                    "doc_context": [],
                    "constraints": "",
                    "sql_query": sql_query,
                    "sql_result": None,
                    "final_answer": None,
                    "explanation": "",
                    "citations": []
                }
                
                # Run the LangGraph workflow
                final_state = app.invoke(initial_state)
                
                # Extract raw outputs
                final_answer_raw = final_state.get("final_answer")
                sql_query = final_state.get("sql_query", "").strip()
                sql_error = final_state.get("sql_error")
                doc_citations = final_state.get("citations", [])
                explanation_raw = final_state.get("explanation", "")
                
                # Normalize and validate final_answer against format_hint
                final_answer, is_valid = normalize_answer(final_answer_raw, format_hint)
                
                # Sanitize citations (DB tables + doc chunk IDs only)
                citations = sanitize_citations(doc_citations, sql_query)
                
                # Truncate explanation to 2 sentences (~250 chars)
                explanation = truncate_explanation(explanation_raw)
                
                # Check if we should retry
                if is_valid and sql_error is None:
                    # Success: valid answer and no SQL error
                    print(f"  [Attempt {repair_count + 1}] SUCCESS. Answer: {str(final_answer)[:50]}")
                    break
                elif repair_count < max_repairs and (sql_error is not None or not is_valid):
                    # Retry: SQL error or format mismatch and repairs remaining
                    if sql_error:
                        print(f"  [Attempt {repair_count + 1}] SQL error, retrying (repair {repair_count + 1}/{max_repairs})...")
                    else:
                        print(f"  [Attempt {repair_count + 1}] Format mismatch, retrying (repair {repair_count + 1}/{max_repairs})...")
                    repair_count += 1
                else:
                    # No more repairs or partial success
                    if is_valid or sql_error is None:
                        print(f"  [Attempt {repair_count + 1}] PARTIAL. Answer: {str(final_answer)[:50]}")
                    else:
                        print(f"  [Attempt {repair_count + 1}] FAILED. No valid answer after {repair_count} repairs.")
                    break
                    
            except Exception as e:
                print(f"  [Attempt {repair_count + 1}] ERROR: {str(e)[:80]}")
                if repair_count < max_repairs:
                    print(f"  Retrying (repair {repair_count + 1}/{max_repairs})...")
                    repair_count += 1
                else:
                    print(f"  No more repairs available.")
                    break
        
        # Determine confidence based on success criteria
        # 1.0: no SQL error AND valid format answer
        # 0.5: partial success (no SQL error OR valid format, but not both)
        # 0.0: complete failure (SQL error AND invalid format)
        has_sql_error = sql_error is not None
        
        if not has_sql_error and is_valid:
            confidence = 1.0  # Perfect: no SQL error and valid format
        elif (not has_sql_error) or is_valid:
            confidence = 0.5  # Partial: one of the two succeeded
        else:
            confidence = 0.0  # Failed: both SQL error and format mismatch
        
        # Build Output Contract-compliant result
        output = {
            "id": question_id,
            "final_answer": final_answer,
            "sql": sql_query,
            "confidence": confidence,
            "explanation": explanation,
            "citations": citations
        }
        
        print(f"Confidence: {confidence} | Citations: {citations}")
        results.append(output)
    
    # Write results to output file (Output Contract format)
    print(f"\n--- Writing Results (Output Contract Format) ---")
    try:
        with open(out, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} results to {out}")
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