import json
import re

def is_valid_citation(c_str):
    """Only accept simple table names or doc chunk IDs."""
    # Doc chunk ID: filename::chunkN
    if '::chunk' in c_str:
        return True
    
    # Reject if length > 50 (likely explanatory)
    if len(c_str) > 50:
        return False
    
    # Reject if contains any special chars or spaces
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', c_str):
        return False
    
    return True

def clean_citations(citations_raw):
    cleaned = []
    for c in citations_raw:
        c_str = str(c).strip()
        if is_valid_citation(c_str):
            cleaned.append(c_str)
    return sorted(list(set(cleaned)))

with open('outputs_hybrid.jsonl', 'r') as f:
    outputs = [json.loads(line) for line in f if line.strip()]

print("Regenerating outputs_hybrid.jsonl with clean citations...\n")
corrected = []
for item in outputs:
    clean_cites = clean_citations(item.get('citations', []))
    
    corrected_item = {
        'id': item.get('id'),
        'final_answer': item.get('final_answer'),
        'sql': item.get('sql', ''),
        'confidence': 0.0,
        'explanation': item.get('explanation', '').strip()[:250],
        'citations': clean_cites
    }
    corrected.append(corrected_item)
    print(f"{item['id']:<40} | final_answer: {item['final_answer']!r:<10} | citations: {clean_cites}")

with open('outputs_hybrid.jsonl', 'w') as f:
    for r in corrected:
        f.write(json.dumps(r) + '\n')

print(f"\nSuccess: Regenerated {len(corrected)} entries per Output Contract")



