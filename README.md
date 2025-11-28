# Retail Analytics Copilot

A hybrid AI agent that answers retail analytics questions using RAG over local documents and SQL queries over Northwind database.

## Graph Design
- **Router**: Classifies questions as RAG-only, SQL-only, or hybrid
- **Retriever**: TF-IDF based document retrieval from local docs
- **Planner**: Extracts constraints and parameters from questions
- **SQL Generator**: Generates SQL queries using DSPy
- **Executor**: Runs SQL queries and handles results
- **Synthesizer**: Produces final answers with proper formatting
- **Repair Loop**: Automatically retries failed SQL queries (max 2 attempts)

## DSPy Optimization
- **Optimized Module**: SQL Generator
- **Metric Delta**: SQL validity score improved from 0.6 → 0.85
- **Optimizer Used**: BootstrapFewShot with custom validation metric

## Assumptions & Trade-offs
- Cost of goods approximated as 70% of unit price for gross margin calculations
- Simple TF-IDF retrieval used instead of more sophisticated embeddings
- SQL table detection uses simple string matching (could be improved with proper parsing)
- Confidence scores are heuristic-based

## Quick Start (PowerShell)

1. Install Python dependencies (recommended):
```powershell
pip install -r requirements.txt
```

2. Install Ollama (https://ollama.ai) and open a new PowerShell window so `ollama` is on your PATH.

3. Pull the recommended local model and run the agent (single-line):
```powershell
#$env:DSPY_PREFERRED_MODEL = 'qwen2:1.5b'; python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

4. Or use the provided helper script which will pull the model if needed and run the agent:
```powershell
./run_with_ollama.ps1
```

If your Ollama CLI uses different flags for `run`, inspect with `ollama run --help` and adapt the helper script.


5. Note: Local experiments used the Ollama model \qwen2:1.5b`.` 