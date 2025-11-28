"""
optimize_sql_dspy.py

Uses DSPy to optimize the GenerateSQL module (NL -> SQL) using a small handcrafted
train/test split. Reports before/after execution success rates (valid-SQL / executes
against `data/northwind.sqlite`).

Usage:
  - Ensure Ollama is running and the model referenced in `agent/graph_hybrid.py`
    (default `qwen2:1.5b`) is available.
  - Run:
      python optimize_sql_dspy.py

Notes:
  - The script will try multiple DSPy optimizer entrypoints depending on installed
    DSPy version (e.g., `dspy.teleprompt.BootstrapFewShot` or
    `dspy.TextPrompt.BootstrapFewShot`).
  - If an LLM or DSPy optimizer isn't available, the script will exit gracefully
    and print instructions.
"""

import json
import traceback
import os
import sys
from typing import List

# Ensure project root is on sys.path so local packages like `tools` and `agent`
# can be imported when the script is executed directly.
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import dspy
except Exception as e:
    print("ERROR: DSPy not importable. Install dspy-ai and ensure it's on PYTHONPATH.")
    raise

from agent.dspy_signatures import GenerateSQL
from agent.tools.sqlite_tool import execute_sql, get_schema


def validate_sql_exec(example, pred, trace=None):
    """Metric used by optimizer: executes successfully against the sqlite DB."""
    sql = getattr(pred, 'sql_query', None)
    if not sql:
        return False
    res = execute_sql(sql)
    return not (isinstance(res, str) and res.startswith('SQL Error'))


def baseline_predict(predictor, examples: List[dict]):
    """Run predictor on examples and return list of generated SQLs."""
    outputs = []
    for ex in examples:
        try:
            pred = predictor(question=ex['question'], schema=ex['schema'], constraints=ex.get('constraints', ''))
            sql = getattr(pred, 'sql_query', '')
        except Exception as e:
            sql = ''
        outputs.append(sql)
    return outputs


def eval_exec_success(sqls: List[str]):
    successes = 0
    details = []
    for s in sqls:
        if not s or not s.strip():
            details.append((s, False, 'empty'))
            continue
        res = execute_sql(s)
        ok = not (isinstance(res, str) and res.startswith('SQL Error'))
        details.append((s, ok, res if isinstance(res, str) else f"rows={len(res)}"))
        if ok:
            successes += 1
    rate = successes / len(sqls) if sqls else 0.0
    return rate, details


def main():
    # Small handcrafted dataset (train + test). Keep it small for local budgets.
    schema = get_schema()

    train_examples = [
        {
            'question': 'List ProductName and UnitPrice for products in Beverages category.',
            'schema': schema,
            'constraints': "",
            'sql_query': "SELECT ProductName, UnitPrice FROM Products JOIN Categories ON Products.CategoryID = Categories.CategoryID WHERE CategoryName = 'Beverages' LIMIT 10"
        },
        {
            'question': 'What is the total revenue all-time?',
            'schema': schema,
            'constraints': "",
            'sql_query': "SELECT SUM(UnitPrice * Quantity * (1 - Discount)) as revenue FROM [Order Details]"
        },
        {
            'question': 'Top 3 products by revenue all-time.',
            'schema': schema,
            'constraints': "",
            'sql_query': "SELECT Products.ProductName, SUM(UnitPrice * Quantity * (1 - Discount)) AS TotalRevenue FROM Products JOIN [Order Details] ON Products.ProductID = [Order Details].ProductID GROUP BY Products.ProductID ORDER BY TotalRevenue DESC LIMIT 3"
        },
        {
            'question': 'Average Order Value (AOV) for Summer 1997 (Jun-Aug).',
            'schema': schema,
            'constraints': "",
            'sql_query': "SELECT ROUND(AVG(order_total), 2) FROM (SELECT OrderID, SUM(UnitPrice * Quantity * (1 - Discount)) AS order_total FROM [Order Details] GROUP BY OrderID) WHERE OrderDate BETWEEN '1997-06-01' AND '1997-08-31'"
        },
    ]

    test_examples = [
        {
            'question': "During Summer Beverages 1997 which category had highest quantity sold?",
            'schema': schema,
            'constraints': "",
            'sql_query': "SELECT Categories.CategoryName, SUM([Order Details].Quantity) as TotalQty FROM Categories JOIN Products ON Categories.CategoryID = Products.CategoryID JOIN [Order Details] ON Products.ProductID = [Order Details].ProductID JOIN Orders ON [Order Details].OrderID = Orders.OrderID WHERE Orders.OrderDate BETWEEN '1997-06-01' AND '1997-08-31' GROUP BY Categories.CategoryName ORDER BY TotalQty DESC LIMIT 1"
        },
        {
            'question': 'Total revenue from Beverages in Summer 1997.',
            'schema': schema,
            'constraints': "",
            'sql_query': "SELECT ROUND(SUM(UnitPrice * Quantity * (1 - Discount)), 2) as TotalRevenue FROM Products JOIN [Order Details] ON Products.ProductID = [Order Details].ProductID JOIN Orders ON [Order Details].OrderID = Orders.OrderID JOIN Categories ON Products.CategoryID = Categories.CategoryID WHERE Categories.CategoryName = 'Beverages' AND Orders.OrderDate BETWEEN '1997-06-01' AND '1997-08-31'"
        },
    ]

    print(f"Train examples: {len(train_examples)}, Test examples: {len(test_examples)}")

    # Baseline: attempt to use dspy.Predict(GenerateSQL)
    baseline_sqls = []
    try:
        predictor = dspy.Predict(GenerateSQL)
        print("Running baseline predictions using dspy.Predict(GenerateSQL) ...")
        baseline_sqls = baseline_predict(predictor, test_examples)
    except Exception as e:
        print("Baseline: failed to run dspy.Predict. Falling back to empty SQL baseline.")
        baseline_sqls = ["" for _ in test_examples]
        print(traceback.format_exc())

    baseline_rate, baseline_details = eval_exec_success(baseline_sqls)
    print("\nBaseline exec success rate:", baseline_rate)
    for s, ok, info in baseline_details:
        print(f" - OK={ok} | {info} | SQL={s[:80]}")

    # Try optimizer (BootstrapFewShot / Teleprompter variants)
    optimized_sqls = []
    try:
        # Detect possible optimizer names
        boot = None
        if hasattr(dspy, 'teleprompt') and hasattr(dspy.teleprompt, 'BootstrapFewShot'):
            boot = dspy.teleprompt.BootstrapFewShot(metric=validate_sql_exec, max_labeled_demos=4)
            print("Using dspy.teleprompt.BootstrapFewShot optimizer")
        elif hasattr(dspy, 'TextPrompt') and hasattr(dspy.TextPrompt, 'BootstrapFewShot'):
            boot = dspy.TextPrompt.BootstrapFewShot(metric=validate_sql_exec, max_labeled_demos=4)
            print("Using dspy.TextPrompt.BootstrapFewShot optimizer")
        else:
            print("No known DSPy optimizer entrypoint found in this DSPy version.")

        if boot is not None:
            # Train the optimizer on a tiny train set
            print("Compiling optimized SQL generator (this will call the LLM)...")
            optimized_module = boot.compile(dspy.ChainOfThought(GenerateSQL), trainset=[
                dspy.Example(
                    question=e['question'],
                    schema=e['schema'],
                    constraints=e.get('constraints', ''),
                    sql_query=e['sql_query']
                ).with_inputs('question', 'schema', 'constraints') for e in train_examples
            ])

            # Evaluate optimized module
            print("Running optimized predictions...")
            optimized_sqls = baseline_predict(optimized_module, test_examples)
        else:
            optimized_sqls = baseline_sqls

    except Exception as e:
        print("Optimizer failed or LLM unavailable.\n", traceback.format_exc())
        optimized_sqls = baseline_sqls

    opt_rate, opt_details = eval_exec_success(optimized_sqls)
    print("\nOptimized exec success rate:", opt_rate)
    for s, ok, info in opt_details:
        print(f" - OK={ok} | {info} | SQL={s[:80]}")

    # Summary
    print("\n=== Summary ===")
    print(f"Baseline success rate: {baseline_rate:.2f} ({sum(1 for _,ok,_ in baseline_details if ok)}/{len(baseline_details)})")
    print(f"Optimized success rate: {opt_rate:.2f} ({sum(1 for _,ok,_ in opt_details if ok)}/{len(opt_details)})")

    if opt_rate > baseline_rate:
        print("Optimizer improved exec-success rate on the tiny test split.")
    elif opt_rate == baseline_rate:
        print("No change in exec-success rate.")
    else:
        print("Exec-success rate decreased after optimization.")


if __name__ == '__main__':
    main()
