#!/usr/bin/env python
"""
Test script to validate Output Contract compliance.
Uses mock responses to avoid Ollama dependency.
"""

import json
import sys

# Mock workflow responses
mock_responses = [
    {
        "question_id": "rag_policy_beverages_return_days",
        "final_answer": 30,
        "sql_query": "",
        "sql_error": None,
        "explanation": "The product policy document states that unopened Beverages have a 30-day return window.",
        "citations": ["product_policy::chunk2"],
    },
    {
        "question_id": "hybrid_top_category_qty_summer_1997",
        "final_answer": {"category": "Beverages", "quantity": 15000},
        "sql_query": "SELECT Categories.CategoryName, SUM([Order Details].Quantity) as TotalQty FROM Categories JOIN Products ON Categories.CategoryID = Products.CategoryID JOIN [Order Details] ON Products.ProductID = [Order Details].ProductID JOIN Orders ON [Order Details].OrderID = Orders.OrderID WHERE Orders.OrderDate BETWEEN '1997-06-01' AND '1997-08-31' GROUP BY Categories.CategoryID, Categories.CategoryName ORDER BY TotalQty DESC LIMIT 1",
        "sql_error": None,
        "explanation": "During Summer Beverages 1997 (June-August), the Beverages category sold 15,000 units, the highest among all categories.",
        "citations": ["Orders", "Order Details", "Products", "Categories", "marketing_calendar::chunk1"],
    },
    {
        "question_id": "hybrid_aov_winter_1997",
        "final_answer": 1234.56,
        "sql_query": "SELECT AVG(total_order_value) FROM (SELECT OrderID, SUM(UnitPrice * Quantity * (1 - Discount)) as total_order_value FROM [Order Details] GROUP BY OrderID) WHERE OrderDate BETWEEN '1997-12-01' AND '1997-02-28'",
        "sql_error": None,
        "explanation": "The average order value during Winter Classics 1997 (Dec 1996 - Feb 1997) was $1,234.56 based on KPI definition.",
        "citations": ["Order Details", "Orders", "kpi_definitions::chunk5"],
    },
    {
        "question_id": "sql_top3_products_by_revenue_alltime",
        "final_answer": [
            {"product": "Queso Cabrales", "revenue": 5234.50},
            {"product": "Camembert Pierrot", "revenue": 4876.00},
            {"product": "Manjimup Dried Apples", "revenue": 4545.75}
        ],
        "sql_query": "SELECT Products.ProductName, SUM(UnitPrice * Quantity * (1 - Discount)) as TotalRevenue FROM Products JOIN [Order Details] ON Products.ProductID = [Order Details].ProductID GROUP BY Products.ProductID, Products.ProductName ORDER BY TotalRevenue DESC LIMIT 3",
        "sql_error": None,
        "explanation": "Top 3 products by lifetime revenue: Queso Cabrales ($5,234.50), Camembert Pierrot ($4,876.00), and Manjimup Dried Apples ($4,545.75).",
        "citations": ["Products", "Order Details"],
    },
    {
        "question_id": "hybrid_revenue_beverages_summer_1997",
        "final_answer": 25678.90,
        "sql_query": "SELECT ROUND(SUM(UnitPrice * Quantity * (1 - Discount)), 2) as TotalRevenue FROM Products JOIN [Order Details] ON Products.ProductID = [Order Details].ProductID JOIN Orders ON [Order Details].OrderID = Orders.OrderID JOIN Categories ON Products.CategoryID = Categories.CategoryID WHERE Categories.CategoryName = 'Beverages' AND Orders.OrderDate BETWEEN '1997-06-01' AND '1997-08-31'",
        "sql_error": None,
        "explanation": "Total revenue from Beverages category during Summer 1997 was $25,678.90, calculated using standard revenue formula.",
        "citations": ["Products", "Order Details", "Orders", "Categories", "marketing_calendar::chunk1"],
    },
    {
        "question_id": "hybrid_best_customer_margin_1997",
        "final_answer": {"customer": "ALFKI", "margin": 0.45},
        "sql_query": "SELECT Customers.CustomerID, SUM(UnitPrice * Quantity * (1 - Discount) - (UnitPrice * Quantity * (1 - Discount) * 0.70)) / SUM(UnitPrice * Quantity * (1 - Discount)) as GrossMargin FROM Customers JOIN Orders ON Customers.CustomerID = Orders.CustomerID JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID WHERE YEAR(Orders.OrderDate) = 1997 GROUP BY Customers.CustomerID ORDER BY GrossMargin DESC LIMIT 1",
        "sql_error": None,
        "explanation": "Customer ALFKI had the highest gross margin in 1997 at 45%, calculated using KPI definition with 70% COGS approximation.",
        "citations": ["Customers", "Orders", "Order Details", "kpi_definitions::chunk3"],
    },
]

def validate_output_contract(record):
    """Validate that a record follows the Output Contract."""
    errors = []
    
    # Check required fields
    required_fields = ["id", "final_answer", "sql", "confidence", "explanation", "citations"]
    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    
    # Validate types
    if "confidence" in record:
        conf = record["confidence"]
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            errors.append(f"Invalid confidence: {conf} (must be float 0.0-1.0)")
    
    if "sql" in record:
        if not isinstance(record["sql"], str):
            errors.append(f"SQL must be string, got {type(record['sql'])}")
    
    if "explanation" in record:
        exp = record["explanation"]
        if not isinstance(exp, str):
            errors.append(f"Explanation must be string, got {type(exp)}")
        if len(exp) > 250:
            errors.append(f"Explanation too long ({len(exp)} chars, max 250)")
    
    if "citations" in record:
        if not isinstance(record["citations"], list):
            errors.append(f"Citations must be list, got {type(record['citations'])}")
        for cite in record["citations"]:
            if not isinstance(cite, str):
                errors.append(f"Citation must be string, got {type(cite)}")
    
    return errors

# Generate test output
print("Generating test output with Output Contract validation...\n")
output_lines = []

for resp in mock_responses:
    record = {
        "id": resp["question_id"],
        "final_answer": resp["final_answer"],
        "sql": resp["sql_query"],
        "confidence": 1.0 if resp["sql_error"] is None else 0.0,
        "explanation": resp["explanation"][:250],  # Truncate to 250 chars
        "citations": sorted(list(set(resp["citations"])))  # Unique, sorted
    }
    
    errors = validate_output_contract(record)
    if errors:
        print(f"❌ {record['id']}:")
        for err in errors:
            print(f"   - {err}")
    else:
        print(f"✓ {record['id']}")
    
    output_lines.append(json.dumps(record))

# Write to file
output_file = "outputs_hybrid_contract_test.jsonl"
with open(output_file, 'w') as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"\n✓ Wrote {len(output_lines)} records to {output_file}")

# Display first record as sample
print(f"\nSample record (first line):")
print(json.dumps(json.loads(output_lines[0]), indent=2))
