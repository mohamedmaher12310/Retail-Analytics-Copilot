import dspy, json
attrs = [a for a in dir(dspy.TextPrompt) if not a.startswith('_')]
print(json.dumps(attrs, indent=2))
# Try to show callables
callables = [a for a in attrs if callable(getattr(dspy.TextPrompt, a))]
print('\ncallables:', json.dumps(callables, indent=2))
