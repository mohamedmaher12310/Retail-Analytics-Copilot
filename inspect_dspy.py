import dspy, json
print('dspy version:', getattr(dspy, '__version__', 'unknown'))
attrs = [a for a in dir(dspy) if not a.startswith('_')]
print('--- attrs ---')
print(json.dumps(attrs, indent=2))
# print presence of common optimizer names
for name in ['teleprompt','teleprompter','TextPrompt','BootstrapFewShot','teleprompting','miprov2','MIPROv2']:
    print(name, '->', hasattr(dspy, name))
