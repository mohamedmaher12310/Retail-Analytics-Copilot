import os, sys
print('CWD=', os.getcwd())
print('sys.path[0]=', sys.path[0])
print('sys.path[:5]=', sys.path[:5])
import importlib
importlib.invalidate_caches()
try:
    import rag
    print('Imported rag OK:', rag.__file__)
except Exception as e:
    print('Import rag failed:', repr(e))
