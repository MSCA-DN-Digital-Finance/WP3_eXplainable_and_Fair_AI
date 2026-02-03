# check_imports.py
import ast
import importlib.util
import sys
from pathlib import Path

BUILTINS = set(sys.builtin_module_names)

def find_imports(pyfile):
    tree = ast.parse(Path(pyfile).read_text())
    modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                modules.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])

    return modules

def is_available(module):
    if module in BUILTINS:
        return True
    return importlib.util.find_spec(module) is not None

missing = []
for m in find_imports("create_preds_chronos.py"):
    if not is_available(m):
        missing.append(m)

if missing:
    print("Missing modules:")
    for m in missing:
        print("  -", m)
    sys.exit(1)
else:
    print("All imports satisfied.")
