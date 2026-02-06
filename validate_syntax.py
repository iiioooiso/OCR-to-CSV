"""Quick syntax validation for app.py"""
import ast
import sys

try:
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    ast.parse(code)
    print("✓ app.py syntax is valid!")
    sys.exit(0)
except SyntaxError as e:
    print(f"✗ Syntax error in app.py:")
    print(f"  Line {e.lineno}: {e.msg}")
    print(f"  {e.text}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
