"""
Quick health check script for the UC to Excel Converter
Run this to verify the app can start without errors
"""

import sys
import os

print("=" * 60)
print("UC to Excel Converter - Health Check")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
if sys.version_info < (3, 9):
    print("   ⚠ WARNING: Python 3.9+ recommended")
elif sys.version_info >= (3, 13):
    print("   ⚠ WARNING: Python 3.13 may have compatibility issues")
    print("   ℹ Use Python 3.11 (specified in runtime.txt)")
else:
    print("   ✓ Python version OK")

# Check required files
print("\n2. Required Files:")
required_files = [
    "app.py",
    "requirements.txt",
    "packages.txt",
    "runtime.txt",
    ".streamlit/config.toml"
]

all_files_exist = True
for file in required_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"   {status} {file}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n   ✗ Some required files are missing!")
    sys.exit(1)

# Try importing critical modules
print("\n3. Critical Dependencies:")
critical_modules = [
    ("streamlit", "Streamlit"),
    ("easyocr", "EasyOCR"),
    ("pdf2image", "pdf2image"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
]

import_success = True
for module, name in critical_modules:
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        import_success = False

if not import_success:
    print("\n   ✗ Some dependencies are missing!")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Check if app.py can be imported
print("\n4. App Import Test:")
try:
    # Just check if the file is valid Python
    with open("app.py", "r", encoding="utf-8") as f:
        compile(f.read(), "app.py", "exec")
    print("   ✓ app.py syntax is valid")
except SyntaxError as e:
    print(f"   ✗ Syntax error in app.py: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ Health check passed!")
print("=" * 60)
print("\nYou can now:")
print("  • Run locally: streamlit run app.py")
print("  • Deploy to Streamlit Cloud")
print("\nFor deployment, push to GitHub and connect at:")
print("  https://share.streamlit.io")
print("=" * 60)
