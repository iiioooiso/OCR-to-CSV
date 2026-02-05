"""
Test script to verify all dependencies are installed correctly
Run this to diagnose import issues before running the main app
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠ {package_name or module_name} - ERROR: {str(e)}")
        return False

print("Testing Python dependencies...\n")
print(f"Python version: {sys.version}\n")

results = []

# Core dependencies
results.append(test_import("streamlit"))
results.append(test_import("easyocr"))
results.append(test_import("pdf2image"))
results.append(test_import("numpy"))
results.append(test_import("pandas"))
results.append(test_import("openpyxl"))
results.append(test_import("PIL", "Pillow"))
results.append(test_import("camelot"))
results.append(test_import("pdfplumber"))
results.append(test_import("fitz", "PyMuPDF"))
results.append(test_import("cv2", "opencv-python-headless"))
results.append(test_import("torch"))
results.append(test_import("torchvision"))

print(f"\n{'='*50}")
print(f"Results: {sum(results)}/{len(results)} packages OK")

if all(results):
    print("✓ All dependencies installed successfully!")
else:
    print("✗ Some dependencies are missing. Check errors above.")
    sys.exit(1)
