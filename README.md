# UC to Excel Converter

Professional tool for extracting Utilisation Certificate documents. Converts PDF UC documents into structured Excel format.

## Features

- **5-Section Extraction**: Heading, Details, Statements, Receipt table, Expenditure table
- **OCR Technology**: EasyOCR for text recognition
- **Table Extraction**: Camelot for structured table parsing
- **Export Options**: Excel (multi-sheet) and CSV formats
- **Error Handling**: Comprehensive error messages and logging

## Deployment on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with these files:
   - `app.py` (main app)
   - `requirements.txt` (Python dependencies)
   - `packages.txt` (system dependencies)
   - `runtime.txt` (Python version - forces 3.11)
   - `.streamlit/config.toml` (configuration)

### Troubleshooting Deployment

If you encounter errors during deployment:

1. **Check the logs** - Click "Manage app" → "Logs" to see detailed error messages
2. **Test imports** - Run `python test_imports.py` locally to verify dependencies
3. **Common issues**:
   - **"No module named 'distutils'"** - Fixed by using Python 3.11 (runtime.txt)
   - **"Failed to convert PDF"** - Ensure poppler-utils is in packages.txt
   - **OCR errors** - EasyOCR needs time to download models on first run
   - **Memory errors** - Large PDFs may exceed Streamlit Cloud limits (1GB RAM)

### Error Handling

The app includes comprehensive error handling:
- Detailed error messages for each processing stage
- Graceful fallbacks if table extraction fails
- Logging for debugging (check Streamlit Cloud logs)
- Per-page error handling (one page failure won't stop others)

## Local Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y poppler-utils ghostscript python3-tk

# Install Python dependencies
pip install -r requirements.txt

# Test imports
python test_imports.py

# Run the app
streamlit run app.py
```

## Usage

1. Upload a UC document PDF
2. Wait for processing (OCR + table extraction)
3. Review extracted data in structured format
4. Download as Excel or CSV

## Technology Stack

- **Streamlit**: Web interface
- **EasyOCR**: Text extraction
- **Camelot**: Table parsing
- **pdf2image**: PDF rendering
- **Pandas**: Data manipulation
- **OpenPyXL**: Excel generation

## File Structure

```
.
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── packages.txt             # System dependencies
├── runtime.txt              # Python version specification
├── test_imports.py          # Dependency testing script
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # This file
```

## Known Limitations

- Maximum file size: 200MB (configurable in config.toml)
- Memory: Large PDFs may fail on Streamlit Cloud (1GB RAM limit)
- Processing time: OCR is CPU-intensive, expect 10-30 seconds per page
- Table detection: Works best with bordered tables (lattice mode)

## Support

If you encounter issues:
1. Check the error message displayed in the app
2. Review Streamlit Cloud logs for detailed traceback
3. Verify all dependencies are installed (`test_imports.py`)
4. Ensure PDF is not corrupted and is text-based (not scanned images)
