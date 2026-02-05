# UC to Excel Converter

Professional tool for extracting Utilisation Certificate documents. Converts PDF UC documents into structured Excel format.

## Features

- **5-Section Extraction**: Heading, Details, Statements, Receipt table, Expenditure table
- **OCR Technology**: EasyOCR for text recognition
- **Table Extraction**: Camelot for structured table parsing
- **Export Options**: Excel (multi-sheet) and CSV formats

## Deployment on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with these files:
   - `ocr.py` (main app)
   - `requirements.txt` (Python dependencies)
   - `packages.txt` (system dependencies)
   - `.streamlit/config.toml` (configuration)

## Local Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y poppler-utils ghostscript python3-tk

# Install Python dependencies
pip install -r requirements.txt

# Run the app
streamlit run ocr.py
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
