"""
UC to Excel Converter
=====================

Professional tool for extracting Utilisation Certificate documents.
Converts PDF UC documents into structured Excel format with 5 sections:
1. Heading
2. Two-column data (labels and values)
3. Statements (numbered points)
4. Receipt details (tabular)
5. Expenditure details (tabular)

Technology: EasyOCR for text extraction, Camelot for table extraction.
"""

import os
import streamlit as st
import easyocr
from pdf2image import convert_from_bytes
import numpy as np
import pandas as pd
from io import BytesIO
import logging
import tempfile

# Try to import table extraction libraries
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="UC to Excel Converter", layout="wide", page_icon="📋")
st.title("UC to Excel Converter")
st.markdown("Professional extraction tool for Utilisation Certificate documents")

# ---------------- LOAD OCR ----------------
@st.cache_resource
def load_ocr():
    """Load OCR with optimized settings for faster processing"""
    try:
        # Use GPU if available, optimize for speed
        return easyocr.Reader(
            ['en'], 
            gpu=False,
            model_storage_directory=None,
            download_enabled=True,
            detector=True,
            recognizer=True,
            verbose=False
        )
    except Exception as e:
        st.error(f"Failed to load OCR model: {str(e)}")
        logger.error(f"OCR loading error: {str(e)}", exc_info=True)
        return None

ocr = load_ocr()

if ocr is None:
    st.error("OCR model failed to load. Please refresh the page or contact support.")
    st.stop()

# ---------------- STRUCTURED UC DOCUMENT EXTRACTION ----------------
def extract_uc_document_complete(pdf_path, page_num, image):
    """
    Complete UC document extraction combining OCR and Camelot.
    Returns structured JSON with all 5 sections.
    IMPROVED: Hardcoded keys, better value extraction, complete row extraction.
    """
    import json
    import re
    
    try:
        structured_data = {
            "heading": "",
            "two_column_data": [],
            "sentences": [],
            "receipt_details": [],
            "expenditure_details": []
        }
        
        # HARDCODED KEYS - Only the 6 main fields in UC documents
        REQUIRED_KEYS = [
            "Deposit Work ID",
            "Name of Work",
            "User Department",
            "Administrative Approval No",
            "Administrative Approval Date",
            "AA Amount in Rupees"
        ]
        
        # Convert PIL image to numpy array
        try:
            img_array = np.array(image)
        except Exception as e:
            logger.error(f"Image conversion error: {str(e)}")
            st.error(f"Failed to convert image: {str(e)}")
            return None
        
        # Run OCR with optimized settings for speed
        try:
            # Use lower confidence threshold and batch processing for speed
            result = ocr.readtext(
                img_array,
                detail=1,  # Get bounding boxes
                paragraph=False,  # Don't merge into paragraphs
                min_size=10,  # Minimum text size
                text_threshold=0.6,  # Lower threshold for faster processing
                low_text=0.3,
                link_threshold=0.3,
                canvas_size=2560,  # Limit canvas size for speed
                mag_ratio=1.0
            )
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            st.error(f"OCR failed: {str(e)}")
            return None
        
        if not result:
            return None
        
        # Sort OCR results by Y position (top to bottom)
        sorted_results = sorted(result, key=lambda x: min([p[1] for p in x[0]]))
        
        # Extract all text lines with position info
        all_lines = []
        for detection in sorted_results:
            text = detection[1].strip()
            box = detection[0]
            y_pos = min([p[1] for p in box])
            if text:
                all_lines.append({'text': text, 'y': y_pos})
        
        # Create text-only list for easier processing
        text_lines = [line['text'] for line in all_lines]
        
        # Build full text for global searching
        full_text = '\n'.join(text_lines)
        
        # DEBUG: Show what OCR extracted
        with st.expander("Debug: OCR Extracted Lines", expanded=False):
            for i, line in enumerate(text_lines[:50]):
                st.text(f"{i}: {line}")
            
            # Show full text for pattern searching
            st.markdown("---")
            st.markdown("**Full Text (first 1000 chars):**")
            st.text(full_text[:1000])
    
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        st.error(f"Error during extraction: {str(e)}")
        return None
    
    # 1. Extract Heading (look for "Utilisation Certificate")
    for i, line in enumerate(text_lines[:5]):
        if "Utilisation" in line or "Certificate" in line or "UTILISATION" in line or "CERTIFICATE" in line:
            structured_data["heading"] = line
            break
    
    # 2. Extract two-column data using HARDCODED KEYS with PRECISE matching
    # Create a dictionary to store found values
    found_keys = {}
    
    # SPECIAL HANDLING: Deposit Work ID has a specific format pattern
    # Pattern: numbers/letters/numbers (e.g., 2312/ZD37/01354)
    work_id_pattern = r'\b\d{3,5}[/\\][A-Z]{2,4}\d*[/\\]\d{3,6}\b'
    
    # Stop index for field extraction (before statements)
    stop_index = len(text_lines)
    for i, line in enumerate(text_lines):
        if re.match(r'^\d+\)\s+', line):
            stop_index = i
            break
    
    # Extract fields in specific order with targeted patterns
    
    # 1. DEPOSIT WORK ID - Search for pattern anywhere in text
    for i in range(stop_index):
        line = text_lines[i]
        if "deposit work id" in line.lower() or "work id" in line.lower():
            # Search this line and next 2 lines for the pattern
            search_text = line
            for j in range(i + 1, min(i + 3, stop_index)):
                search_text += " " + text_lines[j]
            
            match = re.search(work_id_pattern, search_text)
            if match:
                found_keys["Deposit Work ID"] = match.group(0)
                break
    
    # If not found with label, search entire text for the pattern
    if "Deposit Work ID" not in found_keys:
        match = re.search(work_id_pattern, full_text[:full_text.find('1)') if '1)' in full_text else len(full_text)])
        if match:
            found_keys["Deposit Work ID"] = match.group(0)
    
    # Additional fallback: Search all lines for Work ID pattern
    if "Deposit Work ID" not in found_keys:
        for line in text_lines[:stop_index]:
            match = re.search(work_id_pattern, line)
            if match:
                found_keys["Deposit Work ID"] = match.group(0)
                break
    
    # 2. NAME OF WORK - Look for the label and collect full multi-line value
    name_of_work_found = False
    for i in range(stop_index):
        line = text_lines[i]
        if "name of work" in line.lower() and not name_of_work_found:
            # Extract value after colon
            if ':' in line:
                value = line.split(':', 1)[1].strip()
            else:
                value = ""
            
            # Collect continuation lines
            full_value = value
            j = i + 1
            while j < stop_index:
                next_line = text_lines[j].strip()
                
                # Stop if we hit another key (but not if it's part of the work name)
                is_key_line = False
                for k in REQUIRED_KEYS:
                    if k != "Name of Work" and k.lower() in next_line.lower() and ':' in next_line:
                        is_key_line = True
                        break
                
                if is_key_line:
                    break
                
                # Stop if we hit a Work ID pattern
                if re.search(work_id_pattern, next_line):
                    break
                
                # Stop if line is empty or very short
                if not next_line or len(next_line) < 3:
                    j += 1
                    continue
                
                # Stop if we hit a statement
                if re.match(r'^\d+\)', next_line):
                    break
                
                # Add to value
                full_value += " " + next_line
                j += 1
            
            if full_value.strip():
                found_keys["Name of Work"] = full_value.strip()
                name_of_work_found = True
            break
    
    # 3. USER DEPARTMENT - Look for the label
    for i in range(stop_index):
        line = text_lines[i]
        if "user department" in line.lower() or "department" in line.lower():
            # Extract value after colon
            if ':' in line:
                value = line.split(':', 1)[1].strip()
            else:
                value = ""
            
            # Check next line if empty
            if (not value or len(value) < 3) and i + 1 < stop_index:
                next_line = text_lines[i + 1].strip()
                if next_line and not any(k.lower() in next_line.lower() for k in REQUIRED_KEYS):
                    value = next_line
            
            if value.strip():
                found_keys["User Department"] = value.strip()
            break
    
    # 4. ADMINISTRATIVE APPROVAL NO - Look for specific pattern
    approval_no_pattern = r'[A-Z]{2,4}/\d{4}/\d{1,2}/[A-Z\.\d/\(\)]+' 
    for i in range(stop_index):
        line = text_lines[i]
        if "administrative approval no" in line.lower() or "approval no" in line.lower():
            # Extract value after colon
            if ':' in line:
                value = line.split(':', 1)[1].strip()
            else:
                value = ""
            
            # Check next line if empty
            if (not value or len(value) < 3) and i + 1 < stop_index:
                next_line = text_lines[i + 1].strip()
                if next_line and not any(k.lower() in next_line.lower() for k in REQUIRED_KEYS):
                    value = next_line
            
            # Validate it looks like an approval number (not a date or amount)
            if value and not value.startswith('Rs') and not re.match(r'^\d{2}/\d{2}/\d{4}', value):
                found_keys["Administrative Approval No"] = value.strip()
            break
    
    # If not found, search for the pattern
    if "Administrative Approval No" not in found_keys:
        match = re.search(approval_no_pattern, full_text[:full_text.find('1)') if '1)' in full_text else len(full_text)])
        if match:
            found_keys["Administrative Approval No"] = match.group(0)
    
    # 5. ADMINISTRATIVE APPROVAL DATE - Look for date pattern
    date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
    for i in range(stop_index):
        line = text_lines[i]
        if "administrative approval date" in line.lower() or "approval date" in line.lower():
            # Extract value after colon
            if ':' in line:
                value = line.split(':', 1)[1].strip()
            else:
                value = ""
            
            # Check next line if empty
            if (not value or len(value) < 3) and i + 1 < stop_index:
                next_line = text_lines[i + 1].strip()
                if next_line:
                    value = next_line
            
            # Extract date pattern
            match = re.search(date_pattern, value)
            if match:
                found_keys["Administrative Approval Date"] = match.group(0) + " (dd/mm/yyyy)"
            break
    
    # If not found, search for date near "approval date" text
    if "Administrative Approval Date" not in found_keys:
        for i in range(stop_index):
            if "approval date" in text_lines[i].lower():
                search_text = text_lines[i]
                if i + 1 < stop_index:
                    search_text += " " + text_lines[i + 1]
                match = re.search(date_pattern, search_text)
                if match:
                    found_keys["Administrative Approval Date"] = match.group(0) + " (dd/mm/yyyy)"
                    break
    
    # Additional fallback: Search all lines for date pattern (dd/mm/yyyy format)
    if "Administrative Approval Date" not in found_keys:
        for line in text_lines[:stop_index]:
            # Look for date that's NOT in a statement
            if not re.match(r'^\d+\)', line):
                match = re.search(date_pattern, line)
                if match:
                    # Make sure it's not in the statements section
                    date_value = match.group(0)
                    # Check if this line also contains "date" keyword
                    if "date" in line.lower() or text_lines.index(line) < 10:
                        found_keys["Administrative Approval Date"] = date_value + " (dd/mm/yyyy)"
                        break
    
    # 6. AA AMOUNT IN RUPEES - Look for amount pattern
    amount_pattern = r'Rs\.?\s*\d{5,}'
    for i in range(stop_index):
        line = text_lines[i]
        if "aa amount" in line.lower() or "amount in rupees" in line.lower():
            # Extract value after colon
            if ':' in line:
                value = line.split(':', 1)[1].strip()
            else:
                value = ""
            
            # Check next line if empty
            if (not value or len(value) < 3) and i + 1 < stop_index:
                next_line = text_lines[i + 1].strip()
                if next_line:
                    value = next_line
            
            # Extract amount pattern
            match = re.search(amount_pattern, value)
            if match:
                found_keys["AA Amount in Rupees"] = match.group(0)
            break
    
    # If not found, search for "Rs. 15000000" pattern near "AA Amount" text
    if "AA Amount in Rupees" not in found_keys:
        for i in range(stop_index):
            if "aa amount" in text_lines[i].lower() or "amount in rupees" in text_lines[i].lower():
                search_text = text_lines[i]
                if i + 1 < stop_index:
                    search_text += " " + text_lines[i + 1]
                if i + 2 < stop_index:
                    search_text += " " + text_lines[i + 2]
                match = re.search(amount_pattern, search_text)
                if match:
                    found_keys["AA Amount in Rupees"] = match.group(0)
                    break
    
    # Additional fallback: Search all lines for amount pattern
    if "AA Amount in Rupees" not in found_keys:
        for line in text_lines[:stop_index]:
            # Look for Rs. followed by large number (5+ digits)
            match = re.search(amount_pattern, line)
            if match:
                amount_value = match.group(0)
                # Make sure it's a large amount (likely the AA amount, not a small amount)
                amount_num = int(re.sub(r'[^\d]', '', amount_value))
                if amount_num >= 1000000:  # At least 10 lakhs
                    found_keys["AA Amount in Rupees"] = amount_value
                    break
    
    # Add all found keys to structured data in order (ONLY the 6 main fields for UC)
    # Based on the actual UC format, we only need these 6 fields:
    UC_MAIN_FIELDS = [
        "Deposit Work ID",
        "Name of Work",
        "User Department",
        "Administrative Approval No",
        "Administrative Approval Date",
        "AA Amount in Rupees"
    ]
    
    for key in UC_MAIN_FIELDS:
        if key in found_keys:
            structured_data["two_column_data"].append({
                "label": key,
                "value": found_keys[key]
            })
        else:
            # Add empty entry to maintain structure
            structured_data["two_column_data"].append({
                "label": key,
                "value": ""
            })
    
    # 3. Extract ALL sentences (numbered points like "1)", "2)", etc.)
    # More aggressive extraction to catch all statements
    statement_lines = []
    i = 0
    while i < len(text_lines):
        line = text_lines[i]
        
        # Match lines starting with number followed by )
        match = re.match(r'^(\d+)\)\s*(.+)', line)
        if match:
            statement_num = match.group(1)
            statement_text = match.group(2).strip()
            
            # Collect continuation lines (lines that don't start with number or key)
            full_statement = statement_text
            j = i + 1
            while j < len(text_lines):
                next_line = text_lines[j].strip()
                
                # Stop if we hit another numbered statement, table section, or key
                if re.match(r'^\d+\)', next_line):
                    break
                if "Receipt Details" in next_line or "Expenditure Details" in next_line:
                    break
                if any(k.lower() in next_line.lower() for k in REQUIRED_KEYS):
                    break
                
                # Check if it's a continuation (not a table row)
                if next_line and not re.match(r'^[\d\s\-/]+$', next_line):
                    full_statement += " " + next_line
                    j += 1
                else:
                    break
            
            statement_lines.append(f"{statement_num}) {full_statement}")
            i = j
        else:
            i += 1
    
    structured_data["sentences"] = statement_lines
    
    # 4 & 5. Extract tables using Camelot with IMPROVED extraction
    try:
        # Use both lattice and stream modes for better coverage
        tables = []
        
        # Try lattice first (for bordered tables)
        try:
            lattice_tables = camelot.read_pdf(
                pdf_path, 
                pages=str(page_num), 
                flavor='lattice',
                line_scale=40,  # Detect faint lines
                copy_text=['v']  # Vertical text
            )
            tables.extend(lattice_tables)
        except Exception as e:
            logger.warning(f"Lattice extraction failed: {str(e)}")
        
        # Try stream mode as fallback (for borderless tables)
        if len(tables) == 0:
            try:
                stream_tables = camelot.read_pdf(
                    pdf_path, 
                    pages=str(page_num), 
                    flavor='stream',
                    edge_tol=50,  # Tolerance for detecting table edges
                    row_tol=10    # Tolerance for row detection
                )
                tables.extend(stream_tables)
            except Exception as e:
                logger.warning(f"Stream extraction failed: {str(e)}")
        
        for table in tables:
            df = table.df
            
            if df.empty:
                continue
            
            # Clean the dataframe - but keep ALL rows with any data
            df = df.replace('', pd.NA)
            # Only remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            # Remove only rows that are completely empty
            df = df.dropna(how='all')
            df = df.reset_index(drop=True)
            
            if df.empty:
                continue
            
            # Detect which table this is
            if len(df) > 0:
                # Check first few rows for keywords
                first_rows_text = ' '.join(df.iloc[:2].astype(str).values.flatten()).lower()
                
                # Check for Receipt table keywords
                if ('date' in first_rows_text and 'received' in first_rows_text) or 'receipt' in first_rows_text:
                    # Keep ALL rows including those with partial data
                    structured_data["receipt_details"] = df.to_dict('records')
                # Check for Expenditure table keywords
                elif 'financial' in first_rows_text or 'bill' in first_rows_text or 'expenditure' in first_rows_text or 'remark' in first_rows_text:
                    # Keep ALL rows - especially important for Remarks column
                    structured_data["expenditure_details"] = df.to_dict('records')
                else:
                    # Fallback logic based on order
                    if structured_data["receipt_details"]:
                        structured_data["expenditure_details"] = df.to_dict('records')
                    else:
                        structured_data["receipt_details"] = df.to_dict('records')
                        
    except Exception as e:
        st.warning(f"Table extraction warning: {str(e)}")
        logger.warning(f"Table extraction error: {str(e)}", exc_info=True)
    
    return structured_data


def convert_uc_to_excel_format(structured_data):
    """
    Convert UC structured data to Excel format matching PDF layout.
    Returns list of dataframes for sequential writing.
    IMPROVED: Better handling of empty cells and remarks column.
    """
    blocks = []
    
    # Block 1: Heading
    if structured_data.get("heading"):
        df_heading = pd.DataFrame([[structured_data["heading"]]])
        blocks.append({
            'label': 'Heading',
            'df': df_heading
        })
    
    # Block 2: Two-column data
    if structured_data.get("two_column_data"):
        rows = []
        for item in structured_data["two_column_data"]:
            rows.append([item["label"], item["value"]])
        df_two_col = pd.DataFrame(rows, columns=['Label', 'Value'])
        blocks.append({
            'label': 'Details',
            'df': df_two_col
        })
    
    # Block 3: Sentences - Keep ALL statements
    if structured_data.get("sentences"):
        rows = [[s] for s in structured_data["sentences"]]
        df_sentences = pd.DataFrame(rows, columns=['Statement'])
        blocks.append({
            'label': 'Statements',
            'df': df_sentences
        })
    
    # Block 4: Receipt Details - Keep rows with at least some data
    if structured_data.get("receipt_details"):
        df_receipt = pd.DataFrame(structured_data["receipt_details"])
        # Only remove completely empty columns, keep rows with partial data
        df_receipt = df_receipt.dropna(axis=1, how='all')
        # Replace NaN with empty string for better display
        df_receipt = df_receipt.fillna('')
        if not df_receipt.empty:
            blocks.append({
                'label': 'Receipt Details',
                'df': df_receipt
            })
    
    # Block 5: Expenditure Details - Keep ALL rows and handle Remarks properly
    if structured_data.get("expenditure_details"):
        df_expenditure = pd.DataFrame(structured_data["expenditure_details"])
        # Only remove completely empty columns
        df_expenditure = df_expenditure.dropna(axis=1, how='all')
        # Replace NaN with empty string
        df_expenditure = df_expenditure.fillna('')
        
        # Ensure Remarks column is preserved if it exists
        if not df_expenditure.empty:
            blocks.append({
                'label': 'Expenditure Details',
                'df': df_expenditure
            })
    
    return blocks
def extract_text_pdfplumber(pdf_path, page_num):
    """
    Extract simple text content using pdfplumber.
    Returns dataframe with all text content (excluding tables).
    Dynamically detects layout from PDF without hardcoded values.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num > len(pdf.pages):
                return None
            
            page = pdf.pages[page_num - 1]
            
            # Extract text preserving layout
            text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3)
            
            if not text or text.strip() == "":
                return None
            
            # Split into lines
            lines = text.split('\n')
            
            rows = []
            import re
            skip_next = False
            
            for i, line in enumerate(lines):
                # Skip if marked by previous iteration
                if skip_next:
                    skip_next = False
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                # Skip lines that look like table headers or have many columns
                parts = re.split(r'\s{2,}', line)
                
                # If line has more than 4 parts, it's likely a table - skip it
                if len(parts) > 4:
                    continue
                
                # Check if line is mostly numbers separated by spaces (table row)
                number_count = len(re.findall(r'\b\d+\b', line))
                if number_count > 3:
                    continue
                
                # Now handle the line based on content
                # Look for colon separator (common in forms: "Label : Value")
                if ':' in line:
                    # Split by colon
                    colon_parts = line.split(':', 1)
                    if len(colon_parts) == 2:
                        label = colon_parts[0].strip()
                        value = colon_parts[1].strip()
                        
                        # If value is empty, check next line
                        if not value and i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            # Check if next line is a continuation (not a label with colon)
                            if next_line and ':' not in next_line:
                                # Check if it's not a table line
                                next_parts = re.split(r'\s{2,}', next_line)
                                next_numbers = len(re.findall(r'\b\d+\b', next_line))
                                if len(next_parts) <= 4 and next_numbers <= 3:
                                    value = next_line
                                    skip_next = True  # Mark to skip this line in next iteration
                        
                        rows.append([label, value])
                        continue
                
                # If no colon, try to split by multiple spaces
                space_parts = [p.strip() for p in re.split(r'\s{3,}', line) if p.strip()]
                
                if len(space_parts) == 0:
                    continue
                elif len(space_parts) == 1:
                    # Single item - put in first column only
                    rows.append([space_parts[0], ''])
                elif len(space_parts) == 2:
                    # Two items - perfect
                    rows.append([space_parts[0], space_parts[1]])
                elif len(space_parts) == 3:
                    # Three items - first in col1, rest in col2
                    rows.append([space_parts[0], f"{space_parts[1]} {space_parts[2]}"])
                elif len(space_parts) == 4:
                    # Four items - first two in col1, last two in col2
                    rows.append([f"{space_parts[0]} {space_parts[1]}", f"{space_parts[2]} {space_parts[3]}"])
            
            if not rows:
                return None
            
            # Create dataframe
            df = pd.DataFrame(rows, columns=['Column1', 'Column2'])
            return df
            
    except Exception as e:
        return None

# ---------------- TABLE EXTRACTION WITH CAMELOT ----------------
def extract_tables_camelot(pdf_path, page_num):
    """
    Extract tables from PDF using Camelot LATTICE mode only.
    Stream mode creates too many blanks - DISABLED.
    Returns list of table dataframes.
    """
    tables_data = []
    
    # ONLY use lattice mode (tables with borders)
    # Stream mode disabled - creates too many blank cells
    try:
        tables_lattice = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
        for table in tables_lattice:
            df = table.df
            if not df.empty:
                # Clean the dataframe
                df = df.replace('', pd.NA)
                df = df.dropna(how='all')  # Remove completely empty rows
                df = df.dropna(axis=1, how='all')  # Remove completely empty columns
                df = df.reset_index(drop=True)
                
                if not df.empty:
                    tables_data.append({
                        'df': df,
                        'mode': 'lattice'
                    })
    except Exception as e:
        pass
    
    return tables_data

# ---------------- HYBRID EXTRACTION: SEQUENTIAL WRITING ----------------
def extract_hybrid_sequential(pdf_path, page_num):
    """
    CORRECT APPROACH - Sequential extraction:
    1. Extract text with pdfplumber → 1 column
    2. Extract tables with Camelot → many columns
    3. Return them separately (DON'T merge!)
    
    Returns: list of blocks to write sequentially
    """
    blocks = []
    
    # Block 1: Text (headers, labels, etc.)
    text_df = extract_text_pdfplumber(pdf_path, page_num)
    if text_df is not None and not text_df.empty:
        blocks.append({
            'type': 'text',
            'df': text_df,
            'label': 'Text Content'
        })
    
    # Block 2+: Tables (one block per table)
    tables = extract_tables_camelot(pdf_path, page_num)
    for i, table_data in enumerate(tables):
        blocks.append({
            'type': 'table',
            'df': table_data['df'],
            'label': f"Table {i+1} ({table_data['mode']})"
        })
    
    return blocks

# ---------------- EXTRACT TEXT WITH OCR (SIMPLE FORMAT) ----------------
def extract_text_simple_ocr(image):
    """
    Extract text using OCR and format it simply (max 2 columns).
    For non-tabular content like headers, paragraphs, etc.
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Run OCR
    result = ocr.readtext(img_array)
    
    if not result:
        return pd.DataFrame()
    
    blocks = []
    for detection in result:
        box = detection[0]
        text = detection[1]
        
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        x_min = min(x_coords)
        y_min = min(y_coords)
        
        blocks.append({
            "text": text,
            "x": x_min,
            "y": y_min
        })
    
    if not blocks:
        return pd.DataFrame()
    
    # Sort by Y position
    blocks.sort(key=lambda b: b['y'])
    
    # Detect if there are two distinct columns
    x_positions = [b['x'] for b in blocks]
    x_positions.sort()
    
    # Check for column split (if there's a significant gap in X positions)
    has_two_columns = False
    mid_x = None
    
    if len(x_positions) > 2:
        # Find the largest gap in X positions
        gaps = []
        for i in range(len(x_positions) - 1):
            gap = x_positions[i + 1] - x_positions[i]
            gaps.append((gap, x_positions[i]))
        
        gaps.sort(reverse=True)
        largest_gap = gaps[0][0]
        
        # If largest gap is significant, we have two columns
        if largest_gap > 200:
            has_two_columns = True
            mid_x = gaps[0][1] + largest_gap / 2
    
    if has_two_columns:
        # Two column layout
        left_blocks = [b for b in blocks if b['x'] < mid_x]
        right_blocks = [b for b in blocks if b['x'] >= mid_x]
        
        left_blocks.sort(key=lambda b: b['y'])
        right_blocks.sort(key=lambda b: b['y'])
        
        # Create rows by pairing blocks at similar Y positions
        table_data = []
        max_len = max(len(left_blocks), len(right_blocks))
        
        for i in range(max_len):
            left_text = left_blocks[i]['text'] if i < len(left_blocks) else ""
            right_text = right_blocks[i]['text'] if i < len(right_blocks) else ""
            table_data.append([left_text, right_text])
        
        return pd.DataFrame(table_data)
    else:
        # Single column layout - merge nearby blocks
        merged_text = []
        current_text = blocks[0]['text']
        last_y = blocks[0]['y']
        
        for i in range(1, len(blocks)):
            block = blocks[i]
            
            # If blocks are close vertically, merge them
            if abs(block['y'] - last_y) <= 30:
                current_text += " " + block['text']
            else:
                merged_text.append([current_text])
                current_text = block['text']
            
            last_y = block['y']
        
        merged_text.append([current_text])
        
        return pd.DataFrame(merged_text)

# ---------------- STREAMLIT UI ----------------
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_pdf = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload a PDF file to extract tables and text"
    )

with col2:
    st.markdown("### Configuration")
    
    st.info("Structured extraction with 5 sections: Heading, Details, Statements, Receipt table, Expenditure table")
    
    st.markdown("**Key Features:**")
    st.markdown("- Hardcoded field extraction (no missing keys)")
    st.markdown("- Complete statement extraction")
    st.markdown("- Full table row preservation")
    st.markdown("- Optimized for speed")
    
    dpi = st.selectbox(
        "Image Quality (DPI)",
        options=[200, 300, 400],
        index=1,
        help="Higher DPI provides better quality but slower processing"
    )
    
    # Library status
    st.markdown("### System Status")
    st.write(f"Camelot: {'Available' if CAMELOT_AVAILABLE else 'Not installed'}")
    st.write(f"EasyOCR: Available")

if uploaded_pdf:
    st.divider()
    
    try:
        with st.spinner("Processing document..."):
            # Save PDF to temporary file (required for Camelot)
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    tmp_pdf_path = tmp_file.name
            except Exception as e:
                st.error(f"Failed to save PDF: {str(e)}")
                logger.error(f"PDF save error: {str(e)}", exc_info=True)
                st.stop()
            
            # Get number of pages
            try:
                pdf_bytes = open(tmp_pdf_path, 'rb').read()
            except Exception as e:
                st.error(f"Failed to read PDF: {str(e)}")
                logger.error(f"PDF read error: {str(e)}", exc_info=True)
                if os.path.exists(tmp_pdf_path):
                    os.unlink(tmp_pdf_path)
                st.stop()
            
            # Convert PDF to images
            try:
                images = convert_from_bytes(pdf_bytes, dpi=dpi)
                num_pages = len(images)
            except Exception as e:
                st.error(f"Failed to convert PDF to images: {str(e)}")
                st.info("This might be due to missing poppler-utils. Check deployment logs.")
                logger.error(f"PDF conversion error: {str(e)}", exc_info=True)
                if os.path.exists(tmp_pdf_path):
                    os.unlink(tmp_pdf_path)
                st.stop()
            
            st.success(f"Document loaded: {num_pages} page(s) detected")
            
            page_data = {}  # Store structured data per page
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Track extraction statistics
            total_keys_found = 0
            total_statements = 0
            total_receipt_rows = 0
            total_expenditure_rows = 0
            
            for i, img in enumerate(images):
                page_num = i + 1
                status_text.text(f"Processing page {page_num} of {num_pages}...")
                progress_bar.progress(page_num / num_pages)
                
                try:
                    # Extract structured UC document
                    structured_data = extract_uc_document_complete(tmp_pdf_path, page_num, img)
                    
                    if structured_data:
                        # Convert to Excel format blocks
                        blocks = convert_uc_to_excel_format(structured_data)
                        
                        if blocks:
                            page_data[f"Page_{page_num}"] = {
                                'structured': structured_data,
                                'blocks': blocks
                            }
                            
                            # Update statistics
                            total_keys_found += len([item for item in structured_data.get('two_column_data', []) if item['value']])
                            total_statements += len(structured_data.get('sentences', []))
                            total_receipt_rows += len(structured_data.get('receipt_details', []))
                            total_expenditure_rows += len(structured_data.get('expenditure_details', []))
                            
                except Exception as e:
                    st.warning(f"Error processing page {page_num}: {str(e)}")
                    logger.error(f"Page {page_num} processing error: {str(e)}", exc_info=True)
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            # Clean up temp file
            try:
                if os.path.exists(tmp_pdf_path):
                    os.unlink(tmp_pdf_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {str(e)}")
    
    except Exception as e:
        st.error(f"Unexpected error during processing: {str(e)}")
        logger.error(f"Main processing error: {str(e)}", exc_info=True)
        st.info("Please try again or contact support if the issue persists.")
        st.stop()
    
    if not page_data:
        st.error("No content detected in the document")
        st.stop()
    
    # Show extraction summary
    st.success(f" Extraction completed: {len(page_data)} page(s) processed")
    
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fields Extracted", f"{total_keys_found}/6")
    with col2:
        st.metric("Statements", total_statements)
    with col3:
        st.metric("Receipt Rows", total_receipt_rows)
    with col4:
        st.metric("Expenditure Rows", total_expenditure_rows)
    
    # Display structured data
    st.divider()
    st.subheader("Extracted Document Structure")
    
    try:
        tab_list = st.tabs([f"Page {i+1}" for i in range(len(page_data))])
        
        for idx, (page_name, data) in enumerate(page_data.items()):
            with tab_list[idx]:
                structured = data['structured']
                
                # Show JSON structure
                with st.expander("View JSON Structure", expanded=False):
                    st.json(structured)
                
                # Show formatted blocks
                for block in data['blocks']:
                    st.markdown(f"**{block['label']}**")
                    st.dataframe(block['df'], use_container_width=True, height=min(len(block['df']) * 35 + 38, 400))
                    st.markdown("---")
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        logger.error(f"Display error: {str(e)}", exc_info=True)
    
    # Pre-generate Excel file (to avoid UI hanging)
    try:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            for page_name, data in page_data.items():
                current_row = 0
                
                for block in data['blocks']:
                    df = block['df']
                    
                    # Write block label
                    label_df = pd.DataFrame([[f"=== {block['label']} ==="]])
                    label_df.to_excel(
                        writer,
                        sheet_name=page_name,
                        index=False,
                        header=False,
                        startrow=current_row
                    )
                    current_row += 1
                    
                    # Write block data
                    df.to_excel(
                        writer,
                        sheet_name=page_name,
                        index=False,
                        header=True,
                        startrow=current_row
                    )
                    current_row += len(df) + 2  # +2 for header and blank row
                
                # Enable text wrapping
                worksheet = writer.sheets[page_name]
                for row in worksheet.iter_rows():
                    for cell in row:
                        cell.alignment = cell.alignment.copy(wrap_text=True)
        
        excel_buffer.seek(0)
    except Exception as e:
        st.error(f"Failed to generate Excel file: {str(e)}")
        logger.error(f"Excel generation error: {str(e)}", exc_info=True)
        excel_buffer = None
    
    # Export to Excel/CSV
    st.divider()
    st.subheader("Export Options")
    
    # Pre-generate CSV data (to avoid UI hanging on button click)
    try:
        all_rows = []
        for data in page_data.values():
            for block in data['blocks']:
                # Add block label
                all_rows.append([f"=== {block['label']} ==="])
                # Add block data
                df = block['df']
                for _, row in df.iterrows():
                    all_rows.append(row.tolist())
                # Add blank row
                all_rows.append([])
        
        combined_df = pd.DataFrame(all_rows)
        csv_data = combined_df.to_csv(index=False, header=False)
    except Exception as e:
        st.error(f"Failed to generate CSV: {str(e)}")
        logger.error(f"CSV generation error: {str(e)}", exc_info=True)
        csv_data = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        if excel_buffer:
            st.download_button(
                label="Download Excel (Multi-sheet)",
                data=excel_buffer,
                file_name=f"{uploaded_pdf.name.replace('.pdf', '')}_UC_extracted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.error("Excel export unavailable")
    
    with col2:
        if csv_data:
            st.download_button(
                label="Download CSV (Combined)",
                data=csv_data,
                file_name=f"{uploaded_pdf.name.replace('.pdf', '')}_UC_extracted.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.error("CSV export unavailable")

else:
    st.info("Upload a UC document PDF to begin extraction")
    
    st.divider()
    st.markdown("### How It Works")
    st.markdown("""
    **Structured Extraction Process:**
    
    1. **Heading**: Document title extraction
    2. **Details**: Label-value pairs (Work ID, Department, Approval details)
    3. **Statements**: Numbered certification points
    4. **Receipt Details**: Tabular data with dates, amounts, and verification
    5. **Expenditure Details**: Financial year, bill details, and cumulative data
    
    **Technology Stack:**
    - EasyOCR: Text recognition and extraction
    - Camelot: Table structure detection and parsing
    - Structured JSON: Organized data format
    - Excel/CSV: Export with text wrapping and proper formatting
    
    **Output:** Clean, structured Excel file matching original document layout
    """)
    
    # Show installation instructions if libraries are missing
    if not CAMELOT_AVAILABLE or not PDFPLUMBER_AVAILABLE:
        st.divider()
        if not CAMELOT_AVAILABLE:
            st.warning("**Camelot-py** is not installed. Install it for table extraction:")
            st.code("pip install camelot-py[base]", language="bash")
        if not PDFPLUMBER_AVAILABLE:
            st.warning("**PDFPlumber** is not installed. Install it for text extraction:")
            st.code("pip install pdfplumber", language="bash")
