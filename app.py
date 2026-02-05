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
    try:
        return easyocr.Reader(['en'], gpu=False)
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
        
        # Convert PIL image to numpy array
        try:
            img_array = np.array(image)
        except Exception as e:
            logger.error(f"Image conversion error: {str(e)}")
            st.error(f"Failed to convert image: {str(e)}")
            return None
        
        # Run OCR
        try:
            result = ocr.readtext(img_array)
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            st.error(f"OCR failed: {str(e)}")
            return None
        
        if not result:
            return None
        
        # Sort OCR results by Y position (top to bottom)
        sorted_results = sorted(result, key=lambda x: min([p[1] for p in x[0]]))
        
        # Extract all text lines
        all_lines = []
        for detection in sorted_results:
            text = detection[1].strip()
            if text:
                all_lines.append(text)
        
        # DEBUG: Show what OCR extracted (first 30 lines)
        with st.expander("Debug: OCR Extracted Lines", expanded=False):
            for i, line in enumerate(all_lines[:30]):
                st.text(f"{i}: {line}")
    
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        st.error(f"Error during extraction: {str(e)}")
        return None
    
    # 1. Extract Heading (look for "Utilisation Certificate")
    for i, line in enumerate(all_lines[:5]):  # Check first 5 lines
        if "Utilisation" in line or "Certificate" in line:
            structured_data["heading"] = line
            break
    
    # 2. Extract two-column data (lines with colons before table sections)
    # Look for specific patterns
    table_section_reached = False
    i = 0
    
    while i < len(all_lines):
        line = all_lines[i]
        
        # Stop when we reach numbered sentences or table sections
        if re.match(r'^\d+\)', line):
            table_section_reached = True
            i += 1
            continue
        
        if "Receipt Details" in line or "Expenditure Details" in line:
            table_section_reached = True
            i += 1
            continue
        
        # Only process lines before table sections
        if table_section_reached:
            i += 1
            continue
        
        # Look for label: value pattern
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                label = parts[0].strip()
                value = parts[1].strip()
                
                # Filter out noise
                skip_labels = ["Sign of DDO", "Name of DDO", "Receipt Details", "Expenditure Details"]
                if label not in skip_labels and len(label) > 2:
                    # Check if value is empty and next line might be the value
                    if not value and i + 1 < len(all_lines):
                        next_line = all_lines[i + 1].strip()
                        # If next line doesn't have colon and isn't a number, it might be the value
                        if ':' not in next_line and not re.match(r'^\d+\)', next_line):
                            value = next_line
                            i += 1  # Skip the next line since we used it
                    
                    structured_data["two_column_data"].append({
                        "label": label,
                        "value": value
                    })
        
        # Also look for common UC document fields even without colon
        # Pattern: "Label" followed by value on same or next line
        else:
            # Check for known labels
            known_labels = [
                "Deposit Work ID", "Name of Work", "User Department", 
                "Administrative Approval No", "Administrative Approval Date",
                "AA Amount in Rupees", "Sanctioned Amount"
            ]
            
            for known_label in known_labels:
                if known_label.lower() in line.lower():
                    # Extract value from same line or next line
                    value = line.replace(known_label, '').strip()
                    value = value.lstrip(':').strip()
                    
                    if not value and i + 1 < len(all_lines):
                        next_line = all_lines[i + 1].strip()
                        if ':' not in next_line and not re.match(r'^\d+\)', next_line):
                            value = next_line
                            i += 1  # Skip next line
                    
                    if value:
                        structured_data["two_column_data"].append({
                            "label": known_label,
                            "value": value
                        })
                    break
        
        i += 1
    
    # 3. Extract sentences (numbered points like "1)", "2)", etc.)
    for line in all_lines:
        # Match lines starting with number followed by )
        if re.match(r'^\d+\)', line):
            structured_data["sentences"].append(line)
    
    # 4 & 5. Extract tables using Camelot (LATTICE mode only)
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
        
        for table in tables:
            df = table.df
            
            if df.empty:
                continue
            
            # Clean the dataframe
            df = df.replace('', pd.NA)
            df = df.dropna(how='all')  # Remove empty rows
            df = df.dropna(axis=1, how='all')  # Remove empty columns
            df = df.reset_index(drop=True)
            
            if df.empty:
                continue
            
            # Detect which table this is by checking the FIRST ROW (header row)
            # Convert first row to string
            if len(df) > 0:
                first_row_text = ' '.join(df.iloc[0].astype(str).values).lower()
                
                # Check for Receipt table keywords
                if 'date' in first_row_text and 'received' in first_row_text:
                    structured_data["receipt_details"] = df.to_dict('records')
                # Check for Expenditure table keywords
                elif 'financial' in first_row_text and 'year' in first_row_text:
                    structured_data["expenditure_details"] = df.to_dict('records')
                elif 'bill' in first_row_text and 'amount' in first_row_text:
                    structured_data["expenditure_details"] = df.to_dict('records')
                else:
                    # Fallback: check second row or table position
                    # If we already have receipt details, this must be expenditure
                    if structured_data["receipt_details"]:
                        structured_data["expenditure_details"] = df.to_dict('records')
                    else:
                        # First table is usually receipt
                        structured_data["receipt_details"] = df.to_dict('records')
                        
    except Exception as e:
        st.warning(f"Table extraction warning: {str(e)}")
    
    return structured_data


def convert_uc_to_excel_format(structured_data):
    """
    Convert UC structured data to Excel format matching PDF layout.
    Returns list of dataframes for sequential writing.
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
    
    # Block 3: Sentences
    if structured_data.get("sentences"):
        rows = [[s] for s in structured_data["sentences"]]
        df_sentences = pd.DataFrame(rows, columns=['Content'])
        blocks.append({
            'label': 'Statements',
            'df': df_sentences
        })
    
    # Block 4: Receipt Details
    if structured_data.get("receipt_details"):
        df_receipt = pd.DataFrame(structured_data["receipt_details"])
        # Clean up the dataframe
        df_receipt = df_receipt.replace('', pd.NA).dropna(how='all').dropna(axis=1, how='all')
        if not df_receipt.empty:
            blocks.append({
                'label': 'Receipt Details',
                'df': df_receipt
            })
    
    # Block 5: Expenditure Details
    if structured_data.get("expenditure_details"):
        df_expenditure = pd.DataFrame(structured_data["expenditure_details"])
        # Clean up the dataframe
        df_expenditure = df_expenditure.replace('', pd.NA).dropna(how='all').dropna(axis=1, how='all')
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
    
    st.success(f"Extraction completed: {len(page_data)} page(s) processed")
    
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
