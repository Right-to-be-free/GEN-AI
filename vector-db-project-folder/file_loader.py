import os
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from PIL import Image
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# -------- .txt loader --------
def load_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            return file.read()
    except Exception as e:
        print(f"❌ Error reading TXT: {file_path}: {e}")
        return ""

# -------- .pdf loader --------
def load_pdf(file_path):
    try:
        normalized_path = os.path.abspath(file_path)
        doc = fitz.open(normalized_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"❌ Error reading PDF with PyMuPDF: {file_path}: {e}")
        return ""

# -------- .csv loader --------
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        rows = df.astype(str).values.tolist()
        return "\n".join([", ".join(row) for row in rows])
    except Exception as e:
        print(f"❌ Error reading CSV: {file_path}: {e}")
        return ""
def load_image(file_path):
    """
    Extract text from image using Tesseract OCR.
    Supported: .jpg, .jpeg, .png
    """
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        return text
    except Exception as e:
        print(f"❌ Error reading image {file_path}: {e}")
        return ""

# -------- main dispatcher --------
def load_file(file_path):
    """
    Determine file type and extract content accordingly.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.txt':
        return load_txt(file_path)
    elif ext == '.pdf':
        return load_pdf(file_path)
    elif ext == '.csv':
        return load_csv(file_path)
    elif ext == '.docx':
        return load_docx(file_path)
    elif ext == '.xlsx':
        return load_xlsx(file_path)
    elif ext in ['.jpg', '.jpeg', '.png']:
        return load_image(file_path)
    else:
        print(f"⚠️ Unsupported file type: {ext}")
        return ""
from docx import Document

def load_docx(file_path):
    """
    Extract text from a Word (.docx) file.
    """
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Error reading DOCX: {file_path}: {e}")
        return ""
def load_xlsx(file_path):
    """
    Extract text from an Excel (.xlsx) file by flattening all sheets.
    """
    try:
        excel = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        all_text = []

        for sheet_name, df in excel.items():
            all_text.append(f"--- Sheet: {sheet_name} ---")
            rows = df.astype(str).values.tolist()
            all_text.extend([", ".join(row) for row in rows])

        return "\n".join(all_text)
    except Exception as e:
        print(f"❌ Error reading XLSX: {file_path}: {e}")
        return ""
