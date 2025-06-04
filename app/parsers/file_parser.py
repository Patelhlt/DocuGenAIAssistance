import os
from utils.pdf_utils import parse_pdf_with_page_metadata
from utils.docx_utils import parse_docx
from utils.ppt_utils import parse_ppt
import pytesseract
import textract
from PIL import Image

# Optional: Set this if Tesseract is not in your PATH
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise ValueError(f"[OCR ERROR] Failed for {file_path}: {e}")

def parse_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        # Call your PDF parser with metadata and chunks
        parsed_chunks = parse_pdf_with_page_metadata(file_path)
        return parsed_chunks  # Return list of (text, metadata)

    elif ext == ".docx":
        text, metadata = parse_docx(file_path)
        return [(text, metadata)]  # Wrap in list for consistency

    elif ext == ".doc":
        try:
            text = textract.process(file_path).decode('utf-8')
            metadata = {
                "source": file_path,
                "ext": ext,
                "type": "doc"
            }
            return [(text, metadata)]
        except Exception as e:
            raise ValueError(f"[DOC ERROR] Failed to parse .doc: {e}")

    elif ext in [".ppt", ".pptx"]:
        slides = parse_ppt(file_path)  # Already returns list of (text, metadata)
        return slides

    elif ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        metadata = {
            "source": file_path,
            "ext": ext,
            "type": "txt"
        }
        return [(text, metadata)]

    elif ext in [".png", ".jpg", ".jpeg"]:
        text = extract_text_from_image(file_path)
        metadata = {
            "source": file_path,
            "ext": ext,
            "type": "image"
        }
        return [(text, metadata)]

    else:
        raise ValueError(f"Unsupported file type: {ext}")
