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
        parsed_chunks = parse_pdf_with_page_metadata(file_path)  # returns list of (text, metadata)
        return [
            {"text": text, "metadata": metadata}
            for text, metadata in parsed_chunks if text.strip()
        ]

    elif ext == ".docx":
        text, metadata = parse_docx(file_path)
        return [{"text": text.strip(), "metadata": metadata}]

    elif ext == ".doc":
        try:
            text = textract.process(file_path).decode('utf-8')
            metadata = {
                "source": file_path,
                "ext": ext,
                "type": "doc"
            }
            return [{"text": text.strip(), "metadata": metadata}]
        except Exception as e:
            raise ValueError(f"[DOC ERROR] Failed to parse .doc: {e}")

    elif ext in [".ppt", ".pptx"]:
        slides = parse_ppt(file_path)  # returns list of (text, metadata)
        return [
            {"text": text.strip(), "metadata": metadata}
            for text, metadata in slides if text.strip()
        ]

    elif ext == ".txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        metadata = {
            "source": file_path,
            "ext": ext,
            "type": "txt"
        }
        return [{"text": text.strip(), "metadata": metadata}]

    elif ext in [".png", ".jpg", ".jpeg"]:
        text = extract_text_from_image(file_path)
        metadata = {
            "source": file_path,
            "ext": ext,
            "type": "image"
        }
        return [{"text": text.strip(), "metadata": metadata}]

    else:
        raise ValueError(f"Unsupported file type: {ext}")
