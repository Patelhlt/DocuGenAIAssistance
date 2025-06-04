# app/image_ocr.py

from PIL import Image
import pytesseract

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image file using Tesseract OCR.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"[OCR ERROR] Failed to extract from {image_path}: {e}")
        return ""
