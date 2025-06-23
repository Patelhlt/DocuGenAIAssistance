# utils/pdf_utils.py

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

from processing.text_chunker import chunk_text, add_metadata_to_chunks

def parse_pdf_with_page_metadata(file_path):
    doc = fitz.open(file_path)
    page_chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if not text:
            # OCR fallback: extract image and use pytesseract
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)

        # ✅ Chunk the page text (you already configured word-based chunking)
        chunks = chunk_text(text)

        # ✅ Add metadata for each chunk and return (text, metadata) pair
        metadata = {
            "source": file_path,
            "ext": ".pdf",
            "type": "pdf",
            "page": page_num
        }

        for chunk in chunks:
            page_chunks.append((chunk, metadata))

    return page_chunks