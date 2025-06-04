# pdf_utils.py

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

from processing.text_chunker import chunk_text, add_metadata_to_chunks  # better to import at top

def parse_pdf_with_page_metadata(file_path):
    doc = fitz.open(file_path)
    page_chunks_with_meta = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if not text:
            # OCR fallback: get PNG bytes and open as Image
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)

        # Chunk text of this page
        chunks = chunk_text(text)

        # Add page number metadata to each chunk
        metadata = {
            "source": file_path,
            "ext": ".pdf",
            "type": "pdf",
            "page": page_num
        }
        chunks_with_meta = add_metadata_to_chunks(chunks, metadata)
        page_chunks_with_meta.extend(chunks_with_meta)

    return page_chunks_with_meta