from docx import Document
import os

def parse_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"

    metadata = {
        "source": file_path,
        "ext": os.path.splitext(file_path)[1].lower(),
        "type": "docx"
    }

    return text, metadata