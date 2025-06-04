# processing/text_cleaner.py

import re

def clean_text(text: str) -> str:
    """
    Cleans the extracted text for NLP tasks:
    - Removes non-UTF characters
    - Normalizes punctuation spacing
    - Removes unwanted special characters
    - Collapses extra whitespace and newlines
    - Ensures clean, consistent input for language models
    """
    # Remove non-UTF characters
    text = text.encode("utf-8", "ignore").decode()

    # Normalize punctuation spacing (space around .,!?()-)
    text = re.sub(r'([.,!?();:\-])', r' \1 ', text)

    # Remove special characters except basic punctuation
    text = re.sub(r"[^\w\s.,!?();:\-]", "", text)

    # Collapse multiple whitespaces and newlines into one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()
