# processing/text_chunker.py

import re

def chunk_text(text: str, chunk_size=50, chunk_overlap=20):
    """
    Splits text into chunks based on words with overlap for better semantic retention.
    - chunk_size: number of words per chunk
    - chunk_overlap: number of overlapping words between chunks
    """
    # Tokenize the text by words, preserving punctuation
    words = re.findall(r'\w+|\W+', text)  # e.g., ['This', ' ', 'is', ' ', 'text', '.', ' ']

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = ''.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks

def add_metadata_to_chunks(chunks, metadata: dict):
    return [{"text": chunk, "metadata": metadata} for chunk in chunks]