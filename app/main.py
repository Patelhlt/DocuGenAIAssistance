import os
import sys
import tkinter as tk
from tkinter import filedialog

from parsers.file_parser import parse_file
from processing.text_cleaner import clean_text
from processing.text_chunker import chunk_text, add_metadata_to_chunks
from vector_store.embed_store import create_embedding_store
from vector_store.retrieve_store import load_retriever

from some_llm_chain import run_chain  # LLM chain using Llama-2 or other models

def main(file_path: str) -> None:
    print(f"\n🚀 Starting processing for: {file_path}")

    # Step 1: Parse the file to extract multiple (text, metadata) pairs
    try:
        parsed_chunks = parse_file(file_path)  # Now returns list of dicts with 'text' and 'metadata'
    except Exception as e:
        print(f"❌ File parsing failed: {e}")
        return

    # ✅ [CHANGED HERE] Preview first 3 parsed chunks before cleaning
    print("\n🔍 Parsed Chunk Preview (before cleaning):")
    for i, chunk in enumerate(parsed_chunks[:3]):
        print(f"\n✅ Preview {i+1}:\n{chunk['text'][:200]}... | Metadata: {chunk['metadata']}")

    all_chunks_with_meta = []

    for chunk in parsed_chunks:
        raw_text = chunk["text"]
        metadata = chunk["metadata"]

        print("\n📄 Raw Text Sample:")
        print(raw_text[:300])  # Show first 300 chars

        # Step 2: Clean the extracted raw text
        cleaned_text = clean_text(raw_text)
        print("\n🧹 Cleaned Text Sample:")
        print(cleaned_text[:300])

        # Step 3: Chunk the cleaned text
        chunks = chunk_text(cleaned_text)
        print(f"\n✂️ Total chunks created: {len(chunks)}")

        # Step 4: Add metadata to each chunk
        chunked_with_meta = add_metadata_to_chunks(chunks, metadata)
        all_chunks_with_meta.extend(chunked_with_meta)

    # Step 5: Create embeddings and persist in Chroma DB (with better error tracking)
    try:
        print(f"\n🧠 Creating embeddings for {len(all_chunks_with_meta)} chunks...")
        vectordb = create_embedding_store(all_chunks_with_meta)
        print(f"\n✅ Embeddings stored in Chroma DB for {len(all_chunks_with_meta)} chunks.")
    except Exception as e:
        print(f"\n❌ Embedding creation/storage failed: {e}")
        print("🔍 Attempting to isolate the problematic chunk...")

        for idx, chunk in enumerate(all_chunks_with_meta):
            try:
                _ = chunk["text"]  # simulate use
            except Exception as chunk_err:
                print(f"🔴 Chunk #{idx} failed: {chunk_err}")
                print(f"🔎 Problematic Text: {chunk['text'][:150]}...")
                print(f"📄 Metadata: {chunk['metadata']}")
        return

    # Step 6: Load retriever
    retriever = load_retriever()

    # ✅ Step 7: Continuous elastic-like querying until user exits manually
    print("\n📝 You can now enter your questions. Type 'exit' to quit.")
    while True:
        query = input("\n🔎 Ask your query: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting the chatbot. Have a great day!")
            break
        if not query:
            print("⚠️ Please enter a valid question.")
            continue
        try:
            response = run_chain(query, retriever)
            print(f"\n🤖 Answer: {response['answer']}")
        except Exception as e:
            print(f"❌ Failed to get response: {e}")


def select_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a file")
    if file_path:
        return file_path

    folder_path = filedialog.askdirectory(title="Select a folder")
    if folder_path:
        return folder_path

    return None


if __name__ == "__main__":
    path = select_path()
    if not path:
        print("❌ No file or folder selected, exiting.")
        sys.exit(1)

    if os.path.isfile(path):
        if path.lower().endswith((".pdf", ".docx", ".pptx", ".ppt", ".txt", ".png", ".jpg", ".jpeg")):
            main(path)
        else:
            print(f"⛔ Skipping unsupported file: {os.path.basename(path)}")
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                if filename.lower().endswith((".pdf", ".docx", ".pptx", ".ppt", ".txt", ".png", ".jpg", ".jpeg")):
                    print(f"\n📂 Processing file: {filename}")
                    main(file_path)
                else:
                    print(f"⛔ Skipping unsupported file: {filename}")
            else:
                print(f"Skipping {filename}, not a file.")
    else:
        print("❌ Invalid selection. Exiting.")
        sys.exit(1)
