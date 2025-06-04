from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
# OLD (deprecated)
from langchain.embeddings import HuggingFaceEmbeddings

# NEW
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_embedding_store(chunks_with_metadata):
    # Toggle between embedding providers
    USE_OPENAI = False  # Set True only if you have quota

    if USE_OPENAI:
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Optional: Add debug flag to control verbosity
    DEBUG = True

    if DEBUG:
        print(f"\n🧪 Debug: Creating embeddings for {len(chunks_with_metadata)} chunks...")

    texts = []
    metadatas = []

    for i, chunk in enumerate(chunks_with_metadata):
        try:
            text = chunk["text"]
            metadata = chunk["metadata"]

            # ✅ Ensure metadata is a dict (required by LangChain/Chroma)
            if not isinstance(metadata, dict):
                metadata = {"source": metadata}

            texts.append(text)
            metadatas.append(metadata)

            if DEBUG:
                print(f"✅ Chunk {i+1}: {text[:100]}... | Metadata: {metadata}")
        except Exception as e:
            print(f"❌ Failed to process chunk {i}: {e}")
            continue

    vectordb = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    vectordb.persist()
    return vectordb
