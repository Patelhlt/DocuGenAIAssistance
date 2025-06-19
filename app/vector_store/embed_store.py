from langchain_community.vectorstores import FAISS
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

            # ✅ Ensure metadata is a dict (required by LangChain/FAISS)
            if not isinstance(metadata, dict):
                metadata = {"source": metadata}

            texts.append(text)
            metadatas.append(metadata)

            if DEBUG:
                print(f"✅ Chunk {i+1}: {text[:100]}... | Metadata: {metadata}")
        except Exception as e:
            print(f"❌ Failed to process chunk {i}: {e}")
            continue

    # 🔍 Preview the chunks being embedded
    print("\n🔍 Previewing first 5 chunks before embedding:")
    for i, text in enumerate(texts[:5]):
        print(f"\n🔎 Chunk {i+1}:\n{text[:300]}\n{'-'*50}")

    vectordb = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vectordb.save_local("faiss_index")

    return vectordb
