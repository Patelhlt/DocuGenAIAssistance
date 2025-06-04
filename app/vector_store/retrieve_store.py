from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_retriever(persist_directory="chroma_db"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever
