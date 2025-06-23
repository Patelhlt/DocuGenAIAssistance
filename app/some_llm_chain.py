from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ Load Qwen model
model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # You can try a larger one if needed

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# ✅ Build a pipeline using HuggingFace
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=512,
    repetition_penalty=1.2
)

# ✅ Wrap the pipeline for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ✅ Document-answering prompt (plain text, not chat format)
ANSWER_PROMPT = PromptTemplate.from_template("""
Use the following document snippets to answer the question as accurately and concisely as possible.

{context}

Question: {input}

Answer:
""")

# ✅ Rewriting user query with history (currently not used, but future-proofed)
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the chat history and a new question, rewrite the question to be standalone.

Chat History:
{chat_history}

Follow-up question: {input}

Standalone question:
""")

# ✅ Build answer generation chain from documents
combine_docs_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)

# ✅ Full function to run query
def run_chain(query, retriever, filter_by=None):
    if filter_by:
        # Wrap the retriever to filter chunks
        original_get_relevant_documents = retriever.get_relevant_documents

        def filtered_get_relevant_documents(input_str):
            docs = original_get_relevant_documents(input_str)
            if filter_by == "first_page":
                docs = [doc for doc in docs if doc.metadata.get("page", 1) == 1]
            elif filter_by == "abstract":
                docs = [doc for doc in docs if "abstract" in doc.page_content.lower()]
            return docs

        retriever.get_relevant_documents = filtered_get_relevant_documents

    # History-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONDENSE_QUESTION_PROMPT,
    )

    # Final chain
    chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain
    )

    return chain.invoke({"input": query, "chat_history": []})