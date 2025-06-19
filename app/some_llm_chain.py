from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ Load Qwen model from Hugging Face
model_name = "Qwen/Qwen1.5-0.5B-Chat"  # You can also try "Qwen/Qwen1.5-14B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True  # Important for Qwen
)

# Wrap into HF pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=512,
    top_p=0.9,
    repetition_penalty=1.2
)

# LangChain-compatible wrapper
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt to generate answer
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following documents to answer the question.

Documents:
{context}

Question: {input}
""")

# Prompt to make query history-aware (future-proofing)
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Combine documents and answer
combine_docs_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)

# Final retrieval → generation chain
def run_chain(query, retriever):
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONDENSE_QUESTION_PROMPT,
    )

    chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain
    )

    return chain.invoke({"input": query, "chat_history": []})
