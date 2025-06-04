from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

text_gen_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0
)

# Wrap in LangChain LLM interface
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Prompt for answering using context
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following documents to answer the question.

Documents:
{context}

Question: {input}
""")

# Prompt to handle chat history (for future extensions)
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Chain to combine retrieved documents and generate answer
combine_docs_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)

def run_chain(query, retriever):
    # Retriever with history-aware rephrasing (currently history is empty)
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONDENSE_QUESTION_PROMPT,
    )

    # Final chain to retrieve + generate
    chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain
    )

    # Currently using empty chat history
    return chain.invoke({"input": query, "chat_history": []})
