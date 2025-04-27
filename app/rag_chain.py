import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA


def get_rag_chain():
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    db = FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        model="gpt-3.5-turbo",
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
