from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def build_vectorstore():
    """
    Build a local vector store for RAG.
    """
    docs = [
        Document(
            page_content=(
                "LangChain is an open-source framework designed to simplify "
                "the development of applications powered by large language models. "
                "It supports chains, agents, tools, memory, and LangGraph integration."
            )
        )
    ]
    embeddings = DashScopeEmbeddings()
    return FAISS.from_documents(docs, embeddings)