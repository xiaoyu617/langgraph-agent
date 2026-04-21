from langchain_community.chat_models import ChatTongyi
from langgraph_agent.rag_utils import build_vectorstore


def run_knowledge_agent(input_text: str, last_subject: str | None) -> str:
    vectorstore = build_vectorstore()

    query = (
        f"{last_subject} {input_text}"
        if last_subject else input_text
    )

    docs = vectorstore.similarity_search(query, k=1)
    context = docs[0].page_content if docs else ""

    llm = ChatTongyi(model="qwen-plus", temperature=0)
    resp = llm.invoke(
        f"基于以下知识回答问题：\n{context}\n\n问题：{input_text}"
    )

    return resp.content
