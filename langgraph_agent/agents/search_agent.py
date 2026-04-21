from langchain_community.chat_models import ChatTongyi
from langgraph_agent.tools import google_search


def run_search_agent(input_text: str) -> str:
    search_result = google_search.invoke(input_text)

    llm = ChatTongyi(model="qwen-plus", temperature=0)
    resp = llm.invoke(
        f"根据搜索结果回答问题：\n{search_result}"
    )

    return resp.content
