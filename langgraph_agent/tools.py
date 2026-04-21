from langchain.tools import tool


@tool
def google_search(query: str) -> str:
    """
    Mock Google search tool.
    Replace with real search API in production.
    """
    if "LangChain" in query:
        return (
            "LangChain is an open-source framework for building LLM applications. "
            "Official site: https://www.langchain.com"
        )
    return "No relevant search results found."