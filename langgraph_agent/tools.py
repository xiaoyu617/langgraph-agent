"""工具封装层 — 支持 Mock/Real 切换。

通过环境变量 TOOL_MODE 控制：
  TOOL_MODE=mock  （默认，测试和开发用）
  TOOL_MODE=real  （调用真实搜索引擎）

真实搜索引擎通过环境变量配置：
  SEARCH_PROVIDER=duckduckgo  （默认，无需 API Key）
  SEARCH_PROVIDER=baidu       （需配置 BAIDU_API_KEY）
  SEARCH_PROVIDER=bing        （需配置 BING_API_KEY）
"""
from langchain_core.tools import tool
import os
import re


# ========================================
# 知识库（Mock 模式用）
# ========================================
_KNOWLEDGE_BASE: dict[str, str] = {
    "langchain": (
        "LangChain is an open-source framework designed to simplify "
        "the development of applications powered by large language models (LLMs). "
        "It supports chains, agents, tools, memory, and LangGraph integration. "
        "Official site: https://www.langchain.com"
    ),
    "python": (
        "Python is a high-level, general-purpose programming language. "
        "It emphasizes code readability with its notable use of significant indentation. "
        "Created by Guido van Rossum and first released in 1991. "
        "Official site: https://www.python.org"
    ),
    "langgraph": (
        "LangGraph is a library for building stateful, multi-actor applications "
        "with LLMs, built by the LangChain team. It extends LangChain to model "
        "agent workflows as graphs using StateGraph, conditional edges, and checkpointing. "
        "Official site: https://langchain-ai.github.io/langgraph/"
    ),
    "rag": (
        "RAG (Retrieval-Augmented Generation) is a technique that combines "
        "information retrieval with text generation. It retrieves relevant documents "
        "from a knowledge base and uses them as context for LLM generation, "
        "reducing hallucinations and improving factual accuracy."
    ),
    "agent": (
        "An AI Agent is an autonomous system that uses LLMs to reason, "
        "make decisions, and take actions using tools. Key capabilities include: "
        "task planning, tool use (Function Calling), memory management, "
        "and self-reflection (ReAct loop)."
    ),
    "function calling": (
        "Function Calling is a capability of LLMs to output structured JSON "
        "requests for function execution. The LLM decides which function to call "
        "and what parameters to pass, but does NOT execute the function itself. "
        "Execution is handled by the application code."
    ),
    "lintcode": (
        "LintCode is an online platform for coding interview preparation, "
        "offering a large collection of algorithm and data structure problems. "
        "It supports multiple programming languages including Python, Java, C++."
    ),
    "leetcode": (
        "LeetCode is a popular online platform for coding interview preparation. "
        "It offers thousands of algorithm and database problems, "
        "weekly contests, and interview simulation features. "
        "Official site: https://leetcode.com"
    ),
}


# ========================================
# 模式判断
# ========================================

def _is_mock_mode() -> bool:
    """检查当前是否为 Mock 模式。"""
    return os.environ.get("TOOL_MODE", "mock").lower() == "mock"


# ========================================
# Mock 搜索
# ========================================

def _mock_search(query: str) -> str:
    """Mock 搜索：从知识库中匹配关键词。"""
    query_lower = query.lower()
    matched = []

    for keyword, content in _KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            matched.append(content)

    if matched:
        return "\n\n---\n\n".join(matched)

    return f'未找到与"{query}"相关的搜索结果（Mock 知识库无匹配）。'


# ========================================
# 真实搜索（支持多个搜索引擎）
# ========================================

def _real_search(query: str) -> str:
    """真实搜索：根据配置调用外部搜索引擎。"""
    provider = os.environ.get("SEARCH_PROVIDER", "duckduckgo").lower()

    if provider == "duckduckgo":
        return _search_duckduckgo(query)
    elif provider == "baidu":
        return _search_baidu(query)
    elif provider == "bing":
        return _search_bing(query)
    else:
        return f'不支持的搜索引擎: {provider}，可选: duckduckgo, baidu, bing'


def _search_duckduckgo(query: str) -> str:
    """DuckDuckGo 搜索（无需 API Key，但大陆网络可能受限）。"""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return (
            'DuckDuckGo 搜索不可用：缺少 duckduckgo_search 库。\n'
            '安装方式: pip install duckduckgo_search\n'
            '或设置 SEARCH_PROVIDER=baidu 使用其他搜索引擎'
        )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return (
            f'DuckDuckGo 搜索失败: {type(e).__name__}\n'
            f'提示：DuckDuckGo 在大陆网络可能无法访问。\n'
            f'可尝试设置 SEARCH_PROVIDER=baidu 使用百度搜索。'
        )

    if not results:
        return f'未找到与"{query}"相关的搜索结果。'

    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"- {title}: {body[:150]} ({href})")

    return "\n\n".join(lines)


def _search_baidu(query: str) -> str:
    """百度搜索（大陆网络推荐）。"""
    try:
        import requests
    except ImportError:
        return '百度搜索不可用：缺少 requests 库。\n安装方式: pip install requests'

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    params = {"wd": query, "tn": "SE_baiduhome_pg"}

    try:
        resp = requests.get(
            "https://www.baidu.com/s",
            params=params,
            headers=headers,
            timeout=10,
        )
        resp.encoding = "utf-8"
        html = resp.text
    except Exception as e:
        return f'百度搜索失败: {type(e).__name__}: {str(e)}'

    # 提取搜索结果
    results = []
    # 百度搜索结果在 h3 > a 标签中
    for item in re.findall(
        r'<h3[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )[:5]:
        url, title = item
        title = re.sub(r'<[^>]+>', "", title).strip()
        if title:
            results.append(f"- {title} ({url})")

    if not results:
        return (
            f'百度搜索无结果（可能触发了反爬机制）。'
        )

    return "\n\n".join(results)


def _search_bing(query: str) -> str:
    """Bing 搜索（需配置 BING_API_KEY）。"""
    api_key = os.environ.get("BING_API_KEY", "")
    if not api_key:
        return (
            'Bing 搜索需要配置 API Key。\n'
            '设置方式: export BING_API_KEY=your_key_here'
        )

    try:
        import requests
    except ImportError:
        return 'Bing 搜索不可用：缺少 requests 库。'

    try:
        resp = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": query, "count": 3},
            headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f'Bing 搜索失败: {type(e).__name__}: {str(e)}'

    results = []
    for item in data.get("webPages", {}).get("value", [])[:3]:
        results.append(
            f"- {item.get('name', '')}: {item.get('snippet', '')} ({item.get('url', '')})"
        )

    if not results:
        return f'未找到与"{query}"相关的搜索结果。'

    return "\n\n".join(results)


# ========================================
# 公开 Tool 接口
# ========================================

@tool
def google_search(query: str) -> str:
    """
    搜索互联网获取实时信息。

    当用户需要查询以下内容时使用：
    - 官网地址、产品文档
    - 最新资讯、新闻动态
    - 技术概念、框架介绍
    - 编程问题、解决方案

    支持中文和英文搜索。
    当前模式通过 TOOL_MODE 环境变量控制（mock/real）。
    """
    if _is_mock_mode():
        return _mock_search(query)
    return _real_search(query)


# ========================================
# 工具函数
# ========================================

def add_knowledge(keyword: str, content: str) -> None:
    """动态添加知识库条目（用于测试或运行时扩展）。"""
    _KNOWLEDGE_BASE[keyword] = content


def list_knowledge_topics() -> list[str]:
    """列出知识库中所有主题。"""
    return list(_KNOWLEDGE_BASE.keys())
