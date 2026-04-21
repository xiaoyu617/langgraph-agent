import pytest

from langgraph_agent.coordinator_graph import build_coordinator_graph


@pytest.fixture
def graph():
    """
    Build a fresh multi-agent graph for each test.
    """
    return build_coordinator_graph()

def test_knowledge_agent_selected_for_concept_question(graph):
    thread_id = "test-knowledge-agent"

    # 第一步：写入 memory（确认主体）
    graph.invoke(
        {"input": "我指的是 LangChain", "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    # 第二步：概念型问题 → 应走 Knowledge Agent
    result = graph.invoke(
        {"input": "能详细介绍一下", "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    # Trace 中必须记录 coordinator_dispatch
    dispatch_step = next(
        step for step in result["trace"]
        if step["node"] == "coordinator_dispatch"
    )

    assert dispatch_step["agent"] == "knowledge_agent"
    assert "LangChain" in result["output"]


def test_search_agent_selected_for_lookup_question(graph):
    thread_id = "test-search-agent"

    result = graph.invoke(
        {"input": "搜索 LangChain 官网", "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    dispatch_step = next(
        step for step in result["trace"]
        if step["node"] == "coordinator_dispatch"
    )

    assert dispatch_step["agent"] == "search_agent"


def test_clarify_path_not_routed_to_agents(graph):
    thread_id = "test-clarify"

    result = graph.invoke(
        {"input": "它是干什么的？", "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    # Trace 中应该有 clarify
    nodes = [step["node"] for step in result["trace"]]

    assert "clarify" in nodes
    assert "coordinator_dispatch" not in nodes


def test_memory_affects_agent_selection(graph):
    thread_id = "test-memory-effect"

    # 第一轮：确认主体
    graph.invoke(
        {"input": "我指的是 LangChain", "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    # 第二轮：代词问题，应直接走 Knowledge Agent，而非 Clarify
    result = graph.invoke(
        {"input": "它能用来做什么？", "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    dispatch_step = next(
        step for step in result["trace"]
        if step["node"] == "coordinator_dispatch"
    )

    assert dispatch_step["agent"] == "knowledge_agent"
