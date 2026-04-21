from langgraph_agent.conditional_graph import build_graph


def test_rag_path_with_memory():
    graph = build_graph()
    thread_id = "test-session"

    graph.invoke(
        {"input": "我指的是 LangChain", "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    result = graph.invoke(
        {"input": "能详细介绍一下", "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    nodes = [s["node"] for s in result["trace"]]
    assert "rag_retrieve" in nodes
    assert "LangChain" in next(
        s for s in result["trace"] if s["node"] == "rag_retrieve"
    )["rag_query"]