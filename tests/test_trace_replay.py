from langgraph_agent.trace_replay import replay_trace_json


def test_replay_json_structure():
    trace = [
        {"node": "router", "decision": {"rag": True}},
        {"node": "rag_retrieve", "rag_query": "LangChain intro"},
    ]

    replay = replay_trace_json(trace, "demo")

    assert replay["conversation_id"] == "demo"
    assert replay["turns"][0]["node"] == "router"