from langgraph_agent.coordinator_graph import build_coordinator_graph
from langgraph_agent.trace_replay import replay_trace_text

graph = build_coordinator_graph()
thread_id = "multi-agent-demo"

graph.invoke(
    {"input": "我指的是 LangChain", "trace": []},
    config={"configurable": {"thread_id": thread_id}}
)

result = graph.invoke(
    {"input": "能详细介绍一下", "trace": []},
    config={"configurable": {"thread_id": thread_id}}
)

print("===== OUTPUT =====")
print(result["output"])

print("\n===== TRACE =====")
print(replay_trace_text(result["trace"]))