from langgraph_agent.conditional_graph import build_graph
from langgraph_agent.trace_replay import replay_trace_text

graph = build_graph()
thread_id = "demo-session"

# 第 1 轮：确认主体（写入 memory）
graph.invoke(
    {"input": "我指的是 LangChain", "trace": []},
    config={"configurable": {"thread_id": thread_id}},
)

# 第 2 轮：触发 RAG
result = graph.invoke(
    {"input": "能详细介绍一下", "trace": []},
    config={"configurable": {"thread_id": thread_id}},
)

print("===== 最终输出 =====")
print(result["output"])

print("\n===== 执行回放 =====")
print(replay_trace_text(result["trace"]))