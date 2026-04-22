from langgraph_agent.coordinator_graph import build_coordinator_graph
from langgraph_agent.trace_replay import replay_trace_text

def main():
    graph = build_coordinator_graph()
    thread_id = "planner-demo-session"

    print("=== User Input ===")
    user_input = "介绍 LangChain 的作用，并给出官网地址"
    print(user_input)

    print("\n=== Running v3.0 Planner Agent ===")
    result = graph.invoke(
        {"input": user_input, "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    print("\n=== Final Output ===")
    print(result["output"])

    print("\n=== Execution Trace (Replay) ===")
    print(replay_trace_text(result["trace"]))


if __name__ == "__main__":
    main()