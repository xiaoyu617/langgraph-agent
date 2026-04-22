from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent.conditional_graph import (
    ConversationState,
    DerivedState,
    WorkingState,
    GraphState,
    router_node,
    memory_update_node,
    clarify_node,
)

from langgraph_agent.agents.knowledge_agent import run_knowledge_agent
from langgraph_agent.agents.search_agent import run_search_agent


from langgraph_agent.agents.planner_agent import run_planner_agent
from langgraph_agent.planner_executor import execute_plan


def coordinator_dispatch_node(state: GraphState) -> GraphState:
    trace = state["trace"]
    input_text = state["input"]
    last_subject = state.get("last_subject")

    if state["need_rag"]:
        output = run_knowledge_agent(input_text, last_subject)
        agent = "knowledge_agent"

    elif state["need_search"]:
        output = run_search_agent(input_text)
        agent = "search_agent"

    else:
        output = "No suitable agent found."
        agent = "none"

    trace.append({
        "node": "coordinator_dispatch",
        "agent": agent,
        "output": output
    })

    return {
        "input": input_text,
        "last_subject": last_subject,
        "output": output,
        "trace": trace
    }


def planner_node(state: GraphState) -> GraphState:
    trace = state["trace"]
    input_text = state["input"]
    last_subject = state.get("last_subject")

    plan = run_planner_agent(input_text)
    results = execute_plan(plan, last_subject)

    trace.append({
        "node": "planner",
        "plan": plan,
        "results": results
    })

    final_output = "\n".join(
        r["output"] for r in results
    )

    return {
        "input": input_text,
        "last_subject": last_subject,
        "output": final_output,
        "trace": trace
    }


def build_coordinator_graph():
    graph = StateGraph(GraphState)
    checkpointer = MemorySaver()

    graph.add_node("router", router_node)
    graph.add_node("memory_update", memory_update_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("coordinator_dispatch", coordinator_dispatch_node)
    graph.add_node("planner", planner_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda s:
        "memory_update" if s["is_subject_confirm"]
        else "clarify" if s["need_clarify"]
        else "planner" if s.get("need_plan")
        else "coordinator_dispatch"
    )

    graph.add_edge("memory_update", END)
    graph.add_edge("clarify", END)
    graph.add_edge("coordinator_dispatch", END)

    return graph.compile(checkpointer=checkpointer)

