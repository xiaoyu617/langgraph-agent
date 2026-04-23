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

from langgraph_agent.agents.critic_agent import run_critic_agent

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

    # ===== Attempt 1 =====
    plan = run_planner_agent(input_text)
    results = execute_plan(plan, last_subject)
    attempt_output = "\n".join(r["output"] for r in results)

    critic_1 = run_critic_agent(
        user_input=input_text,
        plan=plan,
        execution_results=results,
        final_output=attempt_output,
    )

    trace.append({
        "node": "planner",
        "attempt": 1,
        "plan": plan,
        "results": results,
    })
    trace.append({
        "node": "critic",
        "attempt": 1,
        "verdict": critic_1["verdict"],
        "reason": critic_1["reason"],
    })

    # ✅ 如果第一次就通过，直接返回
    if critic_1["verdict"] == "pass":
        return {
            "input": input_text,
            "last_subject": last_subject,
            "output": attempt_output,
            "trace": trace,
        }

    # ===== Attempt 2 (Replan) =====
    replan = run_planner_agent(
        input_text,
        failure_reason=critic_1["reason"]
    )
    replan_results = execute_plan(replan, last_subject)
    replan_output = "\n".join(
        r["output"] for r in replan_results
    )

    critic_2 = run_critic_agent(
        user_input=input_text,
        plan=replan,
        execution_results=replan_results,
        final_output=replan_output,
    )

    trace.append({
        "node": "planner",
        "attempt": 2,
        "replan": True,
        "plan": replan,
        "results": replan_results,
    })
    trace.append({
        "node": "critic",
        "attempt": 2,
        "verdict": critic_2["verdict"],
        "reason": critic_2["reason"],
    })

    # ✅ Replan 成功 → 覆盖第一次失败结果
    if critic_2["verdict"] == "pass":
        final_output = replan_output
    else:
        final_output = f"⚠️ 多次尝试仍未通过审查：{critic_2['reason']}"

    return {
        "input": input_text,
        "last_subject": last_subject,
        "output": final_output,
        "trace": trace,
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

