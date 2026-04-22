import pytest

from langgraph_agent.coordinator_graph import build_coordinator_graph

@pytest.fixture
def graph():
    """
    Build a fresh v3.0 graph (with planner enabled) for each test.
    """
    return build_coordinator_graph()

def test_planner_is_triggered_for_complex_task(graph):
    thread_id = "test-planner-trigger"

    result = graph.invoke(
        {
            "input": "介绍 LangChain 的作用，并给出官网地址",
            "trace": []
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    nodes = [step["node"] for step in result["trace"]]

    assert "planner" in nodes

def test_planner_generates_multiple_steps(graph):
    thread_id = "test-planner-steps"

    result = graph.invoke(
        {
            "input": "介绍 LangChain 的作用，并给出官网地址",
            "trace": []
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    planner_step = next(
        step for step in result["trace"]
        if step["node"] == "planner"
    )

    plan = planner_step["plan"]

    assert isinstance(plan, list)
    assert len(plan) >= 2

def test_planner_plan_contains_multiple_agents(graph):
    thread_id = "test-planner-agent-types"

    result = graph.invoke(
        {
            "input": "介绍 LangChain 的作用，并给出官网地址",
            "trace": []
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    planner_step = next(
        step for step in result["trace"]
        if step["node"] == "planner"
    )

    agents = {step["agent"] for step in planner_step["plan"]}

    # 至少包含 knowledge 和 search 中的一个，最好两个都有
    assert "knowledge" in agents or "search" in agents


def test_plan_execution_order(graph):
    thread_id = "test-plan-order"

    result = graph.invoke(
        {
            "input": "介绍 LangChain 的作用，并给出官网地址",
            "trace": []
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    planner_step = next(
        step for step in result["trace"]
        if step["node"] == "planner"
    )

    results = planner_step["results"]

    assert isinstance(results, list)
    assert len(results) == len(planner_step["plan"])

    # step 编号应按顺序增长
    step_ids = [r["step"] for r in results]
    assert step_ids == sorted(step_ids)


def test_planner_results_merged_into_output(graph):
    thread_id = "test-planner-output"

    result = graph.invoke(
        {
            "input": "介绍 LangChain 的作用，并给出官网地址",
            "trace": []
        },
        config={"configurable": {"thread_id": thread_id}}
    )

    assert isinstance(result["output"], str)
    assert len(result["output"]) > 0
