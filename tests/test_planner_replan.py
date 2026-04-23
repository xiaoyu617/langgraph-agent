import pytest
from unittest.mock import patch

from langgraph_agent.coordinator_graph import build_coordinator_graph

@pytest.fixture
def graph():
    """
    Build v3.2 planner-enabled graph.
    """
    return build_coordinator_graph()

def test_replan_is_triggered_after_critic_fail(graph):
    thread_id = "test-replan-flow"

    # 模拟 Critic：第一次 fail，第二次 pass
    critic_side_effects = [
        {"verdict": "fail", "reason": "缺少官网信息"},
        {"verdict": "pass", "reason": "已覆盖所有意图"},
    ]

    with patch(
            "langgraph_agent.coordinator_graph.run_critic_agent",
            side_effect=critic_side_effects
    ):
        result = graph.invoke(
            {
                "input": "介绍 LangChain 的作用，并给出官网地址",
                "trace": []
            },
            config={"configurable": {"thread_id": thread_id}}
        )

    # 从 trace 中提取 planner / critic 记录
    planner_steps = [s for s in result["trace"] if s["node"] == "planner"]
    critic_steps = [s for s in result["trace"] if s["node"] == "critic"]

    # ✅ Planner 被调用两次
    assert len(planner_steps) == 2
    assert planner_steps[0]["attempt"] == 1
    assert planner_steps[1]["attempt"] == 2
    assert planner_steps[1].get("replan") is True

    # ✅ Critic 也被调用两次
    assert len(critic_steps) == 2
    assert critic_steps[0]["verdict"] == "fail"
    assert critic_steps[1]["verdict"] == "pass"


def test_replan_result_used_as_final_output(graph):
    thread_id = "test-replan-output"

    critic_side_effects = [
        {"verdict": "fail", "reason": "回答不完整"},
        {"verdict": "pass", "reason": "结果完整"},
    ]

    with patch(
        "langgraph_agent.agents.critic_agent.run_critic_agent",
        side_effect=critic_side_effects
    ):
        result = graph.invoke(
            {
                "input": "介绍 LangChain 的作用，并给出官网地址",
                "trace": []
            },
            config={"configurable": {"thread_id": thread_id}}
        )

    # 系统必须给出最终输出
    assert isinstance(result["output"], str)
    assert len(result["output"]) > 0

    # 不应是失败提示
    assert not result["output"].startswith("⚠️")


def test_replan_failure_graceful_exit(graph):
    thread_id = "test-replan-fail-twice"

    # 两次都 fail
    critic_side_effects = [
        {"verdict": "fail", "reason": "严重缺失信息"},
        {"verdict": "fail", "reason": "仍然缺失"},
    ]

    with patch(
            "langgraph_agent.coordinator_graph.run_critic_agent",
            side_effect=critic_side_effects
    ):
        result = graph.invoke(
            {
                "input": "介绍 LangChain 的作用，并给出官网地址",
                "trace": []
            },
            config={"configurable": {"thread_id": thread_id}}
        )

    # ✅ 系统没有 crash
    assert isinstance(result["output"], str)

    # ✅ 明确告诉失败原因
    assert "未通过审查" in result["output"]