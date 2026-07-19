"""
多 Agent 协调架构 — Multi-Agent Coordinator (v2.0)

架构:
                    User Input
                        │
                ┌───────┴───────┐
                │  Coordinator  │ ← 调度中心，负责任务拆分与分配
                │    Agent      │
                └───────┬───────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
    ┌─────┴─────┐ ┌────┴────┐ ┌─────┴─────┐
    │   Test    │ │  Test   │ │  Result   │
    │ Generator │ │ Executor│ │  Analysis │
    │  Agent    │ │  Agent  │ │   Agent   │
    └─────┬─────┘ └────┬────┘ └─────┬─────┘
          │             │             │
          └─────────────┼─────────────┘
                        │
                   ┌────┴────┐
                   │ Report  │
                   │  Agent  │
                   └─────────┘

特点:
  - 每个 Agent 职责单一，可独立测试
  - Coordinator 负责任务路由和结果聚合
  - 天然支持人类-in-the-loop（审批节点）
"""
from typing import TypedDict, Optional, List, Any, Callable
from enum import Enum
from langgraph.graph import StateGraph, END
import json
import time


# =========================
# 枚举 & 类型定义
# =========================

class AgentTask(Enum):
    """Agent 任务类型"""
    GENERATE_TEST = "generate_test"
    EXECUTE_TEST = "execute_test"
    ANALYZE_RESULT = "analyze_result"
    GENERATE_REPORT = "generate_report"
    UNKNOWN = "unknown"


# =========================
# 多 Agent 状态
# =========================

class AgentMessage(TypedDict):
    """Agent 间通信消息。"""
    from_agent: str              # 发送方
    to_agent: str                # 接收方 (coordinator / all)
    task_type: str               # 任务类型
    payload: dict                # 消息内容
    status: str                  # pending / success / failed
    timestamp: float             # 时间戳


class MultiAgentState(TypedDict):
    """多 Agent 系统的全局状态。"""
    input: str                   # 原始输入
    requirements: List[str]      # 需求列表
    messages: List[AgentMessage] # 消息总线
    current_task: str            # 当前执行的任务
    task_queue: List[str]        # 待处理任务队列
    context: dict                # 共享上下文

    # 各 Agent 的工作产物
    test_cases: Optional[dict]
    execution_results: Optional[list]
    analysis: Optional[list]
    report: Optional[str]

    error: Optional[str]
    completed: bool


# =========================
# Coordinator — 调度中心
# =========================

def coordinator_node(state: MultiAgentState) -> MultiAgentState:
    """
    Coordinator Agent — 调度中心。

    职责：
      1. 解析用户输入，拆解为子任务
      2. 分派任务到对应的 Agent
      3. 收集子任务结果，判断下一步
      4. 异常处理与重试策略
    """
    messages = state.get("messages", [])
    task_queue = state.get("task_queue", [])
    context = state.get("context", {})
    current_task = state.get("current_task", "")

    # 1. 初始时拆分任务并发送规划消息
    if not task_queue and not current_task:
        requirements = state.get("requirements", [state.get("input", "")])
        context["requirements"] = requirements
        messages.append(AgentMessage(
            from_agent="coordinator", to_agent="all", task_type="plan",
            payload={
                "plan": [
                    {"step": 1, "agent": "test_generator", "task": "生成测试用例"},
                    {"step": 2, "agent": "test_executor", "task": "执行测试"},
                    {"step": 3, "agent": "result_analyzer", "task": "分析结果"},
                    {"step": 4, "agent": "report_generator", "task": "生成报告"},
                ],
                "requirements": requirements,
            },
            status="success", timestamp=time.time(),
        ))
        current_task = AgentTask.GENERATE_TEST.value

    # 2. 检查当前任务的 Agent 是否已完成
    agent_map = {
        AgentTask.GENERATE_TEST.value: "test_generator",
        AgentTask.EXECUTE_TEST.value: "test_executor",
        AgentTask.ANALYZE_RESULT.value: "result_analyzer",
        AgentTask.GENERATE_REPORT.value: "report_generator",
    }
    next_map = {
        AgentTask.GENERATE_TEST.value: AgentTask.EXECUTE_TEST.value,
        AgentTask.EXECUTE_TEST.value: AgentTask.ANALYZE_RESULT.value,
        AgentTask.ANALYZE_RESULT.value: AgentTask.GENERATE_REPORT.value,
    }

    if current_task in agent_map:
        agent_name = agent_map[current_task]
        agent_msgs = [m for m in messages if m.get("from_agent") == agent_name and m.get("status") == "success"]
        if agent_msgs:
            if current_task in next_map:
                current_task = next_map[current_task]
            else:
                current_task = ""
                state["completed"] = True

    return {
        **state,
        "messages": messages,
        "task_queue": [],
        "current_task": current_task,
        "context": context,
        "error": None,
    }


# =========================
# Test Generator Agent
# =========================

def test_generator_node(state: MultiAgentState) -> MultiAgentState:
    """
    Test Generator Agent — 专门负责测试用例生成。

    被 Coordinator 调度，接收需求列表，返回结构化测试用例。
    """
    messages = state.get("messages", [])
    context = state.get("context", {})
    requirements = context.get("requirements", [])

    from test_agent.test_generator import generate_test_cases
    test_cases = generate_test_cases(requirements)

    messages.append(AgentMessage(
        from_agent="test_generator",
        to_agent="coordinator",
        task_type=AgentTask.GENERATE_TEST.value,
        payload={
            "test_cases": test_cases,
            "count": len(test_cases.get("test_cases", [])),
        },
        status="success",
        timestamp=time.time(),
    ))

    return {
        **state,
        "messages": messages,
        "test_cases": test_cases,
        "context": {**context, "test_case_count": len(test_cases.get("test_cases", []))},
    }


# =========================
# Test Executor Agent
# =========================

def test_executor_node(state: MultiAgentState) -> MultiAgentState:
    """
    Test Executor Agent — 专门负责测试执行。

    接收测试用例，执行并返回结果。
    """
    messages = state.get("messages", [])
    test_cases = state.get("test_cases", {})
    cases_list = test_cases.get("test_cases", [])

    from test_agent.test_executor import TestExecutor
    executor = TestExecutor(use_mock=True)
    results = executor.run_batch(cases_list, verbose=False)
    exec_report = executor.report(results)

    messages.append(AgentMessage(
        from_agent="test_executor",
        to_agent="coordinator",
        task_type=AgentTask.EXECUTE_TEST.value,
        payload={
            "results": results,
            "report": exec_report,
        },
        status="success",
        timestamp=time.time(),
    ))

    return {
        **state,
        "messages": messages,
        "execution_results": results,
    }


# =========================
# Result Analyzer Agent
# =========================

def result_analyzer_node(state: MultiAgentState) -> MultiAgentState:
    """
    Result Analyzer Agent — 专门负责结果分析。

    分析失败原因、匹配缺陷库、评估输出质量。
    """
    messages = state.get("messages", [])
    results = state.get("execution_results", [])

    from test_agent.test_analyzer import batch_analyze
    analysis = batch_analyze(results)

    # 统计
    passed = sum(1 for r in results if r.get("path_match"))
    total = len(results)
    bugs_found = sum(1 for a in analysis if a.get("known_bugs"))

    messages.append(AgentMessage(
        from_agent="result_analyzer",
        to_agent="coordinator",
        task_type=AgentTask.ANALYZE_RESULT.value,
        payload={
            "analysis_count": len(analysis),
            "passed": passed,
            "total": total,
            "bugs_found": bugs_found,
            "quality_summary": {
                "good": sum(1 for a in analysis if a.get("quality", {}).get("overall_quality") == "good"),
                "needs_review": sum(1 for a in analysis if a.get("quality", {}).get("overall_quality") == "needs_review"),
                "poor": sum(1 for a in analysis if a.get("quality", {}).get("overall_quality") == "poor"),
            },
        },
        status="success",
        timestamp=time.time(),
    ))

    return {
        **state,
        "messages": messages,
        "analysis": analysis,
    }


# =========================
# Report Generator Agent
# =========================

def report_generator_node(state: MultiAgentState) -> MultiAgentState:
    """
    Report Generator Agent — 专门负责报告生成。

    聚合所有 Agent 的产出，生成最终报告。
    """
    messages = state.get("messages", [])
    results = state.get("execution_results", [])
    analysis = state.get("analysis", [])
    test_cases = state.get("test_cases", {})

    passed = sum(1 for r in results if r.get("path_match")) if results else 0
    total = len(results) if results else 0

    # 收集各 Agent 的日志
    agent_logs = []
    for m in messages:
        agent_logs.append(f"    [{m['from_agent']}] {m['task_type']} → {m['status']}")

    report_lines = [
        "=" * 60,
        "  Multi-Agent Test Report",
        "=" * 60,
    ]

    # Agent 执行概览
    report_lines.append("\n  📋 Agent Execution Log:")
    for log in agent_logs:
        report_lines.append(log)

    # 测试结果
    report_lines.extend([
        "\n  📊 Test Results:",
        f"    Requirements:   {len(state.get('requirements', []))}",
        f"    Test Cases:     {len(test_cases.get('test_cases', []))}",
        f"    Passed:         {passed}/{total} ({passed/total*100 if total > 0 else 0:.1f}%)",
    ])

    # 缺陷发现
    if analysis:
        bugs = []
        for a in analysis:
            for b in a.get("known_bugs", []):
                if b["bug"]["id"] not in bugs:
                    bugs.append(b["bug"]["id"])
        if bugs:
            report_lines.append(f"\n  🐛 Bugs Matched:")
            for bug_id in bugs:
                report_lines.append(f"    - {bug_id}")

    # 质量问题
    if analysis:
        poor = sum(1 for a in analysis if a.get("quality", {}).get("overall_quality") in ("needs_review", "poor"))
        report_lines.append(f"\n  ⚠️  Quality Issues: {poor} need review")

    report_lines.append("\n" + "=" * 60)

    report = "\n".join(report_lines)

    messages.append(AgentMessage(
        from_agent="report_generator",
        to_agent="coordinator",
        task_type=AgentTask.GENERATE_REPORT.value,
        payload={"report": report},
        status="success",
        timestamp=time.time(),
    ))

    return {
        **state,
        "messages": messages,
        "report": report,
    }


# =========================
# 路由决策
# =========================

def coordinator_routing(state: MultiAgentState) -> str:
    """Coordinator 根据 current_task 路由到对应的 Agent。"""
    current = state.get("current_task", "")
    if state.get("completed"):
        return END
    if state.get("error"):
        return "end"

    route_map = {
        AgentTask.GENERATE_TEST.value: "test_generator",
        AgentTask.EXECUTE_TEST.value: "test_executor",
        AgentTask.ANALYZE_RESULT.value: "result_analyzer",
        AgentTask.GENERATE_REPORT.value: "report_generator",
    }
    return route_map.get(current, END)


# =========================
# 构建图
# =========================

def build_multi_agent():
    """构建多 Agent 协调图。"""
    graph = StateGraph(MultiAgentState)

    # 添加 Agent 节点
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("test_generator", test_generator_node)
    graph.add_node("test_executor", test_executor_node)
    graph.add_node("result_analyzer", result_analyzer_node)
    graph.add_node("report_generator", report_generator_node)

    # 入口
    graph.set_entry_point("coordinator")

    # Coordinator → 各 Agent → Coordinator
    graph.add_conditional_edges(
        "coordinator",
        coordinator_routing,
        {
            "test_generator": "test_generator",
            "test_executor": "test_executor",
            "result_analyzer": "result_analyzer",
            "report_generator": "report_generator",
            END: END,
        }
    )

    # 所有 Agent 执行完毕后回到 Coordinator
    for agent in ["test_generator", "test_executor", "result_analyzer", "report_generator"]:
        graph.add_edge(agent, "coordinator")

    return graph.compile()


# =========================
# CLI
# =========================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Test Coordinator")
    parser.add_argument("--requirements", "-r", type=str,
                        default="测试主体确认, 测试RAG查询, 搜索功能",
                        help="需求描述")
    args = parser.parse_args()

    requirements = [r.strip() for r in args.requirements.split(",")]

    print("=" * 60)
    print("  🤖 Multi-Agent Test Coordinator (v2.0)")
    print("=" * 60)
    print(f"  Requirements: {requirements}")
    print("-" * 60)

    graph = build_multi_agent()
    result = graph.invoke({
        "input": args.requirements,
        "requirements": requirements,
        "messages": [],
        "current_task": "",
        "task_queue": [],
        "context": {},
        "test_cases": None,
        "execution_results": None,
        "analysis": None,
        "report": None,
        "error": None,
        "completed": False,
    })

    print()
    print(result.get("report", "No report generated"))


if __name__ == "__main__":
    main()
