"""
测试编排 Agent — 基于 LangGraph 的智能测试闭环系统。

流程:
  User Input (需求) → Generator Agent → Executor Agent → Analyzer Agent → Report

这就是北汽岗位要求的"需求转换 → 用例自动化生成 → 测试执行编排 → 结果分析"闭环。
"""
import json
import os
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END


# =========================
# 状态定义
# =========================

class OrchestratorState(TypedDict):
    """编排 Agent 状态。"""
    input: str                          # 原始需求输入
    requirements: List[str]             # 解析后的需求列表
    test_cases: Optional[dict]          # 生成的测试用例
    execution_results: Optional[list]   # 执行结果
    analysis: Optional[list]            # 分析结果
    report: Optional[str]               # 最终报告
    error: Optional[str]                # 错误信息
    path: List[str]                     # 执行路径记录


# =========================
# 节点
# =========================

def parse_requirements_node(state: OrchestratorState) -> OrchestratorState:
    """解析用户输入，提取需求列表。"""
    text = state["input"]
    path = state.get("path", [])

    # 按行/逗号/分号拆分需求
    import re
    lines = re.split(r'[\n,;。]', text)
    requirements = [l.strip() for l in lines if l.strip()]

    path.append("parse_requirements")
    return {
        "input": state["input"],
        "requirements": requirements,
        "test_cases": None,
        "execution_results": None,
        "analysis": None,
        "report": None,
        "error": None,
        "path": path,
    }


def generate_cases_node(state: OrchestratorState) -> OrchestratorState:
    """根据需求生成测试用例。"""
    from test_agent.test_generator import generate_test_cases

    requirements = state.get("requirements", [])
    path = state.get("path", [])

    if not requirements:
        return {**state, "error": "No requirements to process", "path": path + ["generate_cases"]}

    cases = generate_test_cases(requirements)
    path.append("generate_cases")
    return {
        **state,
        "test_cases": cases,
        "error": None,
        "path": path,
    }


def execute_tests_node(state: OrchestratorState) -> OrchestratorState:
    """执行测试用例。"""
    from test_agent.test_executor import TestExecutor

    test_cases = state.get("test_cases", {})
    path = state.get("path", [])

    cases_list = test_cases.get("test_cases", [])
    if not cases_list:
        return {**state, "error": "No test cases to execute", "path": path + ["execute_tests"]}

    executor = TestExecutor(use_mock=True)
    results = executor.run_batch(cases_list, verbose=True)

    path.append("execute_tests")
    return {
        **state,
        "execution_results": results,
        "error": None,
        "path": path,
    }


def analyze_results_node(state: OrchestratorState) -> OrchestratorState:
    """分析测试结果。"""
    from test_agent.test_analyzer import batch_analyze

    results = state.get("execution_results", [])
    path = state.get("path", [])

    if not results:
        return {**state, "error": "No execution results to analyze", "path": path + ["analyze_results"]}

    analysis = batch_analyze(results)
    path.append("analyze_results")
    return {
        **state,
        "analysis": analysis,
        "error": None,
        "path": path,
    }


def generate_report_node(state: OrchestratorState) -> OrchestratorState:
    """生成最终报告。"""
    results = state.get("execution_results", [])
    analysis = state.get("analysis", [])
    test_cases = state.get("test_cases", {})
    path = state.get("path", [])

    passed = sum(1 for r in results if r.get("path_match")) if results else 0
    total = len(results) if results else 0
    failed_cases = [r for r in (results or []) if not r.get("path_match")]

    report_lines = [
        "=" * 60,
        "  智能测试闭环 — 执行报告",
        "=" * 60,
        f"  需求数:     {len(state.get('requirements', []))}",
        f"  生成用例:   {len(test_cases.get('test_cases', []))}",
        f"  执行通过:   {passed}/{total}",
        f"  失败:       {total - passed}",
        f"  通过率:     {(passed/total*100 if total > 0 else 0):.1f}%",
        "-" * 60,
    ]

    if analysis:
        poor_quality = sum(
            1 for a in analysis
            if a.get("quality", {}).get("overall_quality") in ("needs_review", "poor")
        )
        report_lines.append(f"  质量问题:   {poor_quality} 个需要审查")
        report_lines.append("-" * 60)
        report_lines.append("  详细分析:")
        for a in analysis:
            status = "✅" if a.get("path_match") else "❌"
            quality = a.get("quality", {}).get("overall_quality", "?")
            report_lines.append(f"    {status} [{a['case_id']}] path={a['path_match']} quality={quality}")
            if a.get("known_bugs"):
                for m in a["known_bugs"]:
                    report_lines.append(f"      🐛 {m['bug']['id']} match={m['match_score']:.0%}")

    if failed_cases:
        report_lines.append("-" * 60)
        report_lines.append("  失败用例:")
        for c in failed_cases:
            report_lines.append(
                f"    [{c.get('case_id', '?')}] "
                f"expected={'→'.join(c.get('expected_path', []))} "
                f"actual={'→'.join(c.get('actual_path', []))}"
            )

    report_lines.append("=" * 60)
    report = "\n".join(report_lines)

    path.append("generate_report")
    return {
        **state,
        "report": report,
        "error": None,
        "path": path,
    }


# =========================
# 条件边
# =========================

def should_continue(state: OrchestratorState) -> str:
    """路由决策：是否有需求需要处理。"""
    if state.get("error"):
        return "end"
    if not state.get("test_cases"):
        return "generate"
    if not state.get("execution_results"):
        return "execute"
    if not state.get("analysis"):
        return "analyze"
    if not state.get("report"):
        return "report"
    return "end"


# =========================
# 构建图
# =========================

def build_orchestrator():
    """构建测试编排 Agent 图。"""
    graph = StateGraph(OrchestratorState)

    # 添加节点
    graph.add_node("parse_requirements", parse_requirements_node)
    graph.add_node("generate_cases", generate_cases_node)
    graph.add_node("execute_tests", execute_tests_node)
    graph.add_node("analyze_results", analyze_results_node)
    graph.add_node("generate_report", generate_report_node)

    # 设置入口
    graph.set_entry_point("parse_requirements")

    # 添加条件边
    graph.add_conditional_edges(
        "parse_requirements",
        should_continue,
        {
            "generate": "generate_cases",
            "end": END,
        }
    )
    graph.add_conditional_edges(
        "generate_cases",
        should_continue,
        {
            "execute": "execute_tests",
            "end": END,
        }
    )
    graph.add_conditional_edges(
        "execute_tests",
        should_continue,
        {
            "analyze": "analyze_results",
            "end": END,
        }
    )
    graph.add_conditional_edges(
        "analyze_results",
        should_continue,
        {
            "report": "generate_report",
            "end": END,
        }
    )
    graph.add_edge("generate_report", END)

    return graph.compile()


# =========================
# CLI 入口
# =========================

def main():
    """交互式编排 Agent CLI。"""
    import argparse
    parser = argparse.ArgumentParser(description="Test Orchestrator Agent")
    parser.add_argument("--requirements", "-r", type=str,
                        default="测试主体确认功能, 测试RAG知识库查询",
                        help="需求描述，逗号分隔")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="保存报告到文件")
    args = parser.parse_args()

    print("=" * 60)
    print("  🧪 测试编排 Agent")
    print("=" * 60)
    print(f"  输入需求: {args.requirements}")
    print("-" * 60)

    graph = build_orchestrator()
    result = graph.invoke({
        "input": args.requirements,
        "requirements": [],
        "test_cases": None,
        "execution_results": None,
        "analysis": None,
        "report": None,
        "error": None,
        "path": [],
    })

    print()
    print(result.get("report", "No report generated"))

    if args.output:
        with open(args.output, "w") as f:
            f.write(result.get("report", ""))
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
