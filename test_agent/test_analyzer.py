"""
测试结果分析器 — 分析失败原因、缺陷定位、改进建议。

通过以下方式分析测试失败:
  1. 路径比对: 预期路径 vs 实际路径
  2. Trace 分析: 每个节点的输入/输出检查
  3. RAG 知识库检索: 匹配已知缺陷
  4. LLM Judge: 对输出质量打分
"""
import json
import os
from typing import Optional


KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")


def load_bug_library() -> list[dict]:
    """加载缺陷库，返回结构化缺陷列表。"""
    path = os.path.join(KB_DIR, "bug_library.md")
    bugs = []
    if not os.path.exists(path):
        return bugs

    with open(path) as f:
        content = f.read()

    # 解析 Markdown 格式的缺陷
    current_bug = {}
    for line in content.split("\n"):
        if line.startswith("## BUG-"):
            if current_bug:
                bugs.append(current_bug)
            current_bug = {"id": line.replace("## ", "").strip()}
        elif line.startswith("- ") and current_bug:
            key, _, value = line[2:].partition(": ")
            current_bug[key.strip()] = value.strip()
    if current_bug:
        bugs.append(current_bug)
    return bugs


def match_known_bugs(result: dict) -> list[dict]:
    """
    将测试结果与已知缺陷库匹配。
    返回匹配到的缺陷列表。
    """
    bugs = load_bug_library()
    matches = []
    actual_path = result.get("actual_path", [])

    for bug in bugs:
        score = 0
        # 检查路径匹配
        bug_path = bug.get("路径", "")
        if bug_path and bug_path in "→".join(actual_path):
            score += 0.5

        # 检查输出中的关键词
        bug_phenomenon = bug.get("现象", "")
        output = result.get("output", "")
        if bug_phenomenon and any(kw in output for kw in bug_phenomenon.split()):
            score += 0.3

        if score >= 0.3:
            matches.append({
                "bug": bug,
                "match_score": round(score, 2),
                "suggestion": bug.get("修复", "请人工分析"),
            })
    return matches


def analyze_path_mismatch(result: dict) -> str:
    """
    分析路径不匹配的原因。
    返回分析结论。
    """
    expected = result.get("expected_path", [])
    actual = result.get("actual_path", [])

    if not actual:
        return "执行异常: trace 为空，可能在 router 之前就出错了"

    if len(actual) < len(expected):
        return "路径提前终止: 预期需要更多步骤"

    # 检查 router 决策
    router_step = None
    for s in result.get("trace", []):
        if s.get("node") == "router":
            router_step = s
            break

    if router_step:
        decision = router_step.get("decision", {})
        actual_last = actual[-1] if actual else "none"
        # 对比预期路由和实际决策
        expected_last = expected[-1]
        if expected_last == "rag_retrieve" and not decision.get("rag"):
            return "Router 决策错误: 预期走 RAG 路径，但条件不满足"
        if expected_last == "search" and not decision.get("search"):
            return "Router 决策错误: 预期走 Search 路径，但条件不满足"
        if expected_last == "clarify" and not decision.get("clarify"):
            return "Router 决策错误: 预期走 Clarify 路径，但条件不满足"
        if expected_last == "memory_update" and not decision.get("subject_confirm"):
            return "Router 决策错误: 预期走 Memory 路径，但条件不满足"

    return "路径不匹配: 请检查 Router 条件逻辑"


def analyze_output_quality(result: dict) -> dict:
    """
    分析输出质量。
    返回质量评分报告。
    """
    from langgraph_agent.evaluator_advanced import (
        detect_hallucination,
        compute_semantic_similarity,
    )

    output = result.get("output", "")
    trace = result.get("trace", [])

    # 从 trace 中提取 context
    context = ""
    for step in trace:
        if step.get("node") in ("rag_retrieve",):
            context = context or step.get("context", "")
        if step.get("node") in ("search",):
            context = context or step.get("search_result", "")

    hallu = detect_hallucination(output, context)
    semantic = compute_semantic_similarity(
        output=output,
        query=trace[0].get("input", "") if trace else "",
        context=context,
    )

    issues = []
    if hallu.factual_contradiction < 0.8:
        issues.append("❌ 存在事实矛盾")
    if hallu.faithfulness < 0.8:
        issues.append("⚠️ 输出脱离上下文")
    if hallu.self_consistency < 0.8:
        issues.append("⚠️ 多轮回答不自洽")
    if hallu.uncertainty_ratio > 0.1:
        issues.append(f"⚠️ 不确定表达较多 ({hallu.uncertainty_ratio:.1%})")
    if semantic.query_similarity < 0.1:
        issues.append("⚠️ 输出与问题相关性低")

    return {
        "hallucination": {
            "factual_contradiction": hallu.factual_contradiction,
            "faithfulness": hallu.faithfulness,
            "self_consistency": hallu.self_consistency,
            "uncertainty_ratio": hallu.uncertainty_ratio,
        },
        "semantic": {
            "query_similarity": semantic.query_similarity,
            "context_relevance": semantic.context_relevance,
        },
        "issues": issues,
        "overall_quality": "good" if len(issues) == 0 else "needs_review" if len(issues) <= 2 else "poor",
    }


def detailed_analysis(result: dict) -> dict:
    """
    对单个测试结果进行综合分析。

    Returns:
        包含路径分析、缺陷匹配、质量分析的综合报告
    """
    analysis = {
        "case_id": result.get("case_id", "unknown"),
        "path_match": result.get("path_match", False),
        "path_analysis": "",
        "known_bugs": [],
        "quality": {},
        "suggestions": [],
    }

    # 1. 路径分析
    if not result["path_match"]:
        analysis["path_analysis"] = analyze_path_mismatch(result)
        analysis["suggestions"].append(analysis["path_analysis"])

    # 2. 缺陷库匹配
    bug_matches = match_known_bugs(result)
    if bug_matches:
        analysis["known_bugs"] = bug_matches
        for m in bug_matches:
            analysis["suggestions"].append(
                f"[{m['bug']['id']}] {m['suggestion']}"
            )

    # 3. 输出质量分析
    analysis["quality"] = analyze_output_quality(result)
    analysis["suggestions"].extend(analysis["quality"]["issues"])

    # 4. Trace 可视化
    trace = result.get("trace", [])
    analysis["execution_flow"] = " → ".join(s["node"] for s in trace)

    return analysis


def batch_analyze(results: list[dict]) -> list[dict]:
    """批量分析测试结果。"""
    return [detailed_analysis(r) for r in results]


def print_analysis(analyses: list[dict]):
    """打印分析报告。"""
    print("=" * 60)
    print("         Test Result Analysis")
    print("=" * 60)

    total = len(analyses)
    failed = sum(1 for a in analyses if not a["path_match"])
    poor_quality = sum(
        1 for a in analyses
        if a["quality"].get("overall_quality") in ("needs_review", "poor")
    )

    print(f"  Total:      {total}")
    print(f"  Failures:   {failed}")
    print(f"  Quality:    {poor_quality} need review")
    print("-" * 60)

    for a in analyses:
        status = "✅" if a["path_match"] else "❌"
        quality = a["quality"].get("overall_quality", "?")
        print(f"\n  {status} [{a['case_id']}] path={a['path_match']} quality={quality}")
        print(f"     Flow: {a.get('execution_flow', 'N/A')}")
        if a["known_bugs"]:
            for m in a["known_bugs"]:
                print(f"     🐛 Matched: {m['bug']['id']} ({m['match_score']:.0%})")
        if a["quality"].get("issues"):
            for issue in a["quality"]["issues"]:
                print(f"     {issue}")
        if a["suggestions"]:
            for s in a["suggestions"][:3]:
                print(f"     → {s}")
