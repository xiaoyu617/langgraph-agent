"""
质量闭环监控系统 — Quality Monitor (v2.0)

核心功能:
  1. 质量指标追踪：跨运行记录路径准确率、幻觉率、延迟等
  2. 趋势分析：检测质量下降趋势
  3. 缺陷根因分析：通过 Trace Replay 分析异常路径
  4. 回归触发：质量下降时自动触发回归测试

适合多多迦游岗位要求的"跟踪线上质量指标，分析缺陷根因，推动持续改进"。
"""
import json
import os
import time
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# =========================
# 数据模型
# =========================

@dataclass
class QualitySnapshot:
    """单次质量快照。"""
    timestamp: float
    run_id: str
    path_accuracy: float
    avg_latency_ms: float
    hallucination_rate: float
    safety_rate: float
    throughput_req_s: float
    sample_count: int
    passed_count: int
    failed_count: int
    error_rate: float
    dimension_scores: dict = field(default_factory=dict)


@dataclass
class QualityTrend:
    """质量趋势分析结果。"""
    is_degrading: bool = False
    degrading_dimensions: list[str] = field(default_factory=list)
    pct_changes: dict = field(default_factory=dict)
    alert_message: str = ""


@dataclass
class RootCauseReport:
    """根因分析报告。"""
    case_id: str
    failure_type: str
    root_cause: str
    trace_analysis: str
    suggestion: str
    triggered_regression: bool = False


# =========================
# 持久化存储
# =========================

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", ".quality_history.json")


def _load_history() -> list[dict]:
    """加载历史质量数据。"""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def _save_history(history: list[dict]):
    """保存历史质量数据。"""
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# =========================
# 质量快照采集
# =========================

def collect_snapshot(
    run_id: str,
    test_results: list[dict],
    eval_samples: Optional[list] = None,
) -> QualitySnapshot:
    """
    从测试结果采集质量快照。

    Args:
        run_id: 运行标识
        test_results: test_executor 的执行结果
        eval_samples: evaluator_advanced 的评测样本 (可选)
    """
    total = len(test_results)
    passed = sum(1 for r in test_results if r.get("path_match"))
    failed = total - passed
    latencies = [r.get("latency_ms", 0) for r in test_results if r.get("latency_ms")]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    # 计算错误率
    errors = sum(1 for r in test_results if r.get("errors"))
    error_rate = errors / total if total > 0 else 0

    # 评测维度得分
    dim_scores = {}
    if eval_samples:
        hallu_scores = [s.get("hallucination", {}).get("factual_contradiction", 1.0)
                        for s in eval_samples]
        safety_scores = [s.get("hallucination", {}).get("faithfulness", 1.0)
                         for s in eval_samples]
        dim_scores["factual_accuracy"] = round(sum(hallu_scores) / len(hallu_scores), 3) if hallu_scores else 1.0
        dim_scores["faithfulness"] = round(sum(safety_scores) / len(safety_scores), 3) if safety_scores else 1.0
        dim_scores["avg_ttft_ms"] = round(
            sum(s.get("ttft_ms", 0) for s in eval_samples) / len(eval_samples), 2
        ) if eval_samples else 0
        dim_scores["avg_tps"] = round(
            sum(s.get("tps", 0) for s in eval_samples) / len(eval_samples), 1
        ) if eval_samples else 0

    snapshot = QualitySnapshot(
        timestamp=time.time(),
        run_id=run_id,
        path_accuracy=round(passed / total, 3) if total > 0 else 0,
        avg_latency_ms=round(avg_lat, 2),
        hallucination_rate=round(1 - dim_scores.get("factual_accuracy", 1), 3),
        safety_rate=dim_scores.get("faithfulness", 1.0),
        throughput_req_s=round(len(latencies) / (sum(latencies) / 1000), 2) if latencies else 0,
        sample_count=total,
        passed_count=passed,
        failed_count=failed,
        error_rate=round(error_rate, 3),
        dimension_scores=dim_scores,
    )

    # 持久化
    history = _load_history()
    history.append(asdict(snapshot))
    # 只保留最近 100 条
    if len(history) > 100:
        history = history[-100:]
    _save_history(history)

    return snapshot


# =========================
# 趋势分析
# =========================

def analyze_trend(window: int = 5) -> QualityTrend:
    """
    分析质量趋势。

    Args:
        window: 最近 N 次运行对比窗口

    Returns:
        QualityTrend: 趋势分析结果
    """
    history = _load_history()
    if len(history) < window * 2:
        return QualityTrend(
            alert_message=f"数据不足: {len(history)}/{window*2} 需要更多数据"
        )

    recent = history[-window:]
    baseline = history[-(window*2):-window]

    degrading_dims = []
    pct_changes = {}

    # 对比指标
    metrics = [
        ("path_accuracy", "路径准确率", True),
        ("avg_latency_ms", "平均延迟", False),
        ("hallucination_rate", "幻觉率", False),
        ("safety_rate", "安全合规率", True),
        ("throughput_req_s", "吞吐量", True),
        ("error_rate", "错误率", False),
    ]

    for key, name, higher_is_better in metrics:
        recent_avg = sum(s[key] for s in recent) / window
        baseline_avg = sum(s[key] for s in baseline) / window
        pct_change = ((recent_avg - baseline_avg) / baseline_avg * 100) if baseline_avg != 0 else 0
        pct_changes[name] = round(pct_change, 1)

        # 判断劣化
        if higher_is_better and pct_change < -5:
            degrading_dims.append(name)
        elif not higher_is_better and pct_change > 5:
            degrading_dims.append(name)

    alert = ""
    if degrading_dims:
        alert = (
            f"⚠️ 质量劣化告警: 以下指标在最近 {window} 次运行中显著下降:\n"
            + "\n".join(f"  - {d}: {pct_changes.get(d, 0):+.1f}%" for d in degrading_dims)
        )
    else:
        alert = "✅ 质量指标稳定"

    return QualityTrend(
        is_degrading=len(degrading_dims) > 0,
        degrading_dimensions=degrading_dims,
        pct_changes=pct_changes,
        alert_message=alert,
    )


# =========================
# 根因分析
# =========================

def analyze_root_cause(
    result: dict,
    analyze_trace: bool = True,
) -> RootCauseReport:
    """
    对单个失败用例进行根因分析。

    通过 Trace Replay 分析:
      1. Router 决策是否正确
      2. 哪个节点出了问题
      3. 输入/输出是否异常
    """
    trace = result.get("trace", [])
    actual_path = result.get("actual_path", [])
    expected_path = result.get("expected_path", [])
    errors = result.get("errors", [])

    # 1. 检查是否有执行错误
    if errors:
        return RootCauseReport(
            case_id=result.get("case_id", "unknown"),
            failure_type="execution_error",
            root_cause=f"执行异常: {errors[0].get('error', '未知错误')}",
            trace_analysis=f"在第 {errors[0].get('turn', '?')} 轮出错",
            suggestion="检查 Agent 图的节点实现是否有 bug",
        )

    # 2. 分析 Router 决策
    router_step = None
    for s in trace:
        if s.get("node") == "router":
            router_step = s
            break

    if router_step and expected_path:
        decision = router_step.get("decision", {})
        expected_last = expected_path[-1]

        if expected_last == "rag_retrieve" and not decision.get("rag"):
            return RootCauseReport(
                case_id=result.get("case_id", "unknown"),
                failure_type="router_decision_error",
                root_cause="Router 未检测到 RAG 条件",
                trace_analysis=(
                    f"输入: '{router_step.get('input', '')}' "
                    f"决策: {decision}"
                ),
                suggestion="检查 Router 中 need_rag 条件是否覆盖了该输入模式",
                triggered_regression=True,
            )

        if expected_last == "search" and not decision.get("search"):
            return RootCauseReport(
                case_id=result.get("case_id", "unknown"),
                failure_type="router_decision_error",
                root_cause="Router 未检测到 Search 条件",
                trace_analysis=f"输入: '{router_step.get('input', '')}' 决策: {decision}",
                suggestion="检查 Router 中 need_search 条件",
                triggered_regression=True,
            )

    # 3. 分析节点执行
    if actual_path and expected_path and actual_path[-1] != expected_path[-1]:
        return RootCauseReport(
            case_id=result.get("case_id", "unknown"),
            failure_type="path_mismatch",
            root_cause=f"路径终止节点不匹配: 期望={expected_path[-1]}, 实际={actual_path[-1]}",
            trace_analysis=f"完整路径: {'→'.join(actual_path)}",
            suggestion="检查条件边的路由逻辑",
            triggered_regression=True,
        )

    # 4. Trace Replay 异常检测
    if analyze_trace and trace:
        for step in trace:
            # 检查 RAG 上下文为空
            if step.get("node") == "rag_answer" and not step.get("context"):
                return RootCauseReport(
                    case_id=result.get("case_id", "unknown"),
                    failure_type="empty_context",
                    root_cause="RAG 上下文为空，LLM 可能产生幻觉",
                    trace_analysis=f"RAG query: {step.get('rag_query', 'N/A')}",
                    suggestion="在 rag_answer_node 中添加空 context 处理",
                    triggered_regression=True,
                )

    return RootCauseReport(
        case_id=result.get("case_id", "unknown"),
        failure_type="unknown",
        root_cause="未找到明确的根因",
        trace_analysis=f"执行路径: {'→'.join(actual_path)}",
        suggestion="请人工分析 trace 日志",
    )


# =========================
# 回归触发
# =========================

def should_trigger_regression(trend: QualityTrend) -> bool:
    """
    判断是否需要触发回归测试。

    条件:
      - 路径准确率下降超过 5%
      - 幻觉率上升超过 10%
    """
    if not trend.is_degrading:
        return False

    critical_dims = {"路径准确率", "幻觉率", "安全合规率"}
    for dim in trend.degrading_dimensions:
        if dim in critical_dims:
            pct = trend.pct_changes.get(dim, 0)
            if abs(pct) > 5:
                return True
    return False


# =========================
# 质量看板
# =========================

def generate_dashboard(history: Optional[list] = None) -> str:
    """生成质量看板文字报告。"""
    if history is None:
        history = _load_history()

    if not history:
        return "暂无质量数据"

    latest = history[-1]
    trend = analyze_trend()

    lines = [
        "=" * 60,
        "  📊 质量看板 (Quality Dashboard)",
        "=" * 60,
        f"  运行次数:       {len(history)}",
        f"  最新一次:       {latest.get('run_id', 'N/A')}",
        "-" * 60,
        "  当前指标:",
        f"    路径准确率:   {latest.get('path_accuracy', 0):.1%}",
        f"    平均延迟:     {latest.get('avg_latency_ms', 0):.1f} ms",
        f"    幻觉率:       {latest.get('hallucination_rate', 0):.1%}",
        f"    安全合规:     {latest.get('safety_rate', 0):.1%}",
        f"    吞吐量:       {latest.get('throughput_req_s', 0):.1f} req/s",
        f"    错误率:       {latest.get('error_rate', 0):.1%}",
        f"    样本数:       {latest.get('sample_count', 0)}",
        "-" * 60,
        "  趋势分析:",
    ]

    for name, pct in trend.pct_changes.items():
        arrow = "↑" if pct > 0 else "↓"
        lines.append(f"    {name:12s}: {arrow} {abs(pct):.1f}%")

    lines.append(f"\n  {trend.alert_message}")

    if len(history) >= 2:
        prev = history[-2]
        lines.append("-" * 60)
        lines.append("  与前次对比:")
        for key, name in [("path_accuracy", "准确率"), ("avg_latency_ms", "延迟"),
                          ("hallucination_rate", "幻觉率"), ("error_rate", "错误率")]:
            delta = latest.get(key, 0) - prev.get(key, 0)
            arrow = "↑" if delta > 0 else "↓"
            lines.append(f"    {name}: {arrow} {abs(delta):.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)


# =========================
# CLI
# =========================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quality Monitor")
    parser.add_argument("--dashboard", action="store_true", help="显示质量看板")
    parser.add_argument("--trend", action="store_true", help="显示趋势分析")
    parser.add_argument("--history", action="store_true", help="显示历史记录")
    args = parser.parse_args()

    if args.dashboard:
        print(generate_dashboard())
        return

    if args.trend:
        trend = analyze_trend()
        print(trend.alert_message)
        for name, pct in trend.pct_changes.items():
            print(f"  {name}: {pct:+.1f}%")
        if should_trigger_regression(trend):
            print("\n🔁 触发回归测试!")
        return

    if args.history:
        history = _load_history()
        for h in history[-10:]:
            print(f"  [{h.get('run_id', '?')}] "
                  f"acc={h.get('path_accuracy', 0):.1%} "
                  f"lat={h.get('avg_latency_ms', 0):.1f}ms "
                  f"err={h.get('error_rate', 0):.1%}")
        return

    # Default: show full quality report
    print(generate_dashboard())

    trend = analyze_trend()
    if should_trigger_regression(trend):
        print("\n🔁 自动触发回归测试...")
        # 在实际 CI/CD 中会调用 API 触发新的 workflow run


if __name__ == "__main__":
    main()
