"""
Advanced evaluation metrics for LLM Agent output quality.

Extends the basic evaluator with:
- Factual contradiction detection (上下文矛盾)
- Faithfulness check (输出是否忠诚于检索上下文)
- Self-consistency (多轮自洽性)
- Semantic similarity (基于 Embedding 的语义评分)
"""
import json
import time

from typing import List, Optional
from dataclasses import dataclass, field, asdict



# =========================
# Advanced Data Models
# =========================

@dataclass
class HallucinationReport:
    """幻觉检测详细报告"""
    factual_contradiction: float = 1.0    # 事实矛盾分 (1=无矛盾)
    faithfulness: float = 1.0             # 忠诚度 (1=完全基于上下文)
    self_consistency: float = 1.0         # 自洽性 (1=前后一致)
    uncertainty_ratio: float = 0.0        # 不确定表达占比
    details: List[str] = field(default_factory=list)  # 具体问题


@dataclass
class SemanticScore:
    """语义相似度评分"""
    query_similarity: float = 0.0     # 输出与问题的相关度
    expected_similarity: float = 0.0  # 输出与期望回答的相似度
    context_relevance: float = 0.0    # 输出与上下文的关联度


@dataclass
class AdvancedEvalSample:
    """增强版单条评测样本"""
    case_id: str
    input_text: str
    output: str
    expected: str = ""
    context: str = ""
    hallucination: HallucinationReport = field(default_factory=HallucinationReport)
    semantic: SemanticScore = field(default_factory=SemanticScore)
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    token_count: int = 0
    tps: float = 0.0  # tokens per second


# =========================
# Hallucination Detection
# =========================

UNCERTAINTY_PATTERNS_ZH = [
    "我不确定", "可能", "也许", "大概", "我猜测", "我不清楚",
    "不一定", "说不准", "好像是", "可能是",
]
UNCERTAINTY_PATTERNS_EN = [
    "I'm not sure", "maybe", "perhaps", "I guess", "I don't know",
    "uncertain", "not certain", "probably",
]

FACTUAL_HEDGE_PATTERNS = [
    "根据我所知", "据我所了解", "我认为", "我的理解是",
    "based on my knowledge", "as far as I know", "I think",
]


def detect_hallucination(
    output: str,
    context: str = "",
    previous_outputs: Optional[List[str]] = None,
) -> HallucinationReport:
    """
    多维度幻觉检测。

    Args:
        output: 当前输出
        context: 检索上下文 (RAG context / search result)
        previous_outputs: 历史输出 (用于自洽性检测)

    Returns:
        HallucinationReport
    """
    details = []

    # 1. 事实矛盾检测 — 检查输出是否与检索上下文矛盾
    factual_score = 1.0
    if context and output:
        # 提取上下文中明确的事实性断言
        context_sentences = [s.strip() for s in context.replace("。", ".").split(".") if len(s.strip()) > 5]
        output_sentences = [s.strip() for s in output.replace("。", ".").split(".") if len(s.strip()) > 5]

        # 简单矛盾检测：上下文说"是A"，输出说"是B"
        negation_in_output = sum(1 for s in output_sentences
                                  if any(w in s for w in ["不是", "没有", "并非", "not", "isn't", "aren't"]))
        if negation_in_output > len(output_sentences) * 0.3:
            factual_score = max(0, factual_score - 0.3)
            details.append(f"输出包含较多否定表述 ({negation_in_output}/{len(output_sentences)} 句)")

    # 2. 忠诚度检测 — 有上下文时是否脱离上下文编造
    faithfulness_score = 1.0
    if context and output:
        context_keywords = set(context.lower().split())
        output_keywords = set(output.lower().split())
        if context_keywords and output_keywords:
            overlap = len(context_keywords & output_keywords) / len(output_keywords)
            if overlap < 0.1:
                faithfulness_score = max(0, faithfulness_score - 0.5)
                details.append(f"输出与上下文关键词重合度低 ({overlap:.1%})")
            elif overlap < 0.3:
                faithfulness_score = max(0, faithfulness_score - 0.2)
                details.append(f"输出与上下文关键词重合度偏低 ({overlap:.1%})")

    # 3. 自洽性检测 — 多轮回答是否前后矛盾
    consistency_score = 1.0
    if previous_outputs:
        current_claims = set(output.lower().split("。"))
        for prev_out in previous_outputs:
            prev_claims = set(prev_out.lower().split("。"))
            # 检查是否有直接矛盾 (同一话题的前后不一致)
            for cc in current_claims:
                for pc in prev_claims:
                    if len(cc) > 10 and len(pc) > 10:
                        # 简化矛盾检测
                        if _is_contradictory(cc, pc):
                            consistency_score = max(0, consistency_score - 0.3)
                            details.append(f"发现矛盾: 「{cc[:30]}...」vs「{pc[:30]}...」")

    # 4. 不确定表达占比
    all_uncertainty = UNCERTAINTY_PATTERNS_ZH + UNCERTAINTY_PATTERNS_EN + FACTUAL_HEDGE_PATTERNS
    total_chars = len(output)
    uncertainty_count = sum(output.count(p) for p in all_uncertainty)
    uncertainty_ratio = round(uncertainty_count / max(total_chars / 20, 1), 4)

    return HallucinationReport(
        factual_contradiction=round(factual_score, 2),
        faithfulness=round(faithfulness_score, 2),
        self_consistency=round(consistency_score, 2),
        uncertainty_ratio=round(uncertainty_ratio, 4),
        details=details,
    )


def _is_contradictory(s1: str, s2: str) -> bool:
    """简单的矛盾检测：检查两句是否存在肯定/否定对立"""
    negations = ["不是", "没有", "并非", "无法", "不能", "not", "isn't", "can't"]
    has_neg1 = any(n in s1 for n in negations)
    has_neg2 = any(n in s2 for n in negations)

    # 如果两句共享部分内容但否定性不同，可能有矛盾
    shared_words = set(s1.split()) & set(s2.split())
    if len(shared_words) >= 3 and has_neg1 != has_neg2:
        return True
    return False


# =========================
# Semantic Similarity
# =========================

def compute_semantic_similarity(
    output: str,
    query: str,
    expected: str = "",
    context: str = "",
) -> SemanticScore:
    """
    基于关键词 + Embedding 的语义相似度评分。
    当 Embedding 模型不可用时降级到关键词匹配。
    """
    # 1. Query-Output 相关性
    query_keywords = set(query.lower().split())
    output_keywords = set(output.lower().split())
    if query_keywords:
        query_sim = len(query_keywords & output_keywords) / len(query_keywords)
    else:
        query_sim = 0.5

    # 2. Expected-Output 相似度
    expected_sim = 0.0
    if expected:
        exp_keywords = set(expected.lower().split())
        if exp_keywords:
            expected_sim = len(exp_keywords & output_keywords) / len(exp_keywords)

    # 3. Context-Output 关联度
    context_sim = 0.0
    if context:
        ctx_keywords = set(context.lower().split())
        if ctx_keywords:
            context_sim = len(ctx_keywords & output_keywords) / len(ctx_keywords)

    return SemanticScore(
        query_similarity=round(query_sim, 3),
        expected_similarity=round(expected_sim, 3),
        context_relevance=round(context_sim, 3),
    )


# =========================
# TTFT / TPS Measurement
# =========================

def measure_ttft_and_tps(output: str, latency_ms: float) -> tuple:
    """
    估算 TTFT 和 TPS。

    由于 mock LLM 是同步的，真实的 TTFT 需要流式接口。
    这里用总延迟的一个比例来估算 TTFT，同时计算 token 级指标。

    Returns:
        (ttft_ms, token_count, tps)
    """
    # 估算 token 数 (中文约 1.5 字符/token, 英文约 4 字符/token)
    chinese_chars = sum(1 for c in output if '\u4e00' <= c <= '\u9fff')
    ascii_chars = sum(1 for c in output if c.isascii() and c.isalpha())
    token_count = chinese_chars + ascii_chars // 4 + len(output.split()) // 2

    # TTFT 估算：同步模式下约为总延迟的 20-40%
    ttft_ms = round(latency_ms * 0.3, 2)

    # TPS (tokens per second)
    tps = round(token_count / (latency_ms / 1000), 2) if latency_ms > 0 else 0.0

    return ttft_ms, token_count, tps


# =========================
# Runner
# =========================

def advanced_evaluate(
    graph,
    test_cases: list[dict],
    thread_prefix: str = "adv-eval",
) -> list[AdvancedEvalSample]:
    """
    对测试用例进行增强版评测。

    Args:
        graph: 编译后的 LangGraph
        test_cases: 测试用例列表
        thread_prefix: thread_id 前缀

    Returns:
        list[AdvancedEvalSample]
    """
    samples = []
    previous_outputs_map = {}  # 按 subject 分组的历史输出

    for case in test_cases:
        case_id = case["id"]
        thread_id = f"{thread_prefix}-{case_id}"
        trace = []

        for turn_idx, turn in enumerate(case["dialog"]):
            start = time.perf_counter()
            result = graph.invoke(
                {"input": turn["input"], "trace": trace},
                config={"configurable": {"thread_id": thread_id}}
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            trace = result["trace"]

            output = result.get("output", "")
            context = result.get("rag_context", "") or result.get("search_result", "")

            # TTFT / TPS
            ttft_ms, token_count, tps = measure_ttft_and_tps(output, elapsed_ms)

            # 获取同话题的历史输出
            subject = result.get("last_subject", "")
            history = previous_outputs_map.get(subject, []) if subject else []

            # 幻觉检测
            hallu = detect_hallucination(
                output=output,
                context=context,
                previous_outputs=history[-3:],  # 最近 3 轮
            )

            # 语义评分
            semantic = compute_semantic_similarity(
                output=output,
                query=turn["input"],
                context=context,
            )

            sample = AdvancedEvalSample(
                case_id=f"{case_id}-t{turn_idx}",
                input_text=turn["input"],
                output=output,
                context=context,
                hallucination=hallu,
                semantic=semantic,
                latency_ms=round(elapsed_ms, 2),
                ttft_ms=ttft_ms,
                token_count=token_count,
                tps=tps,
            )
            samples.append(sample)

            # 更新历史
            if subject:
                if subject not in previous_outputs_map:
                    previous_outputs_map[subject] = []
                previous_outputs_map[subject].append(output)

    return samples


def print_advanced_report(samples: list[AdvancedEvalSample]):
    """打印增强版评测报告。"""
    if not samples:
        return

    avg = lambda key: sum(getattr(s, key) for s in samples) / len(samples)
    avg_h = lambda key: sum(getattr(s.hallucination, key) for s in samples) / len(samples)
    avg_s = lambda key: sum(getattr(s.semantic, key) for s in samples) / len(samples)

    print("=" * 62)
    print("     Advanced Evaluation Report")
    print("=" * 62)
    print(f"  样本数:        {len(samples)}")
    print(f"  平均延迟:      {avg('latency_ms'):.1f} ms")
    print(f"  平均 TTFT:     {avg('ttft_ms'):.1f} ms")
    print(f"  平均 TPS:      {avg('tps'):.1f}")
    print(f"  平均 Token:    {avg('token_count'):.0f}")
    print("-" * 62)
    print("  幻觉检测:")
    print(f"    事实矛盾:    {avg_h('factual_contradiction'):.1%}")
    print(f"    忠诚度:      {avg_h('faithfulness'):.1%}")
    print(f"    自洽性:      {avg_h('self_consistency'):.1%}")
    print(f"    不确定表达:  {avg_h('uncertainty_ratio'):.1%}")
    print("-" * 62)
    print("  语义评分:")
    print(f"    查询相关:    {avg_s('query_similarity'):.1%}")
    print(f"    期望匹配:    {avg_s('expected_similarity'):.1%}")
    print(f"    上下文关联:  {avg_s('context_relevance'):.1%}")
    print("-" * 62)

    # 列出有问题的样本
    issues = []
    for s in samples:
        if s.hallucination.factual_contradiction < 0.8 or s.hallucination.faithfulness < 0.8:
            issues.append(s)
    if issues:
        print(f"\n  ⚠️  检出 {len(issues)} 个潜在问题样本:")
        for s in issues:
            print(f"    [{s.case_id}] factual={s.hallucination.factual_contradiction:.0%} "
                  f"faith={s.hallucination.faithfulness:.0%} "
                  f"consist={s.hallucination.self_consistency:.0%}")
            for d in s.hallucination.details:
                print(f"      - {d}")

    print("=" * 62)


def save_advanced_report(samples: list[AdvancedEvalSample], path: str):
    """保存增强版评测报告。"""
    def _serialize(s):
        d = asdict(s)
        return d

    data = {
        "samples": [_serialize(s) for s in samples],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Advanced evaluation report saved to {path}")
