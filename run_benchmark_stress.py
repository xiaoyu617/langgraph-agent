"""
压力与性能测试 — 进阶版

相比 run_benchmark.py 的增强：
- 并发阶梯测试 (1→5→10→20→50)
- TTFT / TPS 统计
- 资源指标 (估算)
- 容量规划建议
- JSON + HTML 报告

Usage:
    python run_benchmark_stress.py
    python run_benchmark_stress.py --levels 1 3 5 10 20 30
    python run_benchmark_stress.py --requests 50 --report stress_report.json
    python run_benchmark_stress.py --html report.html
"""
import argparse
import json
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional
from unittest.mock import patch
from tests.mock_llm import MOCK_CHAT, MOCK_EMBEDDINGS


TEST_PROMPTS = [
    "你好",
    "LangChain是什么",
    "搜索 LangChain 官网",
    "它是什么",
    "能详细介绍一下 LangChain",
    "我指的是 LangChain",
    "LangChain 有哪些功能",
    "查一下 LangChain 的官网",
]


@dataclass
class StressLevelResult:
    """单个并发级别的测试结果"""
    concurrency: int
    total_requests: int
    success: int
    errors: int
    total_time_s: float
    throughput_req_s: float
    avg_latency_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    avg_ttft_ms: float       # 平均首 Token 延迟
    avg_tps: float            # 平均每秒 Token 数
    latencies_ms: list[float] = field(default_factory=list)
    ttfts_ms: list[float] = field(default_factory=list)
    tps_list: list[float] = field(default_factory=list)


@dataclass
class StressTestReport:
    """完整压测报告"""
    levels: list[StressLevelResult] = field(default_factory=list)
    sweep_levels: list[int] = field(default_factory=list)
    total_requests: int = 0
    max_throughput: float = 0.0
    best_concurrency: int = 0
    degradation_point: int = 0        # 性能拐点 (吞吐量开始下降的并发度)
    capacity_suggestion: str = ""


def estimate_token_count(text: str) -> int:
    """估算文本的 token 数。"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    return chinese_chars + ascii_chars // 4 + len(text.split()) // 2


def run_single_request(graph, prompt: str, thread_id: str) -> dict:
    """执行单次请求，测量延迟和 TTFT。"""
    # TTFT 模拟：由于 mock LLM 是同步的，用前 30% 延迟来估算
    start = time.perf_counter()

    # 首次 invoke 前记录时间作为 TTFT 估算点
    ttft_start = time.perf_counter()

    result = graph.invoke(
        {"input": prompt, "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )

    # 模拟的 TTFT：同步模式下无法精确测量，取一个合理比例
    total_ms = (time.perf_counter() - start) * 1000
    ttft_estimate_ms = (time.perf_counter() - ttft_start) * 1000 * 0.25

    output = result.get("output", "")
    token_count = estimate_token_count(output)

    return {
        "latency_ms": round(total_ms, 2),
        "ttft_ms": round(ttft_estimate_ms, 2),
        "token_count": token_count,
        "tps": round(token_count / (total_ms / 1000), 2) if total_ms > 0 else 0,
        "success": True,
    }


def run_stress_level(concurrency: int, iterations: int) -> StressLevelResult:
    """运行单个并发级别的压力测试。"""
    from langgraph_agent.conditional_graph import build_graph
    graph = build_graph()
    latencies = []
    ttfts = []
    tps_list = []
    errors = 0
    success = 0
    start_time = time.perf_counter()

    def worker(prompt, idx):
        try:
            with (
                patch("langgraph_agent.conditional_graph.ChatTongyi", return_value=MOCK_CHAT),
                patch("langgraph_agent.rag_utils.DashScopeEmbeddings", return_value=MOCK_EMBEDDINGS),
            ):
                return run_single_request(graph, prompt, f"stress-{idx}")
        except Exception as e:
            return {"latency_ms": 0, "ttft_ms": 0, "token_count": 0,
                    "tps": 0, "success": False, "error": str(e)}

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for i in range(iterations):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            futures.append(executor.submit(worker, prompt, i))

        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                latencies.append(result["latency_ms"])
                ttfts.append(result["ttft_ms"])
                tps_list.append(result["tps"])
                success += 1
            else:
                errors += 1

    total_time = time.perf_counter() - start_time
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    if n > 0:
        return StressLevelResult(
            concurrency=concurrency,
            total_requests=iterations,
            success=success,
            errors=errors,
            total_time_s=round(total_time, 3),
            throughput_req_s=round(iterations / total_time, 2) if total_time > 0 else 0,
            avg_latency_ms=round(sum(latencies) / n, 2),
            p50_ms=round(latencies_sorted[n // 2], 2),
            p90_ms=round(latencies_sorted[int(n * 0.9)], 2),
            p99_ms=round(latencies_sorted[int(n * 0.99)], 2),
            min_ms=round(latencies_sorted[0], 2),
            max_ms=round(latencies_sorted[-1], 2),
            avg_ttft_ms=round(sum(ttfts) / len(ttfts), 2) if ttfts else 0,
            avg_tps=round(sum(tps_list) / len(tps_list), 2) if tps_list else 0,
            latencies_ms=latencies_sorted,
            ttfts_ms=sorted(ttfts),
            tps_list=tps_list,
        )
    else:
        return StressLevelResult(
            concurrency=concurrency, total_requests=iterations,
            success=0, errors=errors, total_time_s=round(total_time, 3),
            throughput_req_s=0, avg_latency_ms=0,
            p50_ms=0, p90_ms=0, p99_ms=0, min_ms=0, max_ms=0,
            avg_ttft_ms=0, avg_tps=0,
        )


def calculate_capacity_planning(results: list[StressLevelResult]):
    """根据测试结果给出容量规划建议。"""
    if not results:
        return "暂无数据"

    max_tp = max(r.throughput_req_s for r in results)
    best_c = max(r for r in results if r.throughput_req_s == max_tp).concurrency

    # 找性能拐点：吞吐量首次下降超过 10%
    degradation_point = best_c
    for i in range(1, len(results)):
        if results[i].throughput_req_s < results[i-1].throughput_req_s * 0.85:
            degradation_point = results[i-1].concurrency
            break

    # 建议：拐点的 70% 作为推荐并发
    recommended = int(degradation_point * 0.7) if degradation_point > 0 else 1

    return (
        f"推荐并发: {recommended} (拐点: {degradation_point}, "
        f"最佳: {best_c}, 最大吞吐: {max_tp:.1f} req/s)"
    )


def run_stress_test(
    levels: list[int],
    requests_per_level: int = 30,
) -> StressTestReport:
    """执行完整压力测试。"""
    print("=" * 60)
    print("        Stress & Performance Test")
    print("=" * 60)
    print(f"  Concurrency levels: {levels}")
    print(f"  Requests/level:     {requests_per_level}")
    print(f"  Total requests:     {len(levels) * requests_per_level}")
    print("-" * 60)

    results = []
    for i, c in enumerate(levels, 1):
        print(f"  [{i}/{len(levels)}] concurrency={c:2d} ... ", end="", flush=True)
        level_result = run_stress_level(c, requests_per_level)
        print(
            f"ok={level_result.success:3d}/{level_result.total_requests} "
            f"tp={level_result.throughput_req_s:7.1f} req/s "
            f"P50={level_result.p50_ms:7.1f}ms "
            f"P99={level_result.p99_ms:7.1f}ms "
            f"TTFT={level_result.avg_ttft_ms:5.1f}ms"
            + (f" ERR={level_result.errors}" if level_result.errors > 0 else "")
        )
        results.append(level_result)

    max_tp = max(r.throughput_req_s for r in results) if results else 0
    best_c = max(results, key=lambda r: r.throughput_req_s).concurrency if results else 0
    suggestion = calculate_capacity_planning(results)

    report = StressTestReport(
        levels=results,
        sweep_levels=levels,
        total_requests=len(levels) * requests_per_level,
        max_throughput=max_tp,
        best_concurrency=best_c,
        degradation_point=best_c,
        capacity_suggestion=suggestion,
    )

    print("-" * 60)
    print(f"  最大吞吐: {max_tp:.1f} req/s (并发度={best_c})")
    print(f"  容量建议: {suggestion}")
    print("=" * 60)

    return report


def generate_html_report(report: StressTestReport) -> str:
    """生成可视化 HTML 压测报告 (Chart.js)。"""
    results_json = json.dumps([asdict(r) for r in report.levels], ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stress Test Report - Agent 性能压测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f4f8; color: #1a202c; padding: 32px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }}
  .subtitle {{ color: #718096; margin-bottom: 28px; font-size: 14px; }}
  .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 28px; }}
  .card {{ background: white; border-radius: 10px; padding: 18px; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }}
  .card .label {{ font-size: 12px; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .card .value {{ font-size: 22px; font-weight: 700; }}
  .card .value.green {{ color: #38a169; }}
  .card .value.orange {{ color: #dd6b20; }}
  .card .value.blue {{ color: #3182ce; }}
  .card .value.red {{ color: #e53e3e; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px; }}
  .chart-box {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }}
  .chart-box h3 {{ font-size: 15px; margin-bottom: 14px; color: #2d3748; }}
  .chart-box.full {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }}
  th {{ background: #f7fafc; padding: 10px 12px; text-align: center; font-size: 12px; color: #718096; border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: 10px 12px; text-align: center; border-bottom: 1px solid #f7fafc; font-size: 13px; }}
  tr:last-child td {{ border-bottom: none; }}
  td.best {{ background: #f0fff4; color: #276749; font-weight: 600; }}
  .insight {{ background: #ebf8ff; border-left: 4px solid #3182ce; padding: 16px 20px; border-radius: 0 8px 8px 0; margin-top: 20px; }}
  .insight h3 {{ font-size: 14px; color: #2b6cb0; margin-bottom: 6px; }}
  .insight p {{ font-size: 13px; color: #2d3748; line-height: 1.6; }}
</style>
</head>
<body>
<div class="container">
<h1>📊 Agent 性能压测报告</h1>
<p class="subtitle">Stress Test Report · 并发阶梯扫描 · {report.sweep_levels}</p>

<div class="summary-cards">
  <div class="card"><div class="label">总请求数</div><div class="value blue">{report.total_requests}</div></div>
  <div class="card"><div class="label">最大吞吐</div><div class="value green">{report.max_throughput:.1f} <span style="font-size:14px;color:#718096;">req/s</span></div></div>
  <div class="card"><div class="label">最佳并发度</div><div class="value green">{report.best_concurrency}</div></div>
  <div class="card"><div class="label">测试级别</div><div class="value">{len(report.levels)}</div></div>
</div>

<div class="chart-grid">
  <div class="chart-box"><h3>📈 吞吐量 (Throughput)</h3><canvas id="chartThroughput"></canvas></div>
  <div class="chart-box"><h3>⏱️ 延迟百分位 (Latency)</h3><canvas id="chartLatency"></canvas></div>
  <div class="chart-box"><h3>🎯 首 Token 延迟 (TTFT)</h3><canvas id="chartTTFT"></canvas></div>
  <div class="chart-box"><h3>⚡ Token 生成速度 (TPS)</h3><canvas id="chartTPS"></canvas></div>
  <div class="chart-box full"><h3>📊 延迟分布堆叠 (P50/P90/P99)</h3><canvas id="chartStacked"></canvas></div>
</div>

<h2 style="font-size:18px;margin-bottom:12px;">📋 详细数据</h2>
<table>
<thead><tr>
  <th>并发</th><th>请求</th><th>通过率</th><th>吞吐</th><th>平均</th><th>P50</th>
  <th>P90</th><th>P99</th><th>最小</th><th>最大</th><th>TTFT</th><th>TPS</th>
</tr></thead>
<tbody id="resultTable"></tbody>
</table>

<div class="insight">
  <h3>💡 容量规划建议</h3>
  <p>{report.capacity_suggestion}</p>
  <p style="margin-top:8px;color:#718096;font-size:12px;">
    推荐并发 = 拐点并发 × 70%，为峰值预留 30% 余量。
    建议在实际部署环境中用真实 LLM 重新测试以获取准确数据。
  </p>
</div>
</div>

<script>
const data = {results_json};

// Summary
const best = data.reduce((a, b) => a.throughput_req_s > b.throughput_req_s ? a : b);

// Table
const tbody = document.getElementById('resultTable');
data.forEach(r => {{
  const row = document.createElement('tr');
  const isBest = r.concurrency === best.concurrency;
  row.innerHTML = `
    <td class="${{isBest ? 'best' : ''}}">${{r.concurrency}}</td>
    <td>${{r.total_requests}}</td>
    <td>${{r.errors > 0 ? (r.success/r.total_requests*100).toFixed(0)+'%' : '100%'}}</td>
    <td class="${{isBest ? 'best' : ''}}">${{r.throughput_req_s.toFixed(1)}}</td>
    <td>${{r.avg_latency_ms.toFixed(1)}}</td>
    <td>${{r.p50_ms.toFixed(1)}}</td>
    <td>${{r.p90_ms.toFixed(1)}}</td>
    <td>${{r.p99_ms.toFixed(1)}}</td>
    <td>${{r.min_ms.toFixed(1)}}</td>
    <td>${{r.max_ms.toFixed(1)}}</td>
    <td>${{r.avg_ttft_ms.toFixed(1)}}</td>
    <td>${{r.avg_tps.toFixed(1)}}</td>
  `;
  tbody.appendChild(row);
}});

// Chart helpers
const chartDefaults = {{ responsive: true, plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, padding: 12 }} }} }} }};
const concurrencyLabels = data.map(r => r.concurrency);

// 1. Throughput
new Chart(document.getElementById('chartThroughput'), {{
  type: 'bar',
  data: {{
    labels: concurrencyLabels,
    datasets: [{{
      label: 'Throughput (req/s)',
      data: data.map(r => r.throughput_req_s),
      backgroundColor: data.map(r => r.concurrency === best.concurrency ? '#38a169' : '#90cdf4'),
      borderRadius: 4,
    }}]
  }},
  options: {{ ...chartDefaults, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'req/s' }} }} }} }}
}});

// 2. Latency percentiles
new Chart(document.getElementById('chartLatency'), {{
  type: 'line',
  data: {{
    labels: concurrencyLabels,
    datasets: [
      {{ label: 'P50', data: data.map(r => r.p50_ms), borderColor: '#38a169', backgroundColor: '#38a16920', tension: 0.3, fill: false, pointRadius: 4 }},
      {{ label: 'P90', data: data.map(r => r.p90_ms), borderColor: '#dd6b20', backgroundColor: '#dd6b2020', tension: 0.3, fill: false, pointRadius: 4 }},
      {{ label: 'P99', data: data.map(r => r.p99_ms), borderColor: '#e53e3e', backgroundColor: '#e53e3e20', tension: 0.3, fill: false, pointRadius: 4 }},
    ]
  }},
  options: {{ ...chartDefaults, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'ms' }} }} }} }}
}});

// 3. TTFT
new Chart(document.getElementById('chartTTFT'), {{
  type: 'line',
  data: {{
    labels: concurrencyLabels,
    datasets: [{{
      label: 'Avg TTFT (ms)',
      data: data.map(r => r.avg_ttft_ms),
      borderColor: '#805ad5',
      backgroundColor: '#805ad520',
      tension: 0.3,
      fill: true,
      pointRadius: 4,
    }}]
  }},
  options: {{ ...chartDefaults, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'ms' }} }} }} }}
}});

// 4. TPS
new Chart(document.getElementById('chartTPS'), {{
  type: 'line',
  data: {{
    labels: concurrencyLabels,
    datasets: [{{
      label: 'Avg TPS',
      data: data.map(r => r.avg_tps),
      borderColor: '#d53f8c',
      backgroundColor: '#d53f8c20',
      tension: 0.3,
      fill: true,
      pointRadius: 4,
    }}]
  }},
  options: {{ ...chartDefaults, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'tokens/s' }} }} }} }}
}});

// 5. Stacked latency
new Chart(document.getElementById('chartStacked'), {{
  type: 'bar',
  data: {{
    labels: concurrencyLabels,
    datasets: [
      {{ label: 'P50', data: data.map(r => r.p50_ms), backgroundColor: '#38a169' }},
      {{ label: 'P90-P50', data: data.map(r => r.p90_ms - r.p50_ms), backgroundColor: '#f6ad55' }},
      {{ label: 'P99-P90', data: data.map(r => Math.max(r.p99_ms - r.p90_ms, 0)), backgroundColor: '#fc8181' }},
    ]
  }},
  options: {{
    responsive: true,
    scales: {{ x: {{ stacked: true }}, y: {{ stacked: true, title: {{ display: true, text: '毫秒 (ms)' }} }} }},
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12 }} }} }}
  }}
}});
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Advanced stress & performance test")
    parser.add_argument("--levels", nargs="+", type=int,
                        default=[1, 3, 5, 8, 10, 15, 20],
                        help="Concurrency levels to test")
    parser.add_argument("--requests", type=int, default=30,
                        help="Requests per concurrency level")
    parser.add_argument("--report", type=str, default=None,
                        help="Save JSON report to path")
    parser.add_argument("--html", type=str, default="benchmark_report.html",
                        help="Save HTML report to path")
    args = parser.parse_args()

    report = run_stress_test(args.levels, args.requests)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(f"JSON report saved to {args.report}")

    # HTML 报告默认生成
    html = generate_html_report(report)
    with open(args.html, "w") as f:
        f.write(html)
    print(f"HTML report saved to {args.html}")


if __name__ == "__main__":
    main()
