"""
Auto sweep benchmark with visual HTML report.

Usage:
    python run_benchmark_sweep.py
    python run_benchmark_sweep.py --levels 1 2 3 5 8 10 15 20 --requests 50
    python run_benchmark_sweep.py --output my_report.html
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from unittest.mock import patch
from tests.mock_llm import MOCK_CHAT, MOCK_EMBEDDINGS


TEST_PROMPTS = [
    "你好",
    "LangChain是什么",
    "搜索 LangChain 官网",
    "它是什么",
    "能详细介绍一下 LangChain",
]


@dataclass
class SweepResult:
    concurrency: int
    total_requests: int
    throughput_req_s: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    success: int
    errors: int
    latencies_ms: list


def run_single_request(graph, prompt: str, thread_id: str) -> dict:
    start = time.perf_counter()
    result = graph.invoke(
        {"input": prompt, "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )
    elapsed = (time.perf_counter() - start) * 1000
    return {
        "latency_ms": round(elapsed, 2),
        "success": True,
    }


def run_benchmark_level(concurrency: int, iterations: int) -> SweepResult:
    from langgraph_agent.conditional_graph import build_graph
    graph = build_graph()
    latencies = []
    errors = 0
    success = 0
    start_time = time.perf_counter()

    def worker(prompt, idx):
        try:
            with (
                patch("langgraph_agent.conditional_graph.ChatTongyi", return_value=MOCK_CHAT),
                patch("langgraph_agent.rag_utils.DashScopeEmbeddings", return_value=MOCK_EMBEDDINGS),
            ):
                return run_single_request(graph, prompt, f"bench-{idx}")
        except Exception as e:
            return {"latency_ms": 0, "success": False, "error": str(e)}

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for i in range(iterations):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            futures.append(executor.submit(worker, prompt, i))

        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                latencies.append(result["latency_ms"])
                success += 1
            else:
                errors += 1

    total_time = time.perf_counter() - start_time
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    return SweepResult(
        concurrency=concurrency,
        total_requests=iterations,
        throughput_req_s=round(iterations / total_time, 2) if total_time > 0 else 0,
        p50_ms=round(latencies_sorted[n // 2], 2) if n > 0 else 0,
        p90_ms=round(latencies_sorted[int(n * 0.9)], 2) if n > 0 else 0,
        p99_ms=round(latencies_sorted[int(n * 0.99)], 2) if n > 0 else 0,
        avg_ms=round(sum(latencies) / n, 2) if n > 0 else 0,
        min_ms=round(latencies_sorted[0], 2) if n > 0 else 0,
        max_ms=round(latencies_sorted[-1], 2) if n > 0 else 0,
        success=success,
        errors=errors,
        latencies_ms=latencies_sorted,
    )


def generate_html_report(results: list[SweepResult]) -> str:
    """Generate a self-contained HTML report with Chart.js."""
    results_json = json.dumps([asdict(r) for r in results])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent 性能压测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #1a1a2e; padding: 40px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; }}
  .subtitle {{ color: #666; margin-bottom: 32px; }}
  .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .card .label {{ font-size: 13px; color: #888; margin-bottom: 4px; }}
  .card .value {{ font-size: 24px; font-weight: 700; }}
  .card .value.green {{ color: #059669; }}
  .card .value.orange {{ color: #d97706; }}
  .card .value.blue {{ color: #2563eb; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 32px; }}
  th {{ background: #f8fafc; padding: 12px 16px; text-align: center; font-size: 13px; color: #64748b; border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: 12px 16px; text-align: center; border-bottom: 1px solid #f1f5f9; font-size: 14px; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}
  .best {{ font-weight: 700; color: #059669; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
  .chart-box {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .chart-box h3 {{ font-size: 15px; margin-bottom: 16px; color: #475569; }}
  .chart-box.full {{ grid-column: 1 / -1; }}
</style>
</head>
<body>
<div class="container">
  <h1>🚀 Agent 性能压测报告</h1>
  <p class="subtitle">测试环境: macOS, Python 3.11, Mock LLM ｜ 测试时间: 2026-07-18</p>

  <div class="summary-cards" id="summaryCards">
    <div class="card"><div class="label">总请求数</div><div class="value blue" id="totalReqs">-</div></div>
    <div class="card"><div class="label">成功率</div><div class="value green" id="successRate">-</div></div>
    <div class="card"><div class="label">最大吞吐量</div><div class="value green" id="maxThroughput">-</div></div>
    <div class="card"><div class="label">最优并发度</div><div class="value orange" id="bestConcurrency">-</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>并发度</th>
        <th>请求数</th>
        <th>成功率</th>
        <th>Throughput</th>
        <th>P50 (ms)</th>
        <th>P90 (ms)</th>
        <th>P99 (ms)</th>
        <th>平均 (ms)</th>
        <th>最小 (ms)</th>
        <th>最大 (ms)</th>
      </tr>
    </thead>
    <tbody id="resultTable"></tbody>
  </table>

  <div class="chart-grid">
    <div class="chart-box"><h3>📊 吞吐量 vs 并发度</h3><canvas id="chartThroughput"></canvas></div>
    <div class="chart-box"><h3>⏱️ 延迟百分位 vs 并发度</h3><canvas id="chartLatency"></canvas></div>
    <div class="chart-box full"><h3>📈 延迟分布 (P50/P90/P99 堆叠)</h3><canvas id="chartStacked"></canvas></div>
  </div>
</div>

<script>
const data = {results_json};

// Summary
const total = data.reduce((s, r) => s + r.total_requests, 0);
const totalOk = data.reduce((s, r) => s + r.success, 0);
const totalErr = data.reduce((s, r) => s + r.errors, 0);
const best = data.reduce((a, b) => a.throughput_req_s > b.throughput_req_s ? a : b);

document.getElementById('totalReqs').textContent = total;
document.getElementById('successRate').textContent = (totalOk / total * 100).toFixed(1) + '%';
document.getElementById('maxThroughput').textContent = best.throughput_req_s.toFixed(1) + ' req/s';
document.getElementById('bestConcurrency').textContent = best.concurrency;

// Table
const tbody = document.getElementById('resultTable');
data.forEach(r => {{
  const row = document.createElement('tr');
  const isBest = r.concurrency === best.concurrency;
  row.innerHTML = `
    <td class="${{isBest ? 'best' : ''}}">${{r.concurrency}}</td>
    <td>${{r.total_requests}}</td>
    <td>${{(r.success/r.total_requests*100).toFixed(0)}}%</td>
    <td class="${{isBest ? 'best' : ''}}">${{r.throughput_req_s.toFixed(1)}}</td>
    <td>${{r.p50_ms.toFixed(1)}}</td>
    <td>${{r.p90_ms.toFixed(1)}}</td>
    <td>${{r.p99_ms.toFixed(1)}}</td>
    <td>${{r.avg_ms.toFixed(1)}}</td>
    <td>${{r.min_ms.toFixed(1)}}</td>
    <td>${{r.max_ms.toFixed(1)}}</td>
  `;
  tbody.appendChild(row);
}});

// Chart: Throughput
new Chart(document.getElementById('chartThroughput'), {{
  type: 'bar',
  data: {{
    labels: data.map(r => r.concurrency),
    datasets: [{{
      label: 'Throughput (req/s)',
      data: data.map(r => r.throughput_req_s),
      backgroundColor: data.map(r => r.concurrency === best.concurrency ? '#059669' : '#93c5fd'),
      borderRadius: 6,
    }}]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
}});

// Chart: Latency percentiles
new Chart(document.getElementById('chartLatency'), {{
  type: 'line',
  data: {{
    labels: data.map(r => r.concurrency),
    datasets: [
      {{ label: 'P50', data: data.map(r => r.p50_ms), borderColor: '#059669', tension: 0.3, fill: false }},
      {{ label: 'P90', data: data.map(r => r.p90_ms), borderColor: '#d97706', tension: 0.3, fill: false }},
      {{ label: 'P99', data: data.map(r => r.p99_ms), borderColor: '#dc2626', tension: 0.3, fill: false }},
    ]
  }},
  options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
}});

// Chart: Stacked latencies
new Chart(document.getElementById('chartStacked'), {{
  type: 'bar',
  data: {{
    labels: data.map(r => r.concurrency),
    datasets: [
      {{ label: 'P50', data: data.map(r => r.p50_ms), backgroundColor: '#059669' }},
      {{ label: 'P90-P50', data: data.map(r => r.p90_ms - r.p50_ms), backgroundColor: '#fbbf24' }},
      {{ label: 'P99-P90', data: data.map(r => r.p99_ms - r.p90_ms), backgroundColor: '#fca5a5' }},
    ]
  }},
  options: {{
    responsive: true,
    scales: {{ x: {{ stacked: true }}, y: {{ stacked: true, title: {{ display: true, text: '毫秒 (ms)' }} }} }},
    plugins: {{ legend: {{ position: 'bottom' }} }}
  }}
}});
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Auto-sweep benchmark with HTML report")
    parser.add_argument("--levels", nargs="+", type=int,
                        default=[1, 2, 3, 5, 8, 10, 15, 20],
                        help="Concurrency levels to test")
    parser.add_argument("--requests", type=int, default=30,
                        help="Requests per level")
    parser.add_argument("--output", default="benchmark_report.html",
                        help="Output HTML report path")
    args = parser.parse_args()

    print("=" * 55)
    print("  Agent 并发压测 — 自动扫描")
    print("=" * 55)
    print(f"  并发级别: {args.levels}")
    print(f"  每级请求: {args.requests}")
    print(f"  总请求数: {len(args.levels) * args.requests}")
    print("-" * 55)

    results = []
    for i, c in enumerate(args.levels, 1):
        print(f"  [{i}/{len(args.levels)}] 并发度={c} ... ", end="", flush=True)
        result = run_benchmark_level(c, args.requests)
        print(f"通过率={result.success}/{result.total_requests} "
              f"吞吐={result.throughput_req_s} req/s "
              f"P50={result.p50_ms}ms P99={result.p99_ms}ms")
        results.append(result)

    # Save raw data
    raw_path = args.output.replace(".html", "_raw.json")
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # Generate HTML report
    html = generate_html_report(results)
    with open(args.output, "w") as f:
        f.write(html)

    print("-" * 55)
    print(f"  ✅ 报告已生成:")
    print(f"     📄 HTML:  {args.output}")
    print(f"     📊 数据:  {raw_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
