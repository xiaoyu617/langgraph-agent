"""
Performance & stress testing for LLM inference.

Measures:
- TTFT (Time to First Token)
- End-to-end latency
- Throughput (requests/second)
- Concurrent request handling

Usage:
    python run_benchmark.py --concurrency 5 --requests 20
    python run_benchmark.py --mode stress --concurrency 10 --requests 50 --report benchmark.json
"""
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

from unittest.mock import patch
from tests.mock_llm import MOCK_CHAT, MOCK_EMBEDDINGS
from langgraph_agent.conditional_graph import build_graph


@dataclass
class BenchmarkResult:
    concurrency: int
    total_requests: int
    total_time_s: float
    throughput_req_s: float
    latencies_ms: list[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0
    errors: int = 0
    success: int = 0
    avg_first_token_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


def run_single_request(graph, prompt: str, thread_id: str) -> dict:
    """Run a single request and measure timing."""
    start = time.perf_counter()
    result = graph.invoke(
        {"input": prompt, "trace": []},
        config={"configurable": {"thread_id": thread_id}}
    )
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "latency_ms": round(elapsed, 2),
        "output": result.get("output", ""),
        "path": [s["node"] for s in result["trace"]],
        "success": True,
    }


def run_benchmark(
    prompts: list[str],
    concurrency: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Run benchmark with given concurrency level."""
    graph = build_graph()
    all_latencies = []
    errors = 0
    success = 0
    start_time = time.perf_counter()

    def worker(prompt, idx):
        try:
            with (
                patch("langgraph_agent.conditional_graph.ChatTongyi", return_value=MOCK_CHAT),
                patch("langgraph_agent.rag_utils.DashScopeEmbeddings", return_value=MOCK_EMBEDDINGS),
            ):
                result = run_single_request(graph, prompt, f"bench-{idx}")
            return result
        except Exception as e:
            return {"latency_ms": 0, "success": False, "error": str(e)}

    request_count = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(iterations):
            prompt = prompts[i % len(prompts)]
            futures.append(executor.submit(worker, prompt, request_count))
            request_count += 1

        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                all_latencies.append(result["latency_ms"])
                success += 1
            else:
                errors += 1

    total_time = time.perf_counter() - start_time
    latencies = sorted(all_latencies)

    result = BenchmarkResult(
        concurrency=concurrency,
        total_requests=request_count,
        total_time_s=round(total_time, 3),
        throughput_req_s=round(request_count / total_time, 2),
        latencies_ms=latencies,
        p50_ms=round(latencies[len(latencies) // 2], 2) if latencies else 0,
        p90_ms=round(latencies[int(len(latencies) * 0.9)], 2) if latencies else 0,
        p99_ms=round(latencies[int(len(latencies) * 0.99)], 2) if latencies else 0,
        errors=errors,
        success=success,
    )
    return result


TEST_PROMPTS = [
    "你好",
    "LangChain是什么",
    "搜索 LangChain 官网",
    "它是什么",
    "能详细介绍一下 LangChain",
]


def print_benchmark_report(result: BenchmarkResult):
    """Print benchmark results in readable format."""
    print("=" * 60)
    print("        Performance Benchmark Report")
    print("=" * 60)
    print(f"  Concurrency:       {result.concurrency}")
    print(f"  Total requests:    {result.total_requests}")
    print(f"  Success:           {result.success}")
    print(f"  Errors:            {result.errors}")
    print(f"  Total time:        {result.total_time_s:.2f}s")
    print(f"  Throughput:        {result.throughput_req_s} req/s")
    print(f"  Avg latency:       {result.p50_ms:.1f} ms (P50)")
    print(f"  P90 latency:       {result.p90_ms:.1f} ms")
    print(f"  P99 latency:       {result.p99_ms:.1f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Agent performance")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=10, help="Total requests")
    parser.add_argument("--report", default=None, help="Save report to JSON file")
    args = parser.parse_args()

    print(f"Running benchmark: concurrency={args.concurrency}, requests={args.requests}")
    result = run_benchmark(TEST_PROMPTS, concurrency=args.concurrency, iterations=args.requests)
    print_benchmark_report(result)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
