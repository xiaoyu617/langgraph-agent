"""
测试执行器 — 运行测试用例并收集执行结果。

支持:
  - 单条用例执行
  - 批量执行
  - 结果聚合与报告
  - 与 CI/CD 集成
"""
import json
import time
from typing import Optional
from unittest.mock import patch


class TestExecutor:
    """测试执行器。"""

    def __init__(self, graph=None, use_mock: bool = True):
        """
        Args:
            graph: 编译后的 LangGraph (None 时自动构建)
            use_mock: 是否使用 Mock LLM
        """
        self.graph = graph
        self.use_mock = use_mock
        self._built = False

    def _ensure_graph(self):
        if self.graph is None and not self._built:
            from langgraph_agent.conditional_graph import build_graph
            from tests.mock_llm import MOCK_CHAT, MOCK_EMBEDDINGS
            if self.use_mock:
                patchers = [
                    patch("langgraph_agent.conditional_graph.create_chat_model",
                          return_value=MOCK_CHAT),
                    patch("langgraph_agent.rag_utils.DashScopeEmbeddings",
                          return_value=MOCK_EMBEDDINGS),
                ]
                for p in patchers:
                    p.start()
            self.graph = build_graph()
            self._built = True

    def run_single(self, dialog: list[dict], thread_id: str = "test") -> dict:
        """
        执行单条测试用例（可能包含多轮对话）。
        返回执行结果。
        """
        self._ensure_graph()
        trace = []
        errors = []
        start = time.perf_counter()

        for turn in dialog:
            try:
                result = self.graph.invoke(
                    {"input": turn["input"], "trace": trace},
                    config={"configurable": {"thread_id": thread_id}}
                )
                trace = result["trace"]
            except Exception as e:
                errors.append({"turn": turn["input"], "error": str(e)})
                break

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "thread_id": thread_id,
            "total_turns": len(dialog),
            "actual_path": [s["node"] for s in trace],
            "trace": trace,
            "output": result.get("output", "") if "result" in dir() else trace[-1].get("output", "") if trace else "",
            "latency_ms": round(elapsed_ms, 2),
            "errors": errors,
            "success": len(errors) == 0,
        }

    def run_batch(
        self,
        test_cases: list[dict],
        thread_prefix: str = "batch",
        verbose: bool = False,
    ) -> list[dict]:
        """批量执行测试用例。"""
        results = []
        for i, case in enumerate(test_cases):
            case_id = case["id"]
            thread_id = f"{thread_prefix}-{case_id}"

            if verbose:
                print(f"  [{i+1}/{len(test_cases)}] {case_id} ... ", end="", flush=True)

            result = self.run_single(case["dialog"], thread_id)

            expected = case.get("expected_path", [])
            actual = result["actual_path"]
            result["expected_path"] = expected
            result["path_match"] = (
                expected == actual[-len(expected):]
                if len(actual) >= len(expected)
                else False
            )
            result["case_id"] = case_id

            if verbose:
                status = "✅" if result["path_match"] else "❌"
                print(f"{status} path={'→'.join(actual[-3:])}")

            results.append(result)

        return results

    def report(self, results: list[dict]) -> dict:
        """生成执行报告。"""
        total = len(results)
        passed = sum(1 for r in results if r["path_match"])
        failed = total - passed
        total_latency = sum(r["latency_ms"] for r in results) if results else 0

        # 按路径汇总
        path_stats = {}
        for r in results:
            path = "→".join(r["actual_path"][-3:]) if r["actual_path"] else "none"
            if path not in path_stats:
                path_stats[path] = {"count": 0, "passed": 0, "failed": 0}
            path_stats[path]["count"] += 1
            path_stats[path]["passed" if r["path_match"] else "failed"] += 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 2) if total > 0 else 0,
            "avg_latency_ms": round(total_latency / total, 2) if total > 0 else 0,
            "path_stats": path_stats,
            "failed_cases": [
                {"id": r["case_id"], "expected": r["expected_path"],
                 "actual": r["actual_path"]}
                for r in results if not r["path_match"]
            ],
        }

    def print_report(self, report: dict):
        """打印可读的执行报告。"""
        print("=" * 55)
        print("       Test Execution Report")
        print("=" * 55)
        print(f"  Total:   {report['total']}")
        print(f"  Passed:  {report['passed']} ✅")
        print(f"  Failed:  {report['failed']} ❌")
        print(f"  Rate:    {report['pass_rate']:.0%}")
        print(f"  Avg Lat: {report['avg_latency_ms']:.1f} ms")
        print("-" * 55)
        print("  Path Statistics:")
        for path, stats in sorted(report["path_stats"].items()):
            bar = "✅" * stats["passed"] + "❌" * stats["failed"]
            print(f"    {path:30s} {bar}")
        if report["failed_cases"]:
            print("-" * 55)
            print("  Failed Cases:")
            for c in report["failed_cases"]:
                print(f"    [{c['id']}] expected={'→'.join(c['expected'])}")
                print(f"             actual=  {'→'.join(c['actual'])}")
        print("=" * 55)
