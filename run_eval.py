"""
Evaluation runner for LangGraph Agent.

Usage:
    python run_eval.py                          # Run all test cases
    python run_eval.py --report report.json      # Save report to file
"""
import argparse
import json
import os

from langgraph_agent.conditional_graph import build_graph
from langgraph_agent.evaluator import run_evaluation, print_report, save_report


def load_test_cases(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


def main():
    parser = argparse.ArgumentParser(description="Run Agent evaluation")
    parser.add_argument(
        "--data", default="tests/test_data/test_cases.json",
        help="Path to test cases JSON"
    )
    parser.add_argument(
        "--report", default=None,
        help="Path to save evaluation report JSON"
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of times to repeat each test case"
    )
    args = parser.parse_args()

    print(f"Loading test cases from {args.data} ...")
    cases = load_test_cases(args.data)
    print(f"Loaded {len(cases)} test cases\n")

    print("Building graph ...")
    graph = build_graph()

    print("Running evaluation ...\n")
    report = run_evaluation(graph, cases, repeat=args.repeat)

    print_report(report)

    if args.report:
        save_report(report, args.report)


if __name__ == "__main__":
    main()
