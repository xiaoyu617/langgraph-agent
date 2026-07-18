"""Safety & compliance path tests."""
import json
import os
import pytest
from langgraph_agent.trace_replay import replay_trace_text


DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def load_cases(filename, category=None):
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        cases = json.load(f)["test_cases"]
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases


class TestSafety:
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(c, id=c["id"])
            for c in load_cases("safety_cases.json", "safety")
        ],
    )
    def test_safety_paths(self, case, built_graph):
        thread_id = f"test-{case['id']}"
        trace = []

        for turn in case["dialog"]:
            result = built_graph.invoke(
                {"input": turn["input"], "trace": trace},
                config={"configurable": {"thread_id": thread_id}}
            )
            trace = result["trace"]

        actual_nodes = [s["node"] for s in trace]
        expected = case["expected_path"]
        assert actual_nodes[-1] == expected[-1], (
            f"Expected last node '{expected[-1]}', got '{actual_nodes[-1]}'. "
            f"Path: {actual_nodes}"
        )

        # If must_not_contain check
        check = case.get("expected_output_check")
        if check and check["type"] == "must_not_contain":
            output = result.get("output", "")
            for forbidden in check["value"]:
                assert forbidden not in output, (
                    f"Output contains forbidden content '{forbidden}'. Output: {output}"
                )


class TestEdgeCases:
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(c, id=c["id"])
            for c in load_cases("safety_cases.json", "edge")
        ],
    )
    def test_edge_cases(self, case, built_graph):
        thread_id = f"test-{case['id']}"
        trace = []

        for turn in case["dialog"]:
            result = built_graph.invoke(
                {"input": turn["input"], "trace": trace},
                config={"configurable": {"thread_id": thread_id}}
            )
            trace = result["trace"]

        actual_nodes = [s["node"] for s in trace]
        expected = case["expected_path"]
        assert actual_nodes[-1] == expected[-1], (
            f"Expected last node '{expected[-1]}', got '{actual_nodes[-1]}'. "
            f"Path: {actual_nodes}"
        )


class TestMultiTurn:
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(c, id=c["id"])
            for c in load_cases("safety_cases.json", "multi_turn")
        ],
    )
    def test_multi_turn(self, case, built_graph):
        thread_id = f"test-{case['id']}"
        trace = []

        for turn in case["dialog"]:
            result = built_graph.invoke(
                {"input": turn["input"], "trace": trace},
                config={"configurable": {"thread_id": thread_id}}
            )
            trace = result["trace"]

        actual_nodes = [s["node"] for s in trace]
        expected = case["expected_path"]

        # For multi-turn, verify expected path nodes appear in order
        idx = 0
        for node in actual_nodes:
            if idx < len(expected) and node == expected[idx]:
                idx += 1
        assert idx == len(expected), (
            f"Expected path {expected} not found in sequence. "
            f"Full path: {actual_nodes}"
        )
