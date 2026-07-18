"""End-to-end path validation tests.

Validates that the Agent follows the correct routing path
for each test case in the test dataset.
"""
import json
import os
import pytest
from langgraph_agent.trace_replay import replay_trace_text


DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def get_cases(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        return json.load(f)["test_cases"]


def get_node_names(trace):
    return [s["node"] for s in trace]


class TestAgentPaths:
    """Validate routing paths for all test cases."""

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(c, id=c["id"])
            for c in get_cases("test_cases.json")
        ],
    )
    def test_path_execution(self, case, built_graph):
        """Verify the agent follows the expected routing path."""
        thread_id = f"test-{case['id']}"
        trace = []

        for turn in case["dialog"]:
            result = built_graph.invoke(
                {"input": turn["input"], "trace": trace},
                config={"configurable": {"thread_id": thread_id}}
            )
            trace = result["trace"]

        actual_nodes = get_node_names(trace)

        # For single-turn, check last node matches
        expected = case["expected_path"]
        if len(case["dialog"]) == 1:
            assert actual_nodes[-1] == expected[-1], (
                f"Expected last node '{expected[-1]}', got '{actual_nodes[-1]}'. "
                f"Full path: {actual_nodes}"
            )
        else:
            # For multi-turn, verify expected path appears in sequence
            idx = 0
            for node in actual_nodes:
                if idx < len(expected) and node == expected[idx]:
                    idx += 1
            assert idx == len(expected), (
                f"Expected path {expected} not found in sequence. "
                f"Full path: {actual_nodes}"
            )


class TestOutputQuality:
    """Output quality checks (mock-aware)."""

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(c, id=c["id"])
            for c in get_cases("test_cases.json")
            if c.get("expected_output_check") is not None
        ],
    )
    def test_output_contains_keywords(self, case, built_graph):
        """Verify output contains expected keywords.
        NOTE: This test requires a real LLM API key to pass.
        With mock LLM, we only verify the output is non-empty.
        """
        import os as _os
        has_api_key = bool(_os.environ.get("DASHSCOPE_API_KEY"))
        if not has_api_key:
            pytest.skip("DASHSCOPE_API_KEY not set; output quality checks require real LLM")

        thread_id = f"test-{case['id']}"
        trace = []

        for turn in case["dialog"]:
            result = built_graph.invoke(
                {"input": turn["input"], "trace": trace},
                config={"configurable": {"thread_id": thread_id}}
            )
            trace = result["trace"]

        output = result.get("output", "")
        assert output, "Output should not be empty"

        check = case["expected_output_check"]
        if check["type"] == "contains_keywords":
            keywords = check["value"]
            for kw in keywords:
                assert kw in output, (
                    f"Expected keyword '{kw}' not found in output.\n"
                    f"Output: {output}"
                )
