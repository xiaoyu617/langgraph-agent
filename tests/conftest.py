"""Shared fixtures and mock setup for all tests."""
import json
import os
import pytest
from .mock_llm import PATCH_TARGETS

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def pytest_configure(config):
    """Apply mock patches globally for all tests."""
    for p in PATCH_TARGETS:
        p.start()


def pytest_unconfigure(config):
    """Clean up mock patches."""
    for p in PATCH_TARGETS:
        p.stop()


@pytest.fixture(scope="session")
def test_cases():
    path = os.path.join(DATA_DIR, "test_cases.json")
    with open(path) as f:
        return json.load(f)["test_cases"]


@pytest.fixture(scope="session")
def built_graph():
    from langgraph_agent.conditional_graph import build_graph
    return build_graph()
