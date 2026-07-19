"""
部署后验证脚本。
验证：包导入、Agent 图构建、所有路由路径可调用。
"""
import sys
import os

# 将项目根目录加入路径以便导入 tests.mock_llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def check_import():
    try:
        from langgraph_agent.conditional_graph import build_graph
        from langgraph_agent.trace_replay import replay_trace_text, replay_trace_json
        from langgraph_agent.evaluator import run_evaluation, EvalReport
        print("[PASS] 所有模块导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] 模块导入失败: {e}")
        return False


def check_graph_build():
    try:
        from langgraph_agent.conditional_graph import build_graph
        graph = build_graph()
        assert graph is not None
        print("[PASS] Agent 图构建成功")
        return True
    except Exception as e:
        print(f"[FAIL] Agent 图构建失败: {e}")
        return False


def check_graph_invoke():
    from langgraph_agent.conditional_graph import build_graph
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from langchain_core.embeddings import Embeddings
    from typing import List
    from unittest.mock import patch

    class MockChatModel(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            message = AIMessage(content="这是一个模拟回复，用于测试目的。")
            return ChatResult(generations=[ChatGeneration(message=message)])
        @property
        def _llm_type(self):
            return "mock"

    class MockEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * 128 for _ in texts]
        def embed_query(self, text: str) -> List[float]:
            return [0.1] * 128

    test_cases = [
        ("主体确认", {"input": "我指的是 LangChain", "trace": []}, "memory_update"),
        ("需澄清",   {"input": "它是什么", "trace": []}, "clarify"),
        ("RAG",      {"input": "LangChain是什么", "trace": []}, "rag_retrieve"),
        ("搜索",     {"input": "搜索 LangChain 官网", "trace": []}, "search"),
        ("直接回答", {"input": "你好", "trace": []}, "direct_answer"),
    ]

    with (
        patch("langgraph_agent.conditional_graph.create_chat_model", return_value=MockChatModel()),
        patch("langgraph_agent.rag_utils.DashScopeEmbeddings", return_value=MockEmbeddings()),
    ):
        graph = build_graph()
        all_ok = True
        for name, inp, expected_node in test_cases:
            try:
                result = graph.invoke(
                    inp,
                    config={"configurable": {"thread_id": f"deploy-{name}"}}
                )
                nodes = [s["node"] for s in result["trace"]]
                assert expected_node in nodes, \
                    f"路径 '{name}': 预期 {expected_node} 不在 {nodes}"
                print(f"  [PASS] {name} → {expected_node}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                all_ok = False

    return all_ok


def main():
    print("=" * 50)
    print("  部署后验证脚本")
    print("=" * 50)

    results = [
        ("模块导入", check_import()),
        ("图构建", check_graph_build()),
        ("路径调用", check_graph_invoke()),
    ]

    print("-" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  结果: {passed}/{total} 通过")

    if passed == total:
        print("  ✅ 部署验证全部通过")
        sys.exit(0)
    else:
        print("  ❌ 部署验证有失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
