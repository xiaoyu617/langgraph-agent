"""
LangGraph Agent 交互式 CLI Demo。

在终端中与 Agent 对话，实时查看路由决策和 Trace。

使用方法：
    python chat.py                          # 默认 Qwen
    LLM_PROVIDER=openai python chat.py      # 切换 OpenAI
    python chat.py --show-trace             # 显示详细路径
    python chat.py --judge                  # 每次回复后 LLM 评分
"""
import argparse
import os
import sys
from unittest.mock import patch

from langgraph_agent.conditional_graph import build_graph
from langgraph_agent.trace_replay import replay_trace_text
from langgraph_agent.models import print_supported_models


def main():
    parser = argparse.ArgumentParser(description="LangGraph Agent CLI Demo")
    parser.add_argument("--show-trace", action="store_true",
                        help="显示每次执行的详细 Trace")
    parser.add_argument("--judge", action="store_true",
                        help="每次回复后自动评分")
    parser.add_argument("--list-models", action="store_true",
                        help="列出支持的模型")
    parser.add_argument("--mock", action="store_true",
                        help="使用 Mock LLM（无需 API Key）")
    args = parser.parse_args()

    if args.list_models:
        print_supported_models()
        return

    provider = os.environ.get("LLM_PROVIDER", "dashscope")
    print(f"🤖 LLM 提供商: {provider}")

    # Mock 模式下注入 patch
    if args.mock:
        from langgraph_agent.mock_llm import MOCK_CHAT, MOCK_EMBEDDINGS
        patchers = [
            patch("langgraph_agent.conditional_graph.create_chat_model",
                  return_value=MOCK_CHAT),
            patch("langgraph_agent.rag_utils.DashScopeEmbeddings",
                  return_value=MOCK_EMBEDDINGS),
        ]
        for p in patchers:
            p.start()

    graph = build_graph()
    thread_id = "cli-session"
    trace = []

    print("=" * 50)
    print("  LangGraph Agent CLI  Demo")
    print("  输入 'exit' 退出，输入 'trace' 查看历史")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("👋 再见！")
            break
        if user_input.lower() == "trace":
            print(replay_trace_text(trace))
            continue
        if user_input.lower() == "models":
            print_supported_models()
            continue

        # 调用 Agent
        try:
            result = graph.invoke(
                {"input": user_input, "trace": trace},
                config={"configurable": {"thread_id": thread_id}}
            )
            trace = result["trace"]
        except Exception as e:
            print(f"❌ 错误: {e}")
            continue

        # 输出
        output = result.get("output", "（无输出）")
        print(f"\n🤖 Agent: {output}")

        # Trace
        if args.show_trace:
            print(f"\n📋 执行路径: {' → '.join(s['node'] for s in trace[-3:])}")

        # Judge
        if args.judge:
            try:
                from langgraph_agent.judge import LLMJudge, print_judge_report
                judge = LLMJudge()
                score = judge.evaluate(
                    input_text=user_input,
                    output=output,
                )
                print_judge_report([score])
            except Exception as e:
                print(f"  评分失败: {e}")


if __name__ == "__main__":
    main()
