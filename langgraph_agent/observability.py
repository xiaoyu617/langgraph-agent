"""
可观测性配置 — LangSmith 集成。

LangSmith 是 LangChain 官方的可观测性平台，提供：
- 完整 Trace 链路（每次 LLM 调用的输入/输出）
- Token 用量统计
- 延迟分布（P50/P90/P99）
- 错误追踪

使用方法：
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=ls_xxxx
    export LANGCHAIN_PROJECT=langgraph-agent

    # 运行时会自动追踪所有 LangChain/LangGraph 调用
    python chat.py
"""
import os


def is_tracing_enabled() -> bool:
    """检查 LangSmith 追踪是否已启用。"""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")


def get_tracing_config() -> dict:
    """获取当前追踪配置。"""
    return {
        "tracing_enabled": is_tracing_enabled(),
        "project": os.environ.get("LANGCHAIN_PROJECT", "default"),
        "api_key_set": bool(os.environ.get("LANGCHAIN_API_KEY")),
        "endpoint": os.environ.get(
            "LANGCHAIN_ENDPOINT",
            "https://api.smith.langchain.com"
        ),
    }


def print_tracing_status():
    """打印追踪状态。"""
    config = get_tracing_config()
    if config["tracing_enabled"]:
        print(f"🔍 LangSmith 追踪: 已启用")
        print(f"   项目:     {config['project']}")
        print(f"   API Key:  {'✅ 已配置' if config['api_key_set'] else '❌ 未配置'}")
        print(f"   端点:     {config['endpoint']}")
        if not config["api_key_set"]:
            print(f"   ⚠️  需要设置 LANGCHAIN_API_KEY 才能上报数据")
    else:
        print(f"🔍 LangSmith 追踪: 未启用")
        print(f"   设置 LANGCHAIN_TRACING_V2=true 启用")
