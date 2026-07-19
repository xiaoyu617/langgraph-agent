"""
多模型切换支持。

支持多种 LLM 后端，通过配置文件或环境变量切换：
- Qwen (DashScope)
- OpenAI
- Claude (Anthropic)
- Mock（测试用）

使用方法：
    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=sk-xxx
    python run_agent.py
"""
import os
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


def get_llm_provider() -> str:
    """获取当前 LLM 提供商（从环境变量读取）。"""
    return os.environ.get("LLM_PROVIDER", "dashscope").lower()


def get_embedding_provider() -> str:
    """获取当前 Embedding 提供商。"""
    return os.environ.get("EMBEDDING_PROVIDER", "dashscope").lower()


def create_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0,
) -> BaseChatModel:
    """创建聊天模型实例，支持多种后端。"""
    provider = (provider or get_llm_provider()).lower()

    if provider == "mock":
        return _MockChatModel()

    if provider == "openai":
        model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=temperature)
        except ImportError:
            raise ImportError(
                "需要安装 langchain-openai: pip install langchain-openai"
            )

    if provider == "claude":
        model_name = model or os.environ.get("CLAUDE_MODEL", "claude-3-haiku-20240307")
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name, temperature=temperature)
        except ImportError:
            raise ImportError(
                "需要安装 langchain-anthropic: pip install langchain-anthropic"
            )

    # 默认：DashScope (Qwen)
    if provider == "dashscope":
        model_name = model or os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
        try:
            from langgraph_agent._chat_tongyi_fix import ChatTongyiWithTokenUsage as ChatTongyi
            return ChatTongyi(model=model_name, temperature=temperature)
        except ImportError:
            raise ImportError(
                "需要安装 dashscope: pip install dashscope"
            )

    raise ValueError(f"不支持的 LLM 提供商: {provider}，可选: dashscope, openai, claude, mock")


def create_embeddings(provider: Optional[str] = None):
    """创建 Embedding 模型实例。"""
    provider = (provider or get_embedding_provider()).lower()

    if provider == "mock":
        from langchain_core.embeddings import Embeddings
        from typing import List

        class _MockEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [[0.1] * 128 for _ in texts]
            def embed_query(self, text: str) -> List[float]:
                return [0.1] * 128
        return _MockEmbeddings()

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()
        except ImportError:
            raise ImportError("需要安装 langchain-openai")

    # 默认：DashScope
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings()
    except ImportError:
        raise ImportError("需要安装 dashscope")


class _MockChatModel(BaseChatModel):
    """Mock 聊天模型，用于测试。"""
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(
                content="这是一个模拟回复，用于测试目的。"
            ))]
        )
    @property
    def _llm_type(self):
        return "mock"


# 列出支持的提供商
SUPPORTED_PROVIDERS = {
    "dashscope": {
        "name": "通义千问 (DashScope)",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "models": ["qwen-plus", "qwen-max", "qwen-turbo"],
    },
    "openai": {
        "name": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    },
    "claude": {
        "name": "Anthropic Claude",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-haiku-20240307",
        "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    },
    "mock": {
        "name": "Mock (测试用)",
        "env_key": None,
        "default_model": "mock",
        "models": ["mock"],
    },
}


def print_supported_models():
    """打印支持的模型列表。"""
    print("=" * 50)
    print("  支持的 LLM 提供商")
    print("=" * 50)
    for key, info in SUPPORTED_PROVIDERS.items():
        print(f"\n  {info['name']} ({key})")
        if info["env_key"]:
            print(f"    环境变量: {info['env_key']}")
        print(f"    默认模型: {info['default_model']}")
        print(f"    可选模型: {', '.join(info['models'])}")
    print()
    print("  切换方式: export LLM_PROVIDER=openai")
    print("=" * 50)
