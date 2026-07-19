"""Mock LLM and embedding providers for testing without API keys."""
from unittest.mock import MagicMock, patch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.embeddings import Embeddings
from typing import List


class MockChatModel(BaseChatModel):
    """Mock chat model that returns predictable responses."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content="这是一个模拟回复，用于测试目的。")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self):
        return "mock"


class MockEmbeddings(Embeddings):
    """Mock embeddings that return fixed vectors."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 128


MOCK_CHAT = MockChatModel()
MOCK_EMBEDDINGS = MockEmbeddings()


PATCH_TARGETS = [
    # Chat models
    patch(
        "langgraph_agent.conditional_graph.create_chat_model",
        return_value=MOCK_CHAT,
    ),
    # Embeddings
    patch(
        "langgraph_agent.rag_utils.DashScopeEmbeddings",
        return_value=MOCK_EMBEDDINGS,
    ),
]
