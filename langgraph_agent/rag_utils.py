"""
RAG 知识库工具。

支持两种模式：
  1. 文件模式：从 docs/ 目录加载 .txt 文件作为知识库
  2. 默认模式：内置信知识库（无 docs/ 目录时的降级）

两种模式下 Embedding 模型都通过 create_embeddings() 工厂方法创建，
支持 mock/dashscope/openai 切换，由环境变量控制。
"""
from .models import create_embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path
import os


# ========================================
# 默认知识库（文件加载失败时的降级）
# ========================================
_DEFAULT_KNOWLEDGE = [
    (
        "LangChain is an open-source framework designed to simplify "
        "the development of applications powered by large language models. "
        "It supports chains, agents, tools, memory, and LangGraph integration."
    ),
    (
        "RAG (Retrieval-Augmented Generation) combines document retrieval "
        "with LLM generation. It retrieves relevant documents from a knowledge "
        "base and uses them as context, reducing hallucinations."
    ),
]


# ========================================
# 文档加载
# ========================================

def _load_docs_from_directory(docs_dir: str) -> list[Document]:
    """从目录加载 .txt 文件作为知识库文档。"""
    docs = []
    doc_dir_path = Path(docs_dir)

    if not doc_dir_path.is_dir():
        return []

    for file_path in sorted(doc_dir_path.glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={"source": file_path.name},
                ))
        except Exception as e:
            print(f"[rag_utils] 跳过 {file_path.name}: {e}")

    return docs


def _get_default_docs() -> list[Document]:
    """获取默认知识库（docs/ 目录不存在或为空时使用）。"""
    return [
        Document(page_content=content)
        for content in _DEFAULT_KNOWLEDGE
    ]


# ========================================
# 构建向量库
# ========================================

def build_vectorstore():
    """
    构建本地向量知识库。

    搜索路径：
      1. 项目根目录下的 docs/ 文件夹（存在 .txt 文件时优先）
      2. 默认内置信知识库（降级方案）

    使用方式：
      在项目根目录创建 docs/ 文件夹，放入 .txt 文件即可扩展知识库。
    """
    # 优先从 docs/ 目录加载
    project_root = Path(__file__).resolve().parent.parent
    docs_dir = str(project_root / "docs")
    docs = _load_docs_from_directory(docs_dir)

    # 降级到默认知识库
    if not docs:
        docs = _get_default_docs()

    # 通过工厂方法创建 Embedding（支持 mock/dashscope/openai 切换）
    embeddings = create_embeddings()

    return FAISS.from_documents(docs, embeddings)
