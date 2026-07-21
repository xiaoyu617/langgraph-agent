"""
LangGraph Agent 条件路由图。

核心架构：
  - 5 条路由路径：memory_update / clarify / RAG / search / direct_answer
  - 三层状态建模：ConversationState + DerivedState + WorkingState
  - MemorySaver 持久化 + conversation_history 结构记忆
  - memory_consolidation 节点：长对话压缩
"""
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .models import create_chat_model

from .tools import google_search
from .rag_utils import build_vectorstore


# =========================
# State Modeling
# =========================
# 三层状态设计：
#   ConversationState — 对话原始数据
#   DerivedState     — 路由推导结果
#   WorkingState     — 中间工作产物

class ConversationState(TypedDict):
    conversation_history: List[dict]  # 结构化对话历史
    input: str                         # 当前用户输入
    last_subject: Optional[str]        # 上一次主题（简化记忆）
    trace: List[dict]                  # 执行链路


class DerivedState(TypedDict):
    is_subject_confirm: bool
    need_clarify: bool
    need_rag: bool
    need_search: bool


class WorkingState(TypedDict, total=False):
    memory_summary: str       # 长对话压缩摘要
    rag_query: str
    rag_context: str
    search_result: str
    output: str


class GraphState(ConversationState, DerivedState, WorkingState):
    pass


# =========================
# Nodes
# =========================

def router_node(state: GraphState) -> GraphState:
    """路由节点 — 规则判断走哪条路径。"""
    trace = state["trace"]
    input_text = state["input"]
    last_subject = state.get("last_subject")
    conversation_history = state.get("conversation_history", [])

    # 主体确认
    is_subject_confirm = any(
        p in input_text for p in ["我指的是", "我说的是", "指的是"]
    )
    has_pronoun = any(p in input_text for p in ["它", "这个", "那个", "这"])

    # 利用 conversation_history 增强上下文判断
    # 如果历史里有主题，即使 last_subject 为空也能从历史推断
    has_history_context = bool(conversation_history) and not last_subject

    need_clarify = has_pronoun and not last_subject and not is_subject_confirm
    need_rag = (
        not need_clarify
        and not is_subject_confirm
        and any(p in input_text for p in ["是什么", "介绍", "解释", "概念", "什么是", "含义"])
    )
    need_search = (
        not need_clarify
        and not is_subject_confirm
        and not need_rag
        and any(p in input_text for p in ["搜索", "官网", "查"])
    )

    # 如果用户使用代词但 last_subject 为空但历史里有主题，走 direct_answer
    # 这样可以利用 MemorySaver 记住的历史上下文
    if need_clarify and has_history_context:
        need_clarify = False

    trace.append({
        "node": "router",
        "input": input_text,
        "last_subject": last_subject,
        "history_depth": len(conversation_history),
        "decision": {
            "subject_confirm": is_subject_confirm,
            "clarify": need_clarify,
            "rag": need_rag,
            "search": need_search,
        }
    })

    return {
        "input": input_text,
        "last_subject": last_subject,
        "trace": trace,
        "conversation_history": conversation_history,
        "is_subject_confirm": is_subject_confirm,
        "need_clarify": need_clarify,
        "need_rag": need_rag,
        "need_search": need_search,
    }


def memory_update_node(state: GraphState) -> GraphState:
    """记忆更新节点 — 用户指定主题时更新 last_subject。"""
    trace = state["trace"]
    input_text = state["input"]
    conversation_history = state.get("conversation_history", [])

    subject = input_text
    for p in ["我指的是", "我说的是", "指的是"]:
        if input_text.startswith(p):
            subject = input_text.replace(p, "").strip()
            break

    trace.append({
        "node": "memory_update",
        "last_subject": subject
    })

    return {
        "input": input_text,
        "last_subject": subject,
        "trace": trace,
        "conversation_history": conversation_history,
    }


def clarify_node(state: GraphState) -> GraphState:
    """澄清节点 — 用户指代不清时追问。"""
    trace = state["trace"]
    conversation_history = state.get("conversation_history", [])
    llm = create_chat_model()

    resp = llm.invoke(f"请澄清用户的问题：{state['input']}")

    trace.append({
        "node": "clarify",
        "output": resp.content
    })

    # 追加到对话历史
    updated_history = conversation_history + [
        {"role": "user", "content": state["input"]},
        {"role": "assistant", "content": resp.content},
    ]

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "output": resp.content,
        "trace": trace,
        "conversation_history": updated_history,
    }


def rag_retrieve_node(state: GraphState) -> GraphState:
    """RAG 检索节点 — 从向量知识库检索相关文档。"""
    trace = state["trace"]
    conversation_history = state.get("conversation_history", [])
    vectorstore = build_vectorstore()

    query = (
        f"{state['last_subject']} {state['input']}"
        if state.get("last_subject")
        else state["input"]
    )

    docs = vectorstore.similarity_search(query, k=1)
    context = docs[0].page_content if docs else ""

    trace.append({
        "node": "rag_retrieve",
        "rag_query": query,
        "context": context
    })

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "rag_query": query,
        "rag_context": context,
        "trace": trace,
        "conversation_history": conversation_history,
    }


def rag_answer_node(state: GraphState) -> GraphState:
    """RAG 回答节点 — 基于检索结果生成回答。"""
    trace = state["trace"]
    conversation_history = state.get("conversation_history", [])
    llm = create_chat_model()

    resp = llm.invoke(
        f"{state['rag_context']}\n\n问题：{state['input']}"
    )

    trace.append({
        "node": "rag_answer",
        "output": resp.content
    })

    # 追加到对话历史
    updated_history = conversation_history + [
        {"role": "user", "content": state["input"]},
        {"role": "assistant", "content": resp.content},
    ]

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "output": resp.content,
        "trace": trace,
        "conversation_history": updated_history,
    }


def search_node(state: GraphState) -> GraphState:
    """搜索节点 — 调用搜索引擎获取实时信息。"""
    trace = state["trace"]
    conversation_history = state.get("conversation_history", [])
    result = google_search.invoke(state["input"])

    trace.append({
        "node": "search",
        "query": state["input"],
        "search_result": result
    })

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "search_result": result,
        "trace": trace,
        "conversation_history": conversation_history,
    }


def search_answer_node(state: GraphState) -> GraphState:
    """搜索回答节点 — 基于搜索结果生成回答。"""
    trace = state["trace"]
    conversation_history = state.get("conversation_history", [])
    llm = create_chat_model()

    resp = llm.invoke(
        f"{state['search_result']}\n\n问题：{state['input']}"
    )

    trace.append({
        "node": "search_answer",
        "output": resp.content
    })

    # 追加到对话历史
    updated_history = conversation_history + [
        {"role": "user", "content": state["input"]},
        {"role": "assistant", "content": resp.content},
    ]

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "output": resp.content,
        "trace": trace,
        "conversation_history": updated_history,
    }


def direct_answer_node(state: GraphState) -> GraphState:
    """直接回答节点 — 不需要外部信息，直接回答。"""
    trace = state["trace"]
    conversation_history = state.get("conversation_history", [])
    llm = create_chat_model()

    query = (
        f"{state['last_subject']} 是什么？"
        if state.get("last_subject")
        else state["input"]
    )

    resp = llm.invoke(query)

    trace.append({
        "node": "direct_answer",
        "output": resp.content
    })

    # 追加到对话历史
    updated_history = conversation_history + [
        {"role": "user", "content": state["input"]},
        {"role": "assistant", "content": resp.content},
    ]

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "output": resp.content,
        "trace": trace,
        "conversation_history": updated_history,
    }


def memory_consolidation_node(state: GraphState) -> GraphState:
    """
    记忆合并节点 — 长对话时压缩旧历史。

    当 conversation_history 超过阈值时，
    将历史摘要为 memory_summary，清空历史保留摘要。
    """
    conversation_history = state.get("conversation_history", [])
    last_subject = state.get("last_subject")
    trace = state["trace"]

    # 阈值：超过 10 轮对话触发压缩
    if len(conversation_history) >= 10:
        # 保留最近 4 条，压缩之前的
        recent = conversation_history[-4:]
        old = conversation_history[:-4]

        # 压缩旧历史为摘要
        user_messages = [
            m["content"] for m in old if m["role"] == "user"
        ]
        summary = (
            f"历史对话摘要：用户询问了 {len(user_messages)} 个问题，"
            f"涉及主题包括：{last_subject or '多个'}。"
            f"完整历史可通过 trace 查看。"
        )

        trace.append({
            "node": "memory_consolidation",
            "compressed_turns": len(old) // 2,
            "summary": summary,
        })

        return {
            "input": state.get("input", ""),
            "last_subject": last_subject,
            "trace": trace,
            "conversation_history": recent,
            "memory_summary": summary,
        }

    # 未达阈值，原样传递
    return {
        "input": state.get("input", ""),
        "last_subject": last_subject,
        "trace": trace,
        "conversation_history": conversation_history,
    }


# =========================
# Build Graph
# =========================

def build_graph():
    """构建 LangGraph 状态图。"""
    graph = StateGraph(GraphState)
    checkpointer = MemorySaver()

    graph.add_node("router", router_node)
    graph.add_node("memory_update", memory_update_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("rag_answer", rag_answer_node)
    graph.add_node("search", search_node)
    graph.add_node("search_answer", search_answer_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("memory_consolidation", memory_consolidation_node)

    graph.set_entry_point("router")

    # 条件路由：router 根据 DerivedState 选择路径
    graph.add_conditional_edges(
        "router",
        lambda s:
            "memory_update" if s["is_subject_confirm"]
            else "clarify" if s["need_clarify"]
            else "rag_retrieve" if s["need_rag"]
            else "search" if s["need_search"]
            else "direct_answer"
    )

    # 固定边
    graph.add_edge("memory_update", END)
    graph.add_edge("clarify", "memory_consolidation")
    graph.add_edge("rag_retrieve", "rag_answer")
    graph.add_edge("rag_answer", "memory_consolidation")
    graph.add_edge("search", "search_answer")
    graph.add_edge("search_answer", "memory_consolidation")
    graph.add_edge("direct_answer", "memory_consolidation")
    graph.add_edge("memory_consolidation", END)

    return graph.compile(checkpointer=checkpointer)
