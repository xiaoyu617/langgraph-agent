from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.chat_models import ChatTongyi

from .tools import google_search
from .rag_utils import build_vectorstore


# =========================
# State Modeling
# =========================

class ConversationState(TypedDict):
    input: str
    last_subject: Optional[str]
    trace: List[dict]


class DerivedState(TypedDict):
    is_subject_confirm: bool
    need_clarify: bool
    need_rag: bool
    need_search: bool


class WorkingState(TypedDict, total=False):
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
    trace = state["trace"]
    input_text = state["input"]
    last_subject = state.get("last_subject")

    is_subject_confirm = any(
        p in input_text for p in ["我指的是", "我说的是", "指的是"]
    )

    has_pronoun = any(p in input_text for p in ["它", "这个", "那个", "这"])

    # --- 新增：能力/用途型问题 ---
    is_capability_question = any(
        p in input_text for p in ["能", "可以", "用途", "做什么", "有什么用"]
    )

    need_clarify = has_pronoun and not last_subject and not is_subject_confirm

    # ✅ 关键修复：memory + 能力型问题 → Knowledge Agent
    need_rag = (
        not need_clarify
        and not is_subject_confirm
        and (
            any(p in input_text for p in ["是什么", "介绍"])
            or (last_subject and is_capability_question)
        )
    )

    need_search = (
        not need_clarify
        and not is_subject_confirm
        and not need_rag
        and any(p in input_text for p in ["搜索", "官网", "查"])
    )

    trace.append({
        "node": "router",
        "input": input_text,
        "last_subject": last_subject,
        "decision": {
            "subject_confirm": is_subject_confirm,
            "clarify": need_clarify,
            "rag": need_rag,
            "search": need_search
        }
    })

    return {
        "input": input_text,
        "last_subject": last_subject,
        "trace": trace,
        "is_subject_confirm": is_subject_confirm,
        "need_clarify": need_clarify,
        "need_rag": need_rag,
        "need_search": need_search,
    }

def memory_update_node(state: GraphState) -> GraphState:
    trace = state["trace"]
    input_text = state["input"]

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
        "trace": trace
    }


def clarify_node(state: GraphState) -> GraphState:
    trace = state["trace"]
    llm = ChatTongyi(model="qwen-plus", temperature=0)

    resp = llm.invoke(f"请澄清用户的问题：{state['input']}")

    trace.append({
        "node": "clarify",
        "output": resp.content
    })

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "output": resp.content,
        "trace": trace
    }


def rag_retrieve_node(state: GraphState) -> GraphState:
    trace = state["trace"]
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
        "trace": trace
    }


def rag_answer_node(state: GraphState) -> GraphState:
    trace = state["trace"]
    llm = ChatTongyi(model="qwen-plus", temperature=0)

    resp = llm.invoke(
        f"{state['rag_context']}\n\n问题：{state['input']}"
    )

    trace.append({
        "node": "rag_answer",
        "output": resp.content
    })

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "output": resp.content,
        "trace": trace
    }


def search_node(state: GraphState) -> GraphState:
    trace = state["trace"]
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
        "trace": trace
    }


def search_answer_node(state: GraphState) -> GraphState:
    trace = state["trace"]
    llm = ChatTongyi(model="qwen-plus", temperature=0)

    resp = llm.invoke(
        f"{state['search_result']}\n\n问题：{state['input']}"
    )

    trace.append({
        "node": "search_answer",
        "output": resp.content
    })

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "output": resp.content,
        "trace": trace
    }


def direct_answer_node(state: GraphState) -> GraphState:
    trace = state["trace"]
    llm = ChatTongyi(model="qwen-plus", temperature=0)

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

    return {
        "input": state["input"],
        "last_subject": state.get("last_subject"),
        "output": resp.content,
        "trace": trace
    }


# =========================
# Build Graph
# =========================

def build_graph():
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

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda s:
            "memory_update" if s["is_subject_confirm"]
            else "clarify" if s["need_clarify"]
            else "rag_retrieve" if s["need_rag"]
            else "search" if s["need_search"]
            else "direct_answer"
    )

    graph.add_edge("memory_update", END)
    graph.add_edge("clarify", END)
    graph.add_edge("rag_retrieve", "rag_answer")
    graph.add_edge("rag_answer", END)
    graph.add_edge("search", "search_answer")
    graph.add_edge("search_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile(checkpointer=checkpointer)
