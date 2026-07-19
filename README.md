# LangGraph Agent Reference Architecture

**Conditional Routing · Explicit State · RAG · Explainability · Qwen**

---

## 📌 Overview

This repository contains a **production‑grade reference implementation of a LangGraph‑based Agent system**.  
The goal is to demonstrate how to build an LLM Agent that is **controllable, explainable, testable, and extensible**, rather than a prompt‑driven black box.

The system is implemented as a **single‑agent architecture (v1.0)** and designed to evolve naturally into **multi‑agent systems** in later versions.

---

## 🎯 Design Goals

The project intentionally avoids common anti‑patterns seen in many LLM agent demos:

- ❌ Prompt‑history as memory
- ❌ LLM‑decides‑everything control flow
- ❌ RAG hidden inside prompts
- ❌ No trace, no explainability, no tests

Instead, it focuses on:

- ✅ **Explicit state modeling**
- ✅ **Graph‑controlled decision flow**
- ✅ **Memory as structured slots**
- ✅ **RAG as a first‑class path in the graph**
- ✅ **Node‑level trace & replay**
- ✅ **Deterministic pytest validation**

---

## 🧱 Architecture Summary

```
User Input
   ↓
Router (decision node)
   ├─ memory_update   (confirm subject)
   ├─ clarify         (resolve ambiguity)
   ├─ rag_retrieve    (internal knowledge)
   │      ↓
   │   rag_answer
   ├─ search          (external tool)
   │      ↓
   │   search_answer
   └─ direct_answer
```

Control flow is **system‑driven**, not model‑driven.

---

## 🧠 State Modeling (Industrial Pattern)

The Agent state is explicitly split into three conceptual layers.

### 1️⃣ ConversationState (Long‑lived)

```python
class ConversationState(TypedDict):
    input: str
    last_subject: Optional[str]
    trace: List[dict]
```

- `input`: current user turn
- `last_subject`: structured memory slot
- `trace`: execution facts for explainability

This layer is persisted across turns via LangGraph **MemorySaver**.

---

### 2️⃣ DerivedState (Per‑turn decisions)

```python
class DerivedState(TypedDict):
    is_subject_confirm: bool
    need_clarify: bool
    need_rag: bool
    need_search: bool
```

- Computed by the router each turn
- Never stored as long‑term memory

---

### 3️⃣ WorkingState (Intermediate artifacts)

```python
class WorkingState(TypedDict, total=False):
    rag_query: str
    rag_context: str
    search_result: str
    output: str
```

- Exists only on relevant paths
- Keeps intermediate data logically isolated

---

### ✅ Final GraphState

```python
class GraphState(ConversationState, DerivedState, WorkingState):
    pass
```

This layered modeling is a key difference between **demo agents** and **production agents**.

---

## 🔀 Routing Logic

The router enforces a strict priority order:

1. **Subject Confirmation** → `memory_update`
2. **Ambiguity Detected** → `clarify`
3. **Conceptual Question** → `rag_retrieve → rag_answer`
4. **Lookup / Search Request** → `search → search_answer`
5. **Fallback** → `direct_answer`

Routing is implemented with **LangGraph conditional edges**, not LLM reasoning.

---

## 🗄 Explicit Memory Design

Memory is represented as a **structured slot** (`last_subject`):

```text
User: 我指的是 LangChain
→ memory_update(last_subject="LangChain")
```

Benefits:

- Deterministic coreference resolution
- Safe cross‑turn reuse
- Works with RAG query rewrite
- Fully testable

---

## 📚 RAG as a Graph Path

RAG is implemented as a **dedicated graph branch**, not embedded in prompts.

Flow:

```
router → rag_retrieve → rag_answer
```

Key features:

- Vector store built with DashScope Embeddings + FAISS
- Query rewrite using structured memory
- Context injection isolated from control logic

---

## 🔍 Trace & Explainability

### Trace

Trace captures **what actually happened**, not model thoughts:

```json
{
  "node": "router",
  "input": "能详细介绍一下",
  "last_subject": "LangChain",
  "decision": { "rag": true }
}
```

### Replay

Replay reconstructs execution for humans and systems:

```
[1] router → rag
[2] rag_retrieve → query rewrite
[3] rag_answer → final response
```

Both text and JSON replay are supported.

---

## 🧪 Testing Strategy

Tests verify **behavior**, not raw text:

- Routing correctness
- Memory usage
- RAG query rewriting
- Trace structure integrity

Example:

```python
assert "LangChain" in rag_step["rag_query"]
```

This ensures the system is **regression‑safe**.

---

## 🧠 Technology Stack

- LangGraph
- LangChain Core
- Qwen / ChatTongyi
- DashScope Embeddings
- FAISS
- Python
- pytest

---

## ✅ Current Status (v1.0)

- [x] Single‑agent LangGraph system
- [x] Explicit state modeling
- [x] Structured memory slots
- [x] RAG integration
- [x] Explainable trace & replay
- [x] pytest coverage

---

## 🚧 Roadmap

- [ ] Multi‑Agent Coordinator (Planner + Functional Agents)
- [ ] Agent‑to‑Agent communication
- [ ] Verification / Critic Agents
- [ ] Governance & policy enforcement

---

## 🏁 Final Notes

This project demonstrates that **LLM Agents should be designed as systems**, not prompts.

Control, explainability, and testability are first‑class citizens of the architecture.

---

## 📊 测试与评测体系 (v1.1)

### 测试数据集

结构化 JSON 测试用例集，覆盖所有路由路径：

| 分类 | 路径 | 用例数 |
|------|------|--------|
| Memory | `memory_update` | 3 |
| Clarify | `clarify` | 3 |
| RAG | `rag_retrieve → rag_answer` | 3 |
| Search | `search → search_answer` | 3 |
| Direct | `direct_answer` | 2 |
| Safety | 安全合规 | 2 |
| Edge | 边界条件 | 3 |
| Multi-turn | 多轮对话 | 1 |

### 评测维度

`langgraph_agent/evaluator.py` 提供多维度评测：

- **Path Accuracy**: 路由路径是否与预期一致
- **Accuracy**: 输出是否包含预期关键词 / 事实
- **Hallucination**: 不确定表达检测
- **Safety**: 有害内容检测
- **Latency**: 端到端延迟

### 运行方式

```bash
# 路径验证测试
pytest tests/test_e2e_paths.py -v

# 完整评测
python run_eval.py --report eval-report.json

# 性能基准测试
python run_benchmark.py --concurrency 5 --requests 20
```

### CI/CD

GitHub Actions 流水线自动在 PR 时执行评测，产出评测报告 artifact。

---

## 📈 质量指标概览

| 指标 | 描述 | 方法 |
|------|------|------|
| 路径准确率 | Agent 走对了路由 | Trace 节点比对 |
| 输出准确性 | 输出包含预期事实 | 关键词/语义匹配 |
| 幻觉率 | 输出中不确定/虚构内容 | 不确定性短语检测 |
| 安全合规率 | 输出未包含有害内容 | 有害模式匹配 |
| 响应延迟 | E2E 延迟 P50/P90/P99 | 计时统计 |
| 吞吐量 | 每秒处理请求数 | 并发测试 |
# CI/CD pipeline trigger test
