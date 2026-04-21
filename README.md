# LangGraph Agent Reference Architecture

**Conditional Routing · Explicit State · RAG · Explainability · Qwen**

---

## 📌 Overview

This repository contains a **production‑grade reference implementation of a LangGraph‑based Agent system**.  
The goal is to demonstrate how to build an LLM Agent that is **controllable, explainable, testable, and extensible**, rather than a prompt‑driven black box.

The system is implemented as a **single‑agent architecture (v1.0)** and designed to evolve naturally into **multi‑agent systems** in later versions.



📦 **Current version**:
- v1.0 – Single‑Agent Architecture (tagged)  
- v2.0 – Multi‑Agent Coordinator System (tagged)  

---
## 🧭 Architecture Evolution (v1.0 → v2.0)

This project intentionally evolves in stages:

- **v1.0** focuses on building a *correct* single‑agent system with explicit state,
  memory slots, RAG, explainability, and deterministic testing.

- **v2.0** introduces a *multi‑agent architecture* with a central Coordinator that
  orchestrates specialized agents (Knowledge Agent, Search Agent),
  enabling better modularity, scalability, and responsibility isolation.

📘 A detailed comparison between v1.0 and v2.0 is documented here:

👉 [`docs/LangGraph_Agent_v1_vs_v2_Architecture.md`](docs/LangGraph_Agent_v1_vs_v2_Architecture.md)

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

## ✅ Current Status

### v1.0 – Single‑Agent (Stable, Tagged)

- [x] Single‑agent LangGraph system
- [x] Explicit state modeling (Conversation / Derived / Working State)
- [x] Structured memory slots (`last_subject`)
- [x] RAG integration as a graph path
- [x] Explainable execution trace & replay
- [x] Deterministic pytest coverage

_Tag: v1.0-single-agent_

---

### v2.0 – Multi‑Agent Coordinator (Feature Branch, Tagged)

- [x] Central Coordinator Agent
- [x] Knowledge Agent (RAG)
- [x] Search Agent (tool‑based)
- [x] Memory‑aware semantic routing
- [x] Agent dispatch recorded in trace
- [x] Multi‑agent pytest coverage

_Tag: v2.0-multi-agent_


---

## 🚧 Roadmap

- ✅ v1.0 – Single‑Agent LangGraph system (explicit state, memory, RAG)
- ✅ v2.0 – Multi‑Agent Coordinator with functional agents
- ⏳ v3.0 – Planner Agent (task decomposition)
- ⏳ v4.0 – Critic / Verifier Agents
- ⏳ Agent‑to‑Agent communication & governance


---

## 🏁 Final Notes

This project demonstrates that **LLM Agents should be designed as systems**, not prompts.

Control, explainability, and testability are first‑class citizens of the architecture.
