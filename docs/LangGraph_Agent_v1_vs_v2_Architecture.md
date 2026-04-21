# LangGraph Agent Architecture – Evolution Overview (v1.0 → v2.0)

This document is optimized for **technical interviews and architectural discussions**.  
It highlights the **evolution** of the system from a **Single‑Agent architecture (v1.0)** to a **Multi‑Agent system with a Coordinator (v2.0)**.

---

## 🔹 v1.0 – Single‑Agent Architecture

### Architecture Summary

```
User Input
   ↓
Router
   ├─ memory_update
   ├─ clarify
   ├─ rag_retrieve → rag_answer
   ├─ search → search_answer
   └─ direct_answer
```

### Core Characteristics

- ✅ Single LangGraph‑based agent
- ✅ Explicit state modeling (no implicit prompt memory)
- ✅ Deterministic routing controlled by the graph
- ✅ RAG and Search implemented as graph paths
- ✅ Structured memory slot (`last_subject`)
- ✅ Full execution trace and replay
- ✅ pytest‑validated behavior

### State Design

```python
ConversationState
DerivedState
WorkingState
```

All decisions are made **within one agent**, but behavior is fully observable and testable.

### Strengths of v1.0

- Predictable behavior
- Easy to reason about
- Strong foundation for production systems

### Limitations

- All capabilities implemented inside one agent
- Harder to scale responsibilities independently

---

## 🔹 v2.0 – Multi‑Agent Coordinator Architecture

### Architecture Summary

```
User Input
   ↓
Router (semantic & memory‑aware)
   ↓
Coordinator Agent
   ├─ Knowledge Agent (RAG)
   └─ Search Agent
```

### Core Characteristics

- ✅ Central **Coordinator Agent** controls execution
- ✅ Functional agents with isolated responsibilities
- ✅ Memory‑aware semantic routing
- ✅ Agent choice recorded in trace
- ✅ Same State model reused (no breaking changes)
- ✅ pytest coverage extended to Agent dispatch logic

### Agents in v2.0

| Agent | Responsibility |
|------|---------------|
| **Coordinator Agent** | Orchestration and decision control |
| **Knowledge Agent** | Internal knowledge & RAG |
| **Search Agent** | External information retrieval |

The Coordinator decides *which agent* to invoke — not the LLM.

### Key Improvement Over v1.0

| Dimension | v1.0 | v2.0 |
|--------|------|------|
| Control | Single Agent | Central Coordinator |
| Scalability | Limited | High |
| Responsibility Isolation | No | Yes |
| Multi‑Agent Testing | N/A | Yes |
| Architecture Maturity | Intermediate | Advanced |

---

## 🔍 Example: Memory‑Driven Agent Selection

**Input Flow**:

```
User: 我指的是 LangChain
User: 它能用来做什么？
```

### v1.0 Behavior

- Router identifies `need_rag`
- Same agent performs retrieval and answer

### v2.0 Behavior

- Router infers capability‑type question with memory context
- Coordinator dispatches to **Knowledge Agent**
- Dispatch decision recorded in trace

```json
{
  "node": "coordinator_dispatch",
  "agent": "knowledge_agent"
}
```

---

## 🧪 Testing Focus Comparison

### v1.0 Tests

- Routing correctness
- Memory slot updates
- RAG query rewriting

### v2.0 Tests

- Coordinator selects correct agent
- Memory affects agent choice
- Clarify path bypasses Coordinator
- Agent dispatch is traceable

---

## ✅ Why This Evolution Matters (Interview Angle)

v1.0 demonstrates the ability to **build a correct agent**.

v2.0 demonstrates the ability to **design an agent system**:

- Modularity
- Clear responsibilities
- Independent evolution of agents
- Production‑ready testing

This evolution mirrors **real‑world system scaling**, not tutorial examples.

---

## 🚀 Next Evolution (Planned)

- Planner Agent (Task Decomposition – v3.0)
- Critic / Verifier Agent
- Agent‑to‑Agent Communication
- Governance / Policy Enforcement

---

## 🏁 Summary

> v1.0 proves *correctness*.  
> v2.0 proves *architecture maturity*.

This project intentionally evolves step‑by‑step to reflect **industrial Agent system design**, not demo‑driven development.
