# LangGraph Agent Reference Architecture

**Single‑Agent → Multi‑Agent → Planner | Explicit State | RAG | Explainability**

---

## 📌 Overview

This repository provides a **production‑grade reference implementation of a LangGraph‑based Agent system**.

The project demonstrates how to design LLM agents as **explicit systems**, not prompt-driven black boxes, with:

- deterministic control flow
- explicit state modeling
- structured memory
- explainable execution
- full pytest validation

The system intentionally evolves across versions, each adding architectural capability while preserving correctness.

📦 **Current versions**:
- **v1.0** – Single‑Agent Architecture (tagged, stable)
- **v2.0** – Multi‑Agent Coordinator System (tagged)
- **v3.0** – Planner‑based Multi‑Agent System (feature branch)

---

## 🧭 Architecture Evolution

This project evolves in clearly separated stages:

### v1.0 – Single‑Agent

- One LangGraph agent
- Explicit state modeling
- Conditional routing (clarify / RAG / search)
- Structured memory slots
- Full execution trace & replay
- Deterministic pytest coverage

Focus: **correctness and observability**.

---

### v2.0 – Multi‑Agent Coordinator

- Central **Coordinator Agent**
- Functional agents with isolated responsibility:
  - Knowledge Agent (RAG)
  - Search Agent (tools)
- Memory‑aware semantic routing
- Agent dispatch recorded in trace

Focus: **modularity and scalability**.

📘 Detailed comparison:
`docs/LangGraph_Agent_v1_vs_v2_Architecture.md`

---

## 🧠 v3.0 – Planner Agent (Plan‑and‑Execute Architecture)

Starting from **v3.0**, the system transitions from *reactive routing* to an explicit
**plan‑and‑execute architecture**.

Instead of immediately dispatching an agent, the system first decides
whether the user request requires **multi‑step reasoning**.

---

### Motivation

Many realistic user requests are implicitly multi‑intent. For example:

> “介绍 LangChain 的作用，并给出官网地址”

This requires:
1. Conceptual understanding (knowledge)
2. Factual lookup (search)

A single-step routing strategy is insufficient.

---

### Planner Agent Responsibilities

The **Planner Agent** is responsible for:

- analyzing the user request
- decomposing it into ordered execution steps
- assigning each step to a specific agent type

The Planner **does not execute** tasks.
Execution is delegated to the Coordinator and functional agents.

---

### Plan‑and‑Execute Flow

```
User Input
   ↓
Router (semantic & memory-aware)
   ↓
Planner Agent
   ↓
Execution Plan
   ├─ Step 1 → Knowledge Agent
   ├─ Step 2 → Search Agent
   └─ ...
   ↓
Coordinator (sequential execution)
   ↓
Final Output
```

---

### Planner Output Contract

The Planner produces a structured plan:

```json
[
  { "step": 1, "agent": "knowledge", "instruction": "介绍 LangChain 的作用" },
  { "step": 2, "agent": "search", "instruction": "查找 LangChain 的官网地址" }
]
```

This enables deterministic execution, traceability, and system-level testing.

---

### Explainability & Testing (v3.0)

Planner behavior is fully recorded in execution trace, including:

- generated execution plan
- step order
- agent assignment
- execution results

Dedicated pytest cases validate:

- planner triggering conditions
- multi-step plan generation
- agent diversity in plans
- strict execution order
- aggregation of planner results

---

## 🧱 State Modeling

State is explicitly layered:

- **ConversationState** (persistent)
- **DerivedState** (per-turn decisions)
- **WorkingState** (ephemeral artifacts)

This separation prevents memory pollution and enables safe evolution.

---

## 🧪 Testing Strategy

Tests validate **system behavior**, not LLM phrasing:

- routing correctness
- memory influence
- agent dispatch
- planner execution order
- trace integrity

---

## ✅ Current Status

### v1.0 – Single‑Agent (Stable)

✔ Explicit state modeling
✔ Memory slots
✔ RAG & search
✔ Explainable trace & replay
✔ pytest coverage

_Tag: v1.0-single-agent_

---

### v2.0 – Multi‑Agent Coordinator

✔ Coordinator Agent
✔ Knowledge Agent
✔ Search Agent
✔ Memory‑aware routing
✔ Agent-level pytest

_Tag: v2.0-multi-agent_

---

### v3.0 – Planner‑Based Multi‑Agent

✔ Planner Agent
✔ Plan‑and‑execute workflow
✔ Multi-step agent orchestration
✔ Planner-level pytest

_Tag: v3.0-planner-agent_

Note:
v3.1 introduces the Critic / Verifier Agent as a quality control layer.
v3.2 further extends v3.1 with automatic replanning based on Critic feedback.

---

## 🚧 Roadmap

- ✅ v1.0 – Single‑Agent
- ✅ v2.0 – Multi‑Agent Coordinator
- ✅ v3.0 – Planner Agent
- ⏳ Critic / Verifier Agent
- ⏳ Re-planning & failure recovery
- ⏳ Agent‑to‑agent communication

---

## 🏁 Final Notes

This repository demonstrates how LLM agents should be designed as **evolving systems**, not prompt artifacts.

The progression from v1.0 → v2.0 → v3.0 mirrors real-world production agent architectures.
