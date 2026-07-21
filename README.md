# LangGraph Agent — 可观测性 & 可测试性 Demo

## 项目定位

一个基于 LangGraph 框架的智能体（Agent）参考实现，核心目标是**探索 Agent 系统的可测试性和可观测性方案**。适用于面试展示 Agent 开发能力和测试体系设计能力。

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.11 | 开发语言 |
| LangGraph | Agent 图编排框架（StateGraph、条件边、MemorySaver） |
| LangChain | LLM 调用、Tool 封装、RAG 向量库 |
| LangSmith | 可观测性 Trace 链路追踪 |
| FAISS | 向量检索（RAG 知识库） |
| Pytest + Mock | 单元测试与路径验证 |
| GitHub Actions | CI/CD 五阶段流水线 |

## 核心架构

```
                    User Input
                        │
                    ┌───┴───┐
                    │ Router│ ← 规则路由（确定可预测）
                    └───┬───┘
                        │
          ┌─────────────┼──────────────┐
          │             │              │
    ┌─────┴────┐  ┌────┴────┐  ┌──────┴──────┐
    │    RAG   │  │  Search │  │ Direct      │
    │ Retrieve │  │  Tool   │  │ Answer      │
    └────┬─────┘  └────┬────┘  └──────┬──────┘
         │             │              │
    ┌────┴────┐   ┌────┴────┐        │
    │ RAG     │   │ Search  │        │
    │ Answer  │   │ Answer  │        │
    └────┬────┘   └────┬────┘        │
         └──────┬──────┘             │
                └──────┬─────────────┘
                       │
                  ┌────┴────┐
                  │  Output │
                  └─────────┘
```

### 5 条路由路径

| 路径 | 触发条件 | 节点链 |
|------|---------|--------|
| 主体确认 | 输入含"我指的是/我说的是" | router → memory_update |
| 需澄清 | 有代词但无上下文 | router → clarify |
| RAG 查询 | 含"是什么/介绍/解释/概念" | router → rag_retrieve → rag_answer |
| 搜索 | 含"搜索/官网/查" | router → search → search_answer |
| 直接回答 | 其他 | router → direct_answer |

### 多 Agent 协调架构

```
User Input → Coordinator → test_generator → test_executor
                          → result_analyzer → report_generator
```

实现智能测试闭环：需求转换 → 用例生成 → 测试执行 → 结果分析 → 报告生成。

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. Mock 模式运行（不需要 API Key）
python chat.py --mock

# 4. 运行测试
python -m pytest tests/ -v
```

## 测试体系

- **22 个 pytest 测试用例**：覆盖 5 条路由路径、Edge Case、Safety、多轮对话
- **Mock 隔离**：MockChatModel + MockEmbeddings，测试不依赖外部 API
- **路径验证**：不验证 LLM 具体输出，验证 Agent 走了正确的决策路径
- **部署验证**：deploy_validate.py 验证 5 条路由全部正常

## CI/CD 流水线

```yaml
test → build → deploy-staging → e2e → deploy-production
```

五阶段质量门禁，GitHub Actions 实现。提交代码自动触发，分阶段验证。

## 项目结构

```
langgraph-agent/
├── chat.py                          # CLI 交互入口
├── server.py                        # HTTP 服务（K8s 部署用）
├── langgraph_agent/
│   ├── conditional_graph.py        # 核心：LangGraph 状态图定义
│   ├── models.py                   # 多模型切换工厂
│   ├── tools.py                    # Tool 封装（搜索引擎）
│   ├── rag_utils.py                # RAG 知识库构建
│   ├── multi_agent.py              # 多 Agent 协调架构
│   ├── evaluator_advanced.py       # 多维评测（语义/幻觉/性能）
│   ├── observability.py            # LangSmith Trace 配置
│   └── quality_monitor.py          # 质量监控
├── tests/
│   ├── mock_llm.py                 # 全局 Mock 配置
│   ├── conftest.py                 # pytest 共享夹具
│   ├── test_conditional_graph.py
│   ├── test_e2e_paths.py
│   └── test_safety_paths.py
├── scripts/
│   └── deploy_validate.py          # 部署后验证
├── .github/workflows/
│   └── ci-cd.yml                   # CI/CD 五阶段流水线
├── k8s/
│   ├── deployment.yaml             # K8s 部署配置
│   └── ci-cd-k8s-demo.yml          # K8s CI/CD 流水线（Demo）
├── Dockerfile                      # 容器化部署
└── requirements.txt
```
