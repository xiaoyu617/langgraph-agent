# Changelog

## v1.1.0 (2026-07-18)

### 🚀 新功能
- 多维度评测框架（路径准确性/输出质量/幻觉检测/安全合规）
- 结构化测试数据集（20+ 用例，覆盖 5 条路由路径 + 安全边界）
- 并发性能基准测试（自动扫描 + HTML 可视化报告）
- 完整 CI/CD 流水线（测试 → 构建 → 预发布 → E2E → 正式发布）

### 🧪 测试
- 22 个 pytest 测试用例，零外部依赖（全 mock）
- GitHub Actions 自动触发

## v1.0.0 (2026-07-15)

### 🎉 初始版本
- LangGraph 单 Agent 系统
- 条件路由（5 条路径）
- 结构化状态建模（Conversation/Derived/Working State）
- RAG 集成（FAISS + DashScope Embeddings）
- Tool calling（Google Search mock）
- Trace & Replay 可解释性
