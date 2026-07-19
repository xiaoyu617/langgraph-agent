# 缺陷库

## BUG-001: RAG 上下文为空时幻觉
- 路径: rag_answer
- 现象: 当 rag_context 为空时，LLM 仍会编造回答
- 根因: rag_answer_node 未处理空 context 情况
- 修复: 在 rag_answer_node 中检查 context 是否为空，空则转向 clarify

## BUG-002: 多轮对话后 topic 丢失
- 路径: memory_update → 任意路径
- 现象: 多轮对话后 last_subject 未被正确继承
- 根因: WorkingState 的 total=False 导致某些状态未返回
- 修复: 确保所有 node 都正确传递 last_subject

## BUG-003: 搜索结果的忠诚度
- 路径: search → search_answer
- 现象: search_answer 有时忽略 search_result 自行编造
- 根因: prompt 中 context 注入不够明确
- 修复: 优化 search_answer_node 的 prompt 模板
