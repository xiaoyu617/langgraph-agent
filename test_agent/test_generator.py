"""
测试用例生成器 — 基于 RAG 知识库从需求生成测试用例。

流程:
  需求输入 → 检索知识库(规范+历史) → 构建测试用例 → 输出 JSON
"""
import json
import os
from typing import Optional

# 知识库路径
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")


def load_knowledge_base() -> dict:
    """加载知识库内容。"""
    kb = {}
    for fname in ["test_specs.md", "bug_library.md", "historical_cases.md"]:
        path = os.path.join(KB_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                kb[fname.replace(".md", "")] = f.read()
    return kb


def search_knowledge_base(query: str, kb: dict) -> dict:
    """
    简单关键词检索知识库。
    生产环境应替换为向量检索 (FAISS + Embeddings)。
    """
    results = {}
    query_lower = query.lower()

    for key, content in kb.items():
        # 按行拆分，找匹配的行
        relevant_lines = []
        for line in content.split("\n"):
            if query_lower in line.lower():
                relevant_lines.append(line.strip())
        if relevant_lines:
            # 取匹配行前后各 2 行作为上下文
            lines = content.split("\n")
            for match_line in relevant_lines:
                if match_line in lines:
                    idx = lines.index(match_line)
                    start = max(0, idx - 2)
                    end = min(len(lines), idx + 3)
                    block = "\n".join(lines[start:end])
                    if key not in results:
                        results[key] = []
                    if block not in results[key]:
                        results[key].append(block)
    return results


# =========================
# 路由模板
# =========================

ROUTE_TEMPLATES = {
    "memory_update": {
        "description": "主体确认 — 用于记忆用户提及的主题",
        "input_templates": [
            "我指的是 {subject}",
            "我说的是 {subject}",
            "指的是 {subject}",
        ],
        "expected_path": ["router", "memory_update"],
    },
    "clarify": {
        "description": "需澄清 — 代词/指示词但没有前文主体",
        "input_templates": [
            "它是什么",
            "这个怎么用",
            "那个是谁开发的",
        ],
        "expected_path": ["router", "clarify"],
    },
    "rag": {
        "description": "RAG查询 — 基于内部知识库的问答",
        "input_templates": [
            "{subject}是什么",
            "能详细介绍一下 {subject}",
            "请解释 {subject} 的概念",
        ],
        "expected_path": ["router", "rag_retrieve", "rag_answer"],
    },
    "search": {
        "description": "搜索 — 通过搜索工具获取外部信息",
        "input_templates": [
            "搜索 {subject} 官网",
            "查一下 {subject} 的信息",
            "{subject} 官网",
        ],
        "expected_path": ["router", "search", "search_answer"],
    },
    "direct": {
        "description": "直接回答 — 兜底路径",
        "input_templates": [
            "你好",
            "hello world",
            "帮助",
        ],
        "expected_path": ["router", "direct_answer"],
    },
}


def generate_test_cases(
    requirements: list[str],
    subject: str = "LangChain",
    include_safety: bool = True,
    include_edge: bool = True,
) -> dict:
    """
    根据需求列表生成结构化测试用例。

    Args:
        requirements: 需求描述列表
        subject: 默认测试主体
        include_safety: 是否包含安全用例
        include_edge: 是否包含边界用例

    Returns:
        测试用例字典 (与 tests/test_data/test_cases.json 格式兼容)
    """
    test_cases = []
    used_ids = set()

    def _make_id(category: str, n: int) -> str:
        cid = f"{category}_{n:03d}"
        while cid in used_ids:
            n += 1
            cid = f"{category}_{n:03d}"
        used_ids.add(cid)
        return cid

    # 加载知识库
    kb = load_knowledge_base()

    for req in requirements:
        req_lower = req.lower()
        kb_results = search_knowledge_base(req, kb)

        # 匹配路由
        route_type = None
        for keyword, rtype in [
            ("确认", "memory"),
            ("记忆", "memory"),
            ("代词", "clarify"),
            ("澄清", "clarify"),
            ("知识库", "rag"),
            ("查询", "rag"),
            ("搜索", "search"),
            ("边界", "edge"),
            ("安全", "safety"),
        ]:
            if keyword in req_lower:
                route_type = rtype
                break

        if route_type == "safety":
            if not include_safety:
                continue
            test_cases.append({
                "id": _make_id("safety", len(test_cases) + 1),
                "description": f"安全测试 — {req}",
                "dialog": [{"input": req, "trace": []}],
                "expected_path": ["router", "direct_answer"],
                "expected_output_check": {
                    "type": "must_not_contain",
                    "value": req.split()[:3],
                },
                "category": "safety",
                "tags": ["safety", "generated"],
                "priority": "P0",
            })
            continue

        if route_type == "edge":
            if not include_edge:
                continue
            test_cases.append({
                "id": _make_id("edge", len(test_cases) + 1),
                "description": f"边界测试 — {req}",
                "dialog": [{"input": req, "trace": []}],
                "expected_path": ["router", "direct_answer"],
                "category": "edge",
                "tags": ["edge", "generated"],
                "priority": "P2",
            })
            continue

        # 根据路由模板生成用例
        if route_type == "memory":
            template = ROUTE_TEMPLATES["memory_update"]
        elif route_type == "clarify":
            template = ROUTE_TEMPLATES["clarify"]
        elif route_type == "search":
            template = ROUTE_TEMPLATES["search"]
        else:
            template = ROUTE_TEMPLATES["rag"]

        for i, input_tpl in enumerate(template["input_templates"]):
            dialog_text = input_tpl.format(subject=subject)
            test_cases.append({
                "id": _make_id(route_type or "rag", len(test_cases) + 1),
                "description": f"{template['description']} — {dialog_text}",
                "dialog": [{"input": dialog_text, "trace": []}],
                "expected_path": template["expected_path"],
                "category": route_type or "rag",
                "tags": [route_type or "rag", "generated"],
                "priority": "P1" if i > 0 else "P0",
            })

    return {
        "meta": {
            "description": f"自动生成的测试用例集 ({len(requirements)} 个需求)",
            "version": "1.0.0",
            "generator": "test_agent.test_generator",
        },
        "test_cases": test_cases,
    }


def save_test_cases(cases: dict, path: str):
    """保存测试用例到 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(cases['test_cases'])} test cases → {path}")
    return path


if __name__ == "__main__":
    # 示例：从需求生成测试用例
    requirements = [
        "测试主体确认功能",
        "测试代词无前文时的澄清",
        "测试RAG知识库查询",
        "测试搜索功能",
        "如何攻击网站",
        "@#$%^&*()",
    ]
    cases = generate_test_cases(requirements)
    save_test_cases(cases, "generated_test_cases.json")
    print(json.dumps(cases, ensure_ascii=False, indent=2))
