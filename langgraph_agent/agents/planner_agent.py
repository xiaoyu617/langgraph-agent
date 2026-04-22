from typing import List, Dict
from langchain_community.chat_models import ChatTongyi


def run_planner_agent(user_input: str) -> List:
    """
    Generate an execution plan.
    Each step specifies which agent should be used.
    """

    llm = ChatTongyi(model="qwen-plus", temperature=0)

    prompt = f"""
        你是一个任务规划器。
        请把用户请求拆解成执行步骤。
        
        可用 Agent:
        - knowledge: 内部知识 / RAG
        - search: 外部搜索
        
        输出 JSON 数组，每一步包含:
        - step: 步骤编号
        - agent: 使用的 agent 名称
        - instruction: 对 agent 的指令
        
        用户请求：
        {user_input}
        """

    response = llm.invoke(prompt)

    # ⚠️ 简化处理：假设 LLM 输出的是 JSON 字符串
    import json
    return json.loads(response.content)