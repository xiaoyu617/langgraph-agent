from langchain_community.chat_models import ChatTongyi
import json


def run_critic_agent(user_input: str, plan, execution_results, final_output):
    """
    Decide whether the plan execution result is acceptable.
    """

    llm = ChatTongyi(model="qwen-plus", temperature=0)

    prompt = f"""
你是一个结果审查员（Critic）。
你的任务是判断系统的执行结果是否满足用户请求。

【用户请求】
{user_input}

【执行计划】
{plan}

【各步骤执行结果】
{execution_results}

【最终输出】
{final_output}

请判断：
1. 是否覆盖了用户的所有意图？
2. 是否存在明显遗漏或错误？
3. 输出是否自洽？

请仅输出 JSON：
{{
  "verdict": "pass" 或 "fail",
  "reason": "简要说明原因"
}}
"""

    response = llm.invoke(prompt)
    return json.loads(response.content)
