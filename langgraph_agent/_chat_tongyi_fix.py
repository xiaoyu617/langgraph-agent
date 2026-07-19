"""
ChatTongyi 补丁 — 让 token 用量正确上报到 LangSmith。

问题: langchain_community 的 ChatTongyi._generate() 返回的 llm_output
只有 model_name，没有 token_usage。LangSmith 从 llm_output.token_usage 读取，
所以 token 和 cost 显示为空。

修复: 重写 _generate，把 API 返回的 usage 信息透传到 llm_output。
"""
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_community.chat_models import ChatTongyi


class ChatTongyiWithTokenUsage(ChatTongyi):
    """
    增强版 ChatTongyi，将 token 用量上报到 llm_output，
    使 LangSmith 能正确显示 token 数和 cost。
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 调用原始的 _generate
        result = super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

        # 从 generation_info 中提取 token_usage，放到 llm_output
        token_usage = None
        for gen in result.generations:
            if isinstance(gen, ChatGeneration) and gen.generation_info:
                usage = gen.generation_info.get("token_usage")
                if usage:
                    token_usage = usage
                    break

        if token_usage:
            llm_output = result.llm_output or {}
            llm_output["token_usage"] = dict(token_usage)
            # 也加上模型名方便看
            llm_output["model_name"] = self.model_name
            return ChatResult(
                generations=result.generations,
                llm_output=llm_output,
            )

        return result
