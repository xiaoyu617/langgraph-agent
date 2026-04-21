from typing import TypedDict, Optional


class AgentInput(TypedDict):
    input: str
    last_subject: Optional[str]


class AgentOutput(TypedDict):
    output: str