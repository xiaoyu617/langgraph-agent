from langgraph_agent.agents.knowledge_agent import run_knowledge_agent
from langgraph_agent.agents.search_agent import run_search_agent


def execute_plan(plan, last_subject=None):
    """
    Execute each step in the plan sequentially.
    """
    results = []

    for step in plan:
        agent = step["agent"]
        instruction = step["instruction"]

        if agent == "knowledge":
            output = run_knowledge_agent(instruction, last_subject)
        elif agent == "search":
            output = run_search_agent(instruction)
        else:
            output = f"Unknown agent: {agent}"

        results.append({
            "step": step["step"],
            "agent": agent,
            "output": output
        })

    return results