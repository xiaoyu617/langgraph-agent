def replay_trace_text(trace: list) -> str:
    """
    Human-readable execution replay.
    """
    lines = []
    for i, step in enumerate(trace, 1):
        lines.append(f"[{i}] Node: {step['node']}")
        for k, v in step.items():
            if k != "node":
                lines.append(f"    {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def replay_trace_json(trace: list, conversation_id: str) -> dict:
    """
    JSON-based replay for frontend / auditing.
    """
    turns = []
    for i, step in enumerate(trace, 1):
        record = {"step": i, "node": step["node"]}
        for k, v in step.items():
            if k != "node":
                record[k] = v
        turns.append(record)

    return {
        "conversation_id": conversation_id,
        "turns": turns,
    }