from __future__ import annotations


def classify_failure(pred: str, tool_calls: int) -> str:
    if not pred.strip():
        return "no_answer"
    if tool_calls == 0:
        return "no_tool_evidence"
    return "unknown_or_ok"

