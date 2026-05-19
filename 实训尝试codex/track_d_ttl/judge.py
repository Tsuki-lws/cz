from __future__ import annotations

from typing import Any


def heuristic_judge(task: dict[str, Any], pred: str, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    if not pred.strip():
        return {"pass": False, "confidence": 0.0, "failure_type": "no_answer", "rationale": "empty prediction"}
    tool_count = sum(1 for r in trajectory if r.get("role") == "tool")
    if tool_count == 0 and len(str(task.get("instruction") or "")) > 80:
        return {"pass": False, "confidence": 0.35, "failure_type": "insufficient_evidence", "rationale": "complex task without tool evidence"}
    return {"pass": True, "confidence": 0.65, "failure_type": "likely_ok", "rationale": "non-empty answer with available trajectory"}

