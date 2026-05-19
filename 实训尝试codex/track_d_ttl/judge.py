from __future__ import annotations

import re
from typing import Any


UNCERTAIN_RE = re.compile(
    r"unknown|unable|cannot determine|not determine|not enough|无法|不能|未能|不确定|无法确定|无法判断",
    re.I,
)


def _is_empty_tool_content(content: Any) -> bool:
    text = str(content or "").strip()
    return text in {"", "[]", "{}"}


def heuristic_judge(task: dict[str, Any], pred: str, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    pred = str(pred or "").strip()
    if not pred:
        return {"pass": False, "confidence": 0.0, "failure_type": "no_answer", "rationale": "empty prediction"}
    if "<tool_call>" in pred or "<function=" in pred:
        return {"pass": False, "confidence": 0.1, "failure_type": "tool_call_leak", "rationale": "final answer contains a raw tool call"}
    if UNCERTAIN_RE.search(pred):
        return {"pass": False, "confidence": 0.2, "failure_type": "uncertain_answer", "rationale": "prediction says the answer is unknown or cannot be determined"}

    tools = [r for r in trajectory if r.get("role") == "tool"]
    tool_count = len(tools)
    if tool_count == 0 and len(str(task.get("instruction") or "")) > 80:
        return {"pass": False, "confidence": 0.35, "failure_type": "insufficient_evidence", "rationale": "complex task without tool evidence"}
    has_image = bool(task.get("image") or task.get("image_url") or task.get("image_b64") or task.get("image_path"))
    image_searches = [r for r in tools if r.get("fn_name") == "search_image"]
    browser_calls = [r for r in tools if str(r.get("fn_name", "")).startswith("browser")]
    if has_image and image_searches and all(_is_empty_tool_content(r.get("content")) for r in image_searches) and not browser_calls:
        return {
            "pass": False,
            "confidence": 0.3,
            "failure_type": "image_evidence_missing",
            "rationale": "image search returned no usable evidence and no browser verification was attempted",
        }
    if has_image and tool_count <= 1 and len(str(task.get("instruction") or "")) > 40:
        return {"pass": False, "confidence": 0.35, "failure_type": "insufficient_evidence", "rationale": "visual fact question used too little evidence"}
    if len(pred) > 180:
        return {"pass": False, "confidence": 0.45, "failure_type": "overbroad_answer", "rationale": "prediction is too verbose for a short-answer task"}
    return {"pass": True, "confidence": 0.65, "failure_type": "likely_ok", "rationale": "non-empty answer with available trajectory"}
