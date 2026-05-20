from __future__ import annotations

import re
from typing import Any

from .skills import classify_task, extract_constraints


UNCERTAIN_RE = re.compile(
    r"unknown|unable|cannot determine|not determine|not enough|无法|不能|未能|不确定|无法确定|无法判断",
    re.I,
)


def _is_empty_tool_content(content: Any) -> bool:
    text = str(content or "").strip()
    return text in {"", "[]", "{}"}


def heuristic_judge(task: dict[str, Any], pred: str, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    pred = str(pred or "").strip()
    task_type = classify_task(task)
    constraints = extract_constraints(task)
    if not pred:
        return {"pass": False, "confidence": 0.0, "failure_type": "no_answer", "rationale": "empty prediction", "task_type": task_type, "constraints": constraints}
    if "<tool_call>" in pred or "<function=" in pred:
        return {"pass": False, "confidence": 0.1, "failure_type": "tool_call_leak", "rationale": "final answer contains a raw tool call", "task_type": task_type, "constraints": constraints}
    if UNCERTAIN_RE.search(pred):
        return {"pass": False, "confidence": 0.2, "failure_type": "uncertain_answer", "rationale": "prediction says the answer is unknown or cannot be determined", "task_type": task_type, "constraints": constraints}

    tools = [r for r in trajectory if r.get("role") == "tool"]
    tool_count = len(tools)
    question = str(task.get("instruction") or "")
    question_lower = question.lower()
    search_queries = [
        str((r.get("fn_args") or {}).get("query") or (r.get("fn_args") or {}).get("search") or "").lower()
        for r in tools
        if r.get("fn_name") == "search_text"
    ]
    browser_errors = [
        r for r in tools
        if str(r.get("content") or "").lower().find("500") >= 0
        or str(r.get("content") or "").lower().find("403") >= 0
        or str(r.get("content") or "").lower().find("error") >= 0
    ]
    if tool_count == 0 and len(str(task.get("instruction") or "")) > 80:
        return {"pass": False, "confidence": 0.35, "failure_type": "insufficient_evidence", "rationale": "complex task without tool evidence", "task_type": task_type, "constraints": constraints}
    has_image = bool(task.get("image") or task.get("image_url") or task.get("image_b64") or task.get("image_path"))
    image_searches = [r for r in tools if r.get("fn_name") == "search_image"]
    browser_calls = [r for r in tools if str(r.get("fn_name", "")).startswith("browser")]
    if has_image and image_searches and all(_is_empty_tool_content(r.get("content")) for r in image_searches) and not browser_calls:
        return {
            "pass": False,
            "confidence": 0.3,
            "failure_type": "image_evidence_missing",
            "rationale": "image search returned no usable evidence and no browser verification was attempted",
            "task_type": task_type,
            "constraints": constraints,
        }
    if has_image and tool_count <= 1 and len(str(task.get("instruction") or "")) > 40:
        return {"pass": False, "confidence": 0.35, "failure_type": "insufficient_evidence", "rationale": "visual fact question used too little evidence", "task_type": task_type, "constraints": constraints}
    if has_image and task_type.startswith("visual") and len(search_queries) >= 4:
        visual_terms = ["logo", "jersey", "player", "company", "team", "actor", "ceo", "stock", "browser", "image", "shown"]
        visual_query_count = sum(1 for q in search_queries if any(token in q for token in visual_terms))
        generic_query_count = len(search_queries) - visual_query_count
        if generic_query_count >= 4 and visual_query_count == 0:
            return {"pass": False, "confidence": 0.42, "failure_type": "entity_drift", "rationale": "visual task repeatedly searched generic queries without preserving the pictured entity", "task_type": task_type, "constraints": constraints}
    if any(key in question_lower for key in ["first", "before", "after", "as of", "season", "shown"]) and search_queries:
        temporal_terms = ["first", "before", "after", "as of", "season", "shown"] + [c for c in constraints if c.isdigit()]
        if not any(any(term in q for term in temporal_terms) for q in search_queries):
            return {"pass": False, "confidence": 0.4, "failure_type": "temporal_drift", "rationale": "searches did not preserve the key temporal constraint", "task_type": task_type, "constraints": constraints}
    if browser_errors and len(browser_errors) >= max(2, tool_count // 2):
        return {"pass": False, "confidence": 0.35, "failure_type": "tool_failure_unrecovered", "rationale": "many tool calls failed and the trajectory lacks robust fallback evidence", "task_type": task_type, "constraints": constraints}
    if len(pred) > 180:
        return {"pass": False, "confidence": 0.45, "failure_type": "overbroad_answer", "rationale": "prediction is too verbose for a short-answer task", "task_type": task_type, "constraints": constraints}
    return {"pass": True, "confidence": 0.65, "failure_type": "likely_ok", "rationale": "non-empty answer with available trajectory", "task_type": task_type, "constraints": constraints}
