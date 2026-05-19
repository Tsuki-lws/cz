from __future__ import annotations

from typing import Any


def classify_task(task: dict[str, Any]) -> str:
    text = str(task.get("instruction") or task.get("question") or "")
    has_image = bool(task.get("image") or task.get("image_url") or task.get("image_b64") or task.get("image_path"))
    if has_image:
        return "visual"
    if len(text) > 120 or any(token in text.lower() for token in ("both", "after", "before", "which of", "compare", "multi")):
        return "hard"
    return "simple"


def max_steps_for(task: dict[str, Any], default: int) -> int:
    kind = classify_task(task)
    if kind == "simple":
        return min(default, 8)
    if kind == "visual":
        return min(default, 16)
    return min(default, 14)
