from __future__ import annotations

from typing import Any


def classify_task(task: dict[str, Any]) -> dict[str, Any]:
    text = str(task.get("instruction") or task.get("question") or "")
    has_image = bool(task.get("image") or task.get("image_url") or task.get("image_b64") or task.get("image_path"))
    lower = text.lower()
    if has_image:
        task_type = "image_to_fact"
    elif any(token in lower for token in ("both", "after", "before", "which", "whose", "where was")) or len(text) > 120:
        task_type = "multi_hop"
    else:
        task_type = "simple_fact"
    return {
        "task_type": task_type,
        "needs_browser": task_type in {"multi_hop", "image_to_fact"},
        "needs_image_search": has_image,
        "budget": "high" if task_type != "simple_fact" else "low",
    }

