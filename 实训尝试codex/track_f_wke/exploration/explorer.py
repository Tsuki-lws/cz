from __future__ import annotations

from typing import Any


def build_exploration_prompt(seed: dict[str, Any]) -> str:
    topic = seed.get("topic") or seed.get("question") or seed.get("instruction") or ""
    return (
        "Explore public world knowledge for this topic using search/browser tools. "
        "Summarize reusable facts, entity relations, and search strategies. "
        "Do not use benchmark data or hidden labels.\n\n"
        f"Topic: {topic}"
    )


def make_lesson(seed: dict[str, Any], pred: str, tool_calls: int) -> dict[str, Any]:
    return {
        "topic": seed.get("topic") or seed.get("question") or seed.get("instruction") or "",
        "lesson": f"tool_calls={tool_calls}; summary={pred[:300]}",
        "source": seed.get("source", "wke"),
    }

