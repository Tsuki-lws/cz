from __future__ import annotations

from statistics import mean
from typing import Any

from .types import AgentRunResult


def normalize(text: Any) -> str:
    return " ".join(str(text or "").lower().strip().split())


def summarize(results: list[AgentRunResult], gold_by_index: dict[str, str] | None = None) -> dict[str, Any]:
    gold_by_index = gold_by_index or {}
    accuracies: list[int] = []
    for result in results:
        if result.index in gold_by_index:
            accuracies.append(int(normalize(result.pred) == normalize(gold_by_index[result.index])))
    metrics = [r.metrics for r in results]
    return {
        "count": len(results),
        "accuracy": mean(accuracies) if accuracies else None,
        "avg_tokens": mean([m.get("tokens", 0) for m in metrics]) if metrics else 0,
        "avg_turns": mean([m.get("turns", 0) for m in metrics]) if metrics else 0,
        "avg_tool_calls": mean([m.get("tool_calls", 0) for m in metrics]) if metrics else 0,
        "avg_latency": mean([m.get("latency", 0.0) for m in metrics]) if metrics else 0,
    }

