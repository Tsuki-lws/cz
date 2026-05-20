from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared_sii_adapter.react_runner import find_similar_search_query

from .skills import classify_task, extract_constraints


class EvoMemory:
    def __init__(self, path: str = "experiments/track_d_evo/evo_memory.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def retrieve(self, task: dict[str, Any], k: int = 2) -> tuple[str, list[dict[str, Any]]]:
        task_type = classify_task(task)
        constraints = set(extract_constraints(task))
        rows = self._rows()
        if not rows:
            return "", []
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            score = 0.0
            if row.get("task_type") == task_type:
                score += 3.0
            if row.get("failure_type") in {"temporal_drift", "entity_drift", "image_evidence_missing"}:
                score += 0.6
            row_constraints = set(row.get("constraints") or [])
            score += min(2.0, 0.5 * len(constraints & row_constraints))
            if row.get("outcome") == "passed":
                score += 0.5
            if row.get("tool_pattern"):
                score += 0.2
            if score > 0:
                scored.append((score, idx, row))
        hits = [row for _, _, row in sorted(scored, key=lambda item: (-item[0], -item[1]))[:k]]
        lines = [f"Run memory: {len(hits)} similar cases. Reuse only planning strategy, not answers."]
        strategy_counts: dict[str, int] = {}
        for row in hits:
            strategy = str(row.get("strategy") or "").strip()
            if strategy:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        for strategy, count in sorted(strategy_counts.items(), key=lambda item: (-item[1], item[0]))[:1]:
            lines.append(f"- x{count}: {strategy}")
        duplicate_risks = sum(int(row.get("duplicate_query_count") or 0) for row in hits)
        if duplicate_risks:
            lines.append("- Avoid near-duplicate searches. If a query repeats the same entity+slot, answer or query a different missing slot.")
        return "\n".join(lines), hits

    def update(self, item: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def reflection_from_signal(self, task: dict[str, Any], result: Any, signal: dict[str, Any]) -> str:
        failure = str(signal.get("failure_type") or "")
        task_type = classify_task(task)
        if failure == "likely_ok":
            return "local reflection: answer is non-empty and tool trajectory looks usable; keep the same evidence discipline."
        if failure == "no_answer":
            return "local reflection: no final answer was produced; future similar tasks should stop earlier and force a short final answer."
        if failure == "tool_call_leak":
            return "local reflection: raw tool syntax leaked into the answer; future similar tasks must use native tool calls only."
        if failure == "uncertain_answer":
            return "local reflection: answer was uncertain; future similar tasks should issue one targeted query with entity plus requested slot."
        if failure == "image_evidence_missing":
            return "local reflection: image search lacked usable evidence; future visual tasks should combine image clues with one text query."
        if failure == "temporal_drift":
            return "local reflection: temporal constraint was weak; future tasks should include first/before/after/as-of/year terms in the query."
        if failure == "entity_drift":
            return "local reflection: entity may have drifted; future visual tasks should lock the pictured entity before asking attributes."
        return f"local reflection: {task_type} ended with {failure or 'unknown signal'}; keep evidence concise and answer only the target slot."

    def build_item(self, task: dict[str, Any], result: Any, signal: dict[str, Any]) -> dict[str, Any]:
        task_type = classify_task(task)
        failure = str(signal.get("failure_type") or "")
        tool_names = [str(row.get("fn_name")) for row in result.trajectory if row.get("role") == "tool" and row.get("fn_name")]
        search_queries = [
            str((row.get("fn_args") or {}).get("query") or "")
            for row in result.trajectory
            if row.get("role") == "tool" and row.get("fn_name") == "search_text"
        ]
        seen_queries: list[str] = []
        duplicate_query_count = 0
        for query in search_queries:
            if find_similar_search_query(query, seen_queries):
                duplicate_query_count += 1
            else:
                seen_queries.append(query)
        image_tools = sum(1 for name in tool_names if name == "search_image")
        text_tools = sum(1 for name in tool_names if name == "search_text")
        browser_tools = sum(1 for name in tool_names if name.startswith("browser"))
        if task_type.startswith("visual"):
            if image_tools:
                strategy = "For visual questions, use search_image once to lock the entity, then use one targeted text query for the requested attribute."
            else:
                strategy = "For visual questions, prefer search_image before broad text search; keep the pictured entity locked."
        elif task_type in {"multi_hop", "temporal_constraint"}:
            strategy = "For text multi-hop or temporal questions, decompose into entity plus target slot; include temporal words or years in the first query."
        else:
            strategy = "For open QA, search only when the named entity or requested slot is uncertain, then answer with the shortest slot value."
        if failure == "overbroad_answer":
            strategy = "The final answer must contain only the requested slot; avoid explanations, citations, and Markdown."
        elif failure == "tool_call_leak":
            strategy = "Never write XML/JSON tool calls as text; use native tool calls and final answer JSON only at the end."
        elif failure == "image_evidence_missing":
            strategy = "If image search fails, extract visible text/entity clues and run one focused search_text query instead of guessing."
        elif failure == "temporal_drift":
            strategy = "Preserve temporal constraints in queries and final reasoning: first, before, after, as-of, season, and explicit years."
        if duplicate_query_count:
            strategy = "Avoid near-duplicate searches: after one entity+slot query, either answer from evidence or query a different missing slot."
        return {
            "index": result.index,
            "task_type": task_type,
            "constraints": extract_constraints(task),
            "failure_type": failure,
            "outcome": "passed" if signal.get("pass") else "reviewed",
            "strategy": strategy,
            "risk": str(signal.get("rationale") or ""),
            "tool_pattern": {
                "search_image": image_tools,
                "search_text": text_tools,
                "browser": browser_tools,
            },
            "duplicate_query_count": duplicate_query_count,
            "pred_preview": str(result.pred or "")[:120],
            "metrics": {
                "tokens": result.metrics.get("tokens", 0),
                "turns": result.metrics.get("turns", 0),
                "tool_calls": result.metrics.get("tool_calls", 0),
                "latency": result.metrics.get("latency", 0),
            },
        }
