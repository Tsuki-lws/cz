from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from shared_sii_adapter.react_runner import collect_reasoning_answer_candidates

from .skills import classify_task, extract_constraints


QUERY_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "from", "by", "was", "is", "are",
    "who", "what", "where", "when", "which", "born", "director", "film", "movie",
}


def query_terms(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(query or "").lower())
        if len(token) > 2 and token not in QUERY_STOPWORDS
    }


def query_redundancy(queries: list[str]) -> dict[str, Any]:
    exact: dict[str, int] = {}
    near_pairs = 0
    normalized = [" ".join(str(query or "").lower().split()) for query in queries if str(query or "").strip()]
    for query in normalized:
        exact[query] = exact.get(query, 0) + 1
    term_sets = [query_terms(query) for query in normalized]
    for idx, terms in enumerate(term_sets):
        if not terms:
            continue
        for other in term_sets[:idx]:
            if not other:
                continue
            overlap = len(terms & other) / max(1, len(terms | other))
            if overlap >= 0.72:
                near_pairs += 1
                break
    return {
        "search_text_queries": len(normalized),
        "exact_duplicates": sum(count - 1 for count in exact.values() if count > 1),
        "near_duplicate_queries": near_pairs,
    }


def infer_answer_slot(task: dict[str, Any]) -> str:
    question = str(task.get("instruction") or task.get("question") or task.get("query") or "").lower()
    if any(key in question for key in ["who", "ceo", "chief executive", "founder", "author", "director", "person", "人物"]):
        return "person"
    if any(key in question for key in ["where", "place", "city", "country", "国家", "哪里", "地点"]):
        return "place"
    if any(key in question for key in ["year", "date", "when", "哪一年", "时间"]):
        return "date"
    if any(key in question for key in ["acronym", "abbreviation", "缩写", "简称"]):
        return "acronym"
    if any(key in question for key in ["number", "price", "how many", "多少", "数值"]):
        return "number"
    return "entity"


def infer_answer_slot_failure(task: dict[str, Any], pred: str, candidates: list[str]) -> str:
    pred = str(pred or "").strip()
    pred_l = pred.lower()
    if pred_l in {"", "none", "unknown", "n/a", "null", "</tool_call>"}:
        return "invalid_final_answer"
    slot = infer_answer_slot(task)
    if slot == "person" and candidates:
        if " " not in pred and any(" " in candidate for candidate in candidates):
            return "person_name_truncated"
        if pred_l in {"hugging face", "nike", "sam", "openai", "google", "microsoft"}:
            return "company_instead_of_person"
    if slot == "acronym" and candidates and not (pred.isupper() and len(pred) <= 12):
        if any(candidate.isupper() and 2 <= len(candidate) <= 12 for candidate in candidates):
            return "description_instead_of_acronym"
    if re.search(r"</?tool", pred_l):
        return "tool_text_as_answer"
    return ""


class EvoMemory:
    def __init__(self, path: str = "experiments/track_d_slot_evo/evo_memory.jsonl") -> None:
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

    def retrieve(self, task: dict[str, Any], k: int = 3) -> tuple[str, list[dict[str, Any]]]:
        task_type = classify_task(task)
        constraints = set(extract_constraints(task))
        answer_slot = infer_answer_slot(task)
        rows = self._rows()
        if not rows:
            return "", []
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            score = 0.0
            if row.get("task_type") == task_type:
                score += 3.0
            if row.get("answer_slot") == answer_slot:
                score += 1.4
            if row.get("failure_type") in {"temporal_drift", "entity_drift", "image_evidence_missing"}:
                score += 0.6
            if row.get("slot_failure"):
                score += 0.4
            row_constraints = set(row.get("constraints") or [])
            score += min(2.0, 0.5 * len(constraints & row_constraints))
            if row.get("outcome") == "passed":
                score += 0.5
            if row.get("tool_pattern"):
                score += 0.2
            if score > 0:
                scored.append((score, idx, row))
        hits = [row for _, _, row in sorted(scored, key=lambda item: (-item[0], -item[1]))[:k]]
        lines = [
            "Run-level memory summary:",
            f"- Similar prior cases in this run: {len(hits)}. Use only the reusable strategy, never copy an old answer.",
            f"- Current answer slot: {answer_slot}. Final answer must fill exactly this slot.",
            "- Memory is a finalization guard, not a reason to skip evidence. Keep the normal minimum evidence requirement for this task.",
            "- Avoid repeated search intent: if a query returns evidence, the next query must add a missing relation, date, disambiguator, or candidate name.",
        ]
        strategy_counts: dict[str, int] = {}
        for row in hits:
            strategy = str(row.get("strategy") or "").strip()
            if strategy:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        for strategy, count in sorted(strategy_counts.items(), key=lambda item: (-item[1], item[0]))[:2]:
            lines.append(f"- x{count}: {strategy}")
        policy_lessons = []
        for row in hits:
            state = row.get("policy_state") or {}
            missing = state.get("missing_slots") or []
            if "source_trust_filter" in missing:
                policy_lessons.append("drop low-trust/search-polluted sources before continuing")
            if "temporal_constraint_in_query" in missing:
                policy_lessons.append("carry temporal constraints into the query and final comparison")
            if "query_information_gain" in missing:
                policy_lessons.append("avoid same-intent query loops; add a new relation/date/entity")
            if "tool_failure_fallback" in missing:
                policy_lessons.append("switch source after repeated browser errors")
        if policy_lessons:
            lines.append("- Policy reminders: " + "; ".join(sorted(set(policy_lessons))[:3]) + ".")
        slot_failures = sorted({str(row.get("slot_failure")) for row in hits if row.get("slot_failure")})
        if slot_failures:
            lines.append("- Avoid repeated finalization errors: " + ", ".join(slot_failures[:3]))
        return "\n".join(lines), hits

    def update(self, item: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def reflection_from_signal(self, task: dict[str, Any], result: Any, signal: dict[str, Any]) -> str:
        failure = str(signal.get("failure_type") or "")
        task_type = classify_task(task)
        if failure == "likely_ok":
            return "local reflection: answer is non-empty and tool trajectory looks usable; no long-term failure memory needed."
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

    def build_item(self, task: dict[str, Any], result: Any, signal: dict[str, Any], *, reflection_hint: str = "") -> dict[str, Any]:
        from .policy import evidence_state, strategy_from_state

        task_type = classify_task(task)
        failure = str(signal.get("failure_type") or "")
        candidates = collect_reasoning_answer_candidates(result.trajectory)
        answer_slot = infer_answer_slot(task)
        slot_failure = infer_answer_slot_failure(task, result.pred, candidates)
        policy_state = evidence_state(task, result.trajectory)
        tool_rows = [row for row in result.trajectory if row.get("role") == "tool"]
        tool_names = [str(row.get("fn_name")) for row in tool_rows if row.get("fn_name")]
        search_queries = [
            str((row.get("fn_args") or {}).get("query") or "")
            for row in tool_rows
            if row.get("fn_name") == "search_text"
        ]
        redundancy = query_redundancy(search_queries)
        image_tools = sum(1 for name in tool_names if name == "search_image")
        text_tools = sum(1 for name in tool_names if name == "search_text")
        browser_tools = sum(1 for name in tool_names if name.startswith("browser"))
        if task_type.startswith("visual"):
            if image_tools:
                strategy = "For visual questions, use search_image once to lock the entity, then use one targeted text query for the requested attribute."
            else:
                strategy = "For visual questions, prefer search_image before broad text search; keep the pictured entity locked."
        elif task_type in {"multi_hop", "temporal_constraint"}:
            strategy = "For text multi-hop or temporal questions, use provided context first; if searching, query only the missing relation and then answer the target slot."
        else:
            strategy = "For open QA, keep enough evidence for named entities, then answer with the shortest slot value."
        if failure == "overbroad_answer":
            strategy = "The final answer must contain only the requested slot; avoid explanations, citations, and Markdown."
        elif failure == "tool_call_leak":
            strategy = "Never write XML/JSON tool calls as text; use native tool calls and final answer JSON only at the end."
        elif failure == "image_evidence_missing":
            strategy = "If image search fails, extract visible text/entity clues and run one focused search_text query instead of guessing."
        elif failure == "temporal_drift":
            strategy = "Preserve temporal constraints in queries and final reasoning: first, before, after, as-of, season, and explicit years."
        if redundancy["exact_duplicates"] or redundancy["near_duplicate_queries"]:
            strategy = (
                "Avoid repeated or near-duplicate search queries. After one search, either answer from gathered evidence or add a new missing relation/date/disambiguator; "
                "do not re-search the same concept with only wording changes."
            )
        policy_strategy = strategy_from_state(policy_state)
        if policy_strategy:
            strategy = policy_strategy
        if slot_failure:
            strategy = (
                "Before final answer, run a target-slot check using the gathered evidence. Prefer a short candidate only when it is explicitly supported and matches the requested "
                f"{answer_slot} slot; do not output broader entities, sources, or tool text."
            )
        return {
            "index": result.index,
            "task_type": task_type,
            "answer_slot": answer_slot,
            "constraints": extract_constraints(task),
            "failure_type": failure,
            "slot_failure": slot_failure,
            "outcome": "passed" if signal.get("pass") else "reviewed",
            "strategy": strategy,
            "reflection": reflection_hint or self.reflection_from_signal(task, result, signal),
            "risk": str(signal.get("rationale") or ""),
            "tool_pattern": {
                "search_image": image_tools,
                "search_text": text_tools,
                "browser": browser_tools,
            },
            "query_redundancy": redundancy,
            "policy_state": policy_state,
            "reasoning_candidate_types": [
                {
                    "len": len(candidate),
                    "has_space": " " in candidate,
                    "looks_acronym": candidate.isupper() and 2 <= len(candidate) <= 12,
                }
                for candidate in candidates[:3]
            ],
            "pred_preview": str(result.pred or "")[:120],
            "metrics": {
                "tokens": result.metrics.get("tokens", 0),
                "turns": result.metrics.get("turns", 0),
                "tool_calls": result.metrics.get("tool_calls", 0),
                "latency": result.metrics.get("latency", 0),
            },
        }
