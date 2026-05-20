from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlparse
from typing import Any

from .memory_evo import infer_answer_slot, query_redundancy


LOW_TRUST_TITLE_RE = re.compile(
    r"(pdf\s+download|free\s+pdf|ebook|slideshare|scribd|coursehero|studocu|archive|"
    r"electric life of|answers?|chegg|quizlet|brainly)",
    re.I,
)
ERROR_RE = re.compile(r"\b(403|404|429|500|502|503|forbidden|access denied|timeout|error)\b", re.I)
TEMPORAL_RE = re.compile(
    r"\b(first|before|after|as of|shown|season|current|earlier|later|between|"
    r"\d+\s+years?\s+before|\d+\s+years?\s+after|in\s+\d{4}|by\s+\d{4})\b",
    re.I,
)


def _instruction(task: dict[str, Any]) -> str:
    return str(task.get("instruction") or task.get("question") or task.get("query") or "")


def _tool_rows(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in trajectory if row.get("role") == "tool"]


def _search_queries(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str((row.get("fn_args") or {}).get("query") or "")
        for row in rows
        if row.get("fn_name") == "search_text"
    ]


def _domain(url: str) -> str:
    host = urlparse(str(url or "")).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _extract_result_objects(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    text = str(content or "")
    objects: list[dict[str, Any]] = []
    for title, url in re.findall(r'"title"\s*:\s*"([^"]+)".{0,300}?"url"\s*:\s*"([^"]+)"', text, flags=re.S):
        objects.append({"title": title, "url": url})
    return objects


def source_trust_state(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _tool_rows(trajectory)
    suspicious: list[dict[str, Any]] = []
    error_domains: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    for row in rows:
        content = row.get("content")
        if ERROR_RE.search(str(content or "")):
            args = row.get("fn_args") or {}
            domain = _domain(str(args.get("url") or args.get("page_url") or ""))
            if domain:
                error_domains[domain] += 1
        for item in _extract_result_objects(content):
            title = str(item.get("title") or "")
            url = str(item.get("url") or "")
            domain = _domain(url)
            if domain:
                domains[domain] += 1
            if LOW_TRUST_TITLE_RE.search(title) or ERROR_RE.search(title):
                suspicious.append({"title": title[:120], "domain": domain, "reason": "title_or_access_error"})
    return {
        "domains": dict(domains.most_common(8)),
        "error_domains": dict(error_domains.most_common(5)),
        "suspicious_sources": suspicious[:5],
        "low_trust_source_count": len(suspicious),
    }


def evidence_state(task: dict[str, Any], trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _tool_rows(trajectory)
    question = _instruction(task)
    queries = _search_queries(rows)
    tool_names = [str(row.get("fn_name") or "") for row in rows]
    has_temporal_constraint = bool(TEMPORAL_RE.search(question))
    temporal_queries = [query for query in queries if TEMPORAL_RE.search(query)]
    trust = source_trust_state(trajectory)
    redundancy = query_redundancy(queries)
    missing: list[str] = []
    if has_temporal_constraint and queries and not temporal_queries:
        missing.append("temporal_constraint_in_query")
    if redundancy.get("exact_duplicates") or redundancy.get("near_duplicate_queries"):
        missing.append("query_information_gain")
    if trust.get("low_trust_source_count"):
        missing.append("source_trust_filter")
    if trust.get("error_domains") and not any(name in {"search_text", "browser_parallel"} for name in tool_names[-2:]):
        missing.append("tool_failure_fallback")
    return {
        "answer_slot": infer_answer_slot(task),
        "has_temporal_constraint": has_temporal_constraint,
        "temporal_queries": temporal_queries[:4],
        "query_redundancy": redundancy,
        "source_trust": trust,
        "tool_counts": dict(Counter(tool_names)),
        "missing_slots": missing,
    }


def strategy_from_state(state: dict[str, Any]) -> str:
    missing = set(state.get("missing_slots") or [])
    if "source_trust_filter" in missing:
        return "Discard low-trust or semantically mismatched sources before spending more searches; anchor on official/planning/wiki-like evidence."
    if "temporal_constraint_in_query" in missing:
        return "Carry the temporal relation into the next query and compute the requested date relation before finalizing."
    if "query_information_gain" in missing:
        return "Stop near-duplicate search loops; every new query must add a missing entity, relation, date, or candidate source."
    if "tool_failure_fallback" in missing:
        return "After repeated browser errors, switch source or use search snippets instead of retrying the same failing page."
    return ""
