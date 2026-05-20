from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
_QUOTED_RE = re.compile(r'"([^"]{3,90})"|“([^”]{3,90})”|\'([^\']{3,90})\'')
_CAP_PHRASE_RE = re.compile(r"\b[A-Z][A-Za-z0-9&'.-]*(?:\s+[A-Z][A-Za-z0-9&'.-]*){0,5}\b")
_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
_RELATION_WORDS = [
    "wife",
    "husband",
    "spouse",
    "mother",
    "father",
    "aunt",
    "director",
    "author",
    "founder",
    "ceo",
    "chief executive",
    "coach",
    "advisor",
    "teacher",
    "student",
    "publisher",
    "producer",
]
_TEMPORAL_WORDS = [
    "first",
    "last",
    "before",
    "after",
    "as of",
    "current",
    "between",
    "from",
    "to",
    "season",
    "shown",
    "anniversary",
]


@dataclass(slots=True)
class SearchPlan:
    answer_slot: str
    task_type: str
    high_constraints: list[str]
    medium_constraints: list[str]
    low_constraints: list[str]
    query_seeds: list[str]
    policy: str


def _question(task: dict[str, Any]) -> str:
    return str(task.get("instruction") or task.get("question") or task.get("query") or "")


def _has_image(task: dict[str, Any]) -> bool:
    return bool(task.get("image") or task.get("image_url") or task.get("image_b64") or task.get("image_path"))


def infer_answer_slot(task: dict[str, Any]) -> str:
    question = _question(task).lower()
    head = question[:140]
    if any(key in head for key in ["when", "what date", "which date", "what day", "which day"]):
        return "date"
    if any(key in head for key in ["what year", "which year", "year did", "year was"]):
        return "date_or_year"
    if any(key in head for key in ["where", "which city", "what city", "which country", "what country", "nationality"]):
        return "place_or_nationality"
    if any(key in head for key in ["how many", "what number", "which number", "price", "percentage", "score", "rank"]):
        return "number"
    if any(key in head for key in ["which film", "what film", "which movie", "what movie"]):
        return "film"
    if any(key in head for key in ["which book", "what book", "which novel", "what novel"]):
        return "book"
    if any(key in head for key in ["which song", "what song", "which album", "what album"]):
        return "music_work"
    if any(key in head for key in ["which team", "what team", "which club", "what club"]):
        return "team"
    if any(key in head for key in ["which company", "what company", "which organization", "what organization"]):
        return "organization"
    if any(key in head for key in ["acronym", "abbreviation", "abbreviated", "short for"]):
        return "acronym"
    if any(key in head for key in ["color", "colour"]):
        return "color"
    if question.strip().startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ")):
        return "yes_no"
    if any(key in head for key in ["who", "whose wife", "wife of", "husband of", "spouse", "ceo", "chief executive", "founder", "author", "director", "actor", "player", "person"]):
        return "person"
    return "entity"


def classify_task(task: dict[str, Any]) -> str:
    question = _question(task).lower()
    if _has_image(task):
        return "visual_qa"
    if len(question) > 320 or question.count(" between ") >= 2 or sum(word in question for word in _RELATION_WORDS) >= 2:
        return "multi_hop"
    if any(word in question for word in _TEMPORAL_WORDS):
        return "temporal_qa"
    return "open_qa"


def _dedupe(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item or "").strip(" ,.;:()[]{}").split())
        if len(text) < 3:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _quoted_phrases(question: str) -> list[str]:
    phrases: list[str] = []
    for match in _QUOTED_RE.findall(question):
        phrases.extend(part for part in match if part)
    return phrases


def _capital_phrases(question: str) -> list[str]:
    blocked = {
        "What",
        "Which",
        "Who",
        "Where",
        "When",
        "The",
        "In",
        "All",
        "According",
        "If",
    }
    phrases = []
    for phrase in _CAP_PHRASE_RE.findall(question):
        if phrase.split()[0] in blocked:
            continue
        if len(phrase) <= 3:
            continue
        phrases.append(phrase)
    return phrases


def build_search_plan(task: dict[str, Any]) -> SearchPlan:
    question = _question(task)
    lowered = question.lower()
    quoted = _quoted_phrases(question)
    caps = _capital_phrases(question)
    dates = _DATE_RE.findall(question)
    years = _YEAR_RE.findall(question)
    relations = [word for word in _RELATION_WORDS if word in lowered]
    temporal = [word for word in _TEMPORAL_WORDS if word in lowered]

    rare_patterns = []
    rare_patterns.extend(re.findall(r"\b\d+\s*(?:year|years|km|kilometer|kilometers|mile|miles|percent|%)\b[^,.;]{0,40}", question, flags=re.I))
    rare_patterns.extend(re.findall(r"\b(?:born|died|founded|published|released|launched|enrolled|admitted|reviewed)\b[^,.;]{0,60}", question, flags=re.I))

    high = _dedupe(quoted + rare_patterns + dates + caps[:4], 6)
    medium = _dedupe(relations + temporal + years[:6], 8)
    low = _dedupe(re.findall(r"\b(?:actor|film|movie|book|university|company|team|album|song|country|city|school|station|program)\b", lowered), 8)

    query_seeds: list[str] = []
    if quoted:
        query_seeds.append(f'"{quoted[0]}"')
    if caps:
        query_seeds.append(" ".join(caps[:2]))
    if rare_patterns and caps:
        query_seeds.append(f"{caps[0]} {rare_patterns[0]}")
    elif rare_patterns:
        query_seeds.append(rare_patterns[0])
    if caps and relations:
        query_seeds.append(f"{caps[0]} {relations[0]}")
    if years and caps:
        query_seeds.append(f"{caps[0]} {years[0]}")
    query_seeds = _dedupe(query_seeds, 4)

    policy = (
        "Fill slots in this order: target slot -> high-info constraints -> candidate -> missing/conflicting constraints. "
        "Use search to create hypotheses, then verify candidates. New queries should add a new information dimension, "
        "not just paraphrase the previous query. Browser pages are for evidence when snippets are insufficient."
    )

    return SearchPlan(
        answer_slot=infer_answer_slot(task),
        task_type=classify_task(task),
        high_constraints=high,
        medium_constraints=medium,
        low_constraints=low,
        query_seeds=query_seeds,
        policy=policy,
    )


def format_search_plan(plan: SearchPlan) -> str:
    lines = [
        "Search policy for this task:",
        f"- answer_slot: {plan.answer_slot}",
        f"- task_type: {plan.task_type}",
    ]
    if plan.high_constraints:
        lines.append("- high_information_constraints: " + "; ".join(plan.high_constraints[:5]))
    if plan.medium_constraints:
        lines.append("- medium_constraints: " + "; ".join(plan.medium_constraints[:6]))
    if plan.low_constraints:
        lines.append("- low_information_constraints: " + "; ".join(plan.low_constraints[:6]))
    if plan.query_seeds:
        lines.append("- query_seed_examples: " + " | ".join(plan.query_seeds[:4]))
    lines.append("- policy: " + plan.policy)
    lines.append("- final_check: answer only the requested answer_slot, not the source, company, broader entity, or explanation.")
    return "\n".join(lines)


def build_planner_context(task: dict[str, Any]) -> tuple[str, SearchPlan]:
    plan = build_search_plan(task)
    return format_search_plan(plan), plan
