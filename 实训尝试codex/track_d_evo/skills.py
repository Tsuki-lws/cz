from __future__ import annotations

import re
from typing import Any


def classify_task(task: dict[str, Any]) -> str:
    question = str(task.get("instruction") or task.get("question") or task.get("query") or "").lower()
    has_image = bool(task.get("image") or task.get("image_url") or task.get("image_b64") or task.get("image_path"))
    if has_image:
        if any(key in question for key in ["stock", "price", "open", "close", "higher", "drop"]):
            return "visual_stock"
        if any(key in question for key in ["team", "player", "match", "lpl", "uefa", "season"]):
            return "visual_sports_esports"
        if any(key in question for key in ["ceo", "company", "founder", "person", "actor", "who"]):
            return "visual_entity"
        return "visual_general"
    if any(key in question for key in ["first", "before", "after", "current", "as of", "in 2024", "season"]):
        return "temporal_constraint"
    if len(question) > 350 or question.count(" between ") >= 2:
        return "multi_hop"
    return "open_qa"


def extract_constraints(task: dict[str, Any]) -> list[str]:
    question = str(task.get("instruction") or task.get("question") or task.get("query") or "")
    lowered = question.lower()
    constraints: list[str] = []
    for key in ["first", "before", "after", "as of", "current", "shown", "image", "season", "final match"]:
        if key in lowered:
            constraints.append(key)
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    constraints.extend(years[:6])
    return constraints


BASE_SKILLS = [
    {
        "name": "entity_lock",
        "applies_to": ["visual_entity", "visual_sports_esports", "visual_general"],
        "priority": 0.95,
        "instruction": (
            "For image questions, first identify the pictured entity and keep it as an explicit hypothesis. "
            "Do not switch to another person/team/company unless a later source directly contradicts the image evidence."
        ),
        "avoid": "Do not answer from a generic search result that ignores the pictured entity.",
    },
    {
        "name": "temporal_lock",
        "applies_to": ["temporal_constraint", "visual_sports_esports", "visual_stock"],
        "priority": 0.9,
        "instruction": (
            "Preserve temporal words such as first, before, after, shown day, current, season, and as-of dates. "
            "When search returns later news, verify whether it is the requested time point before using it."
        ),
        "avoid": "Do not replace a first/season-specific answer with a later update.",
    },
    {
        "name": "slot_answer",
        "applies_to": ["multi_hop", "open_qa", "visual_general"],
        "priority": 0.75,
        "instruction": (
            "Before searching, write the target slot mentally: person, city, date, color, team, technology, or numeric value. "
            "Final answer must fill only that slot."
        ),
        "avoid": "Do not output explanation or the evidence source instead of the requested slot.",
    },
    {
        "name": "cross_check_candidate",
        "applies_to": ["multi_hop", "visual_entity", "visual_sports_esports", "temporal_constraint"],
        "priority": 0.8,
        "instruction": (
            "If two candidate entities appear, run one discriminating query containing both the candidate and a unique clue from the question. "
            "Prefer the candidate matching more constraints, not the more popular search result."
        ),
        "avoid": "Do not keep broadening queries after a plausible candidate appears; verify constraints instead.",
    },
    {
        "name": "tool_failure_recovery",
        "applies_to": ["visual_general", "visual_entity", "multi_hop", "open_qa"],
        "priority": 0.7,
        "instruction": (
            "When browser navigation returns 403/500/empty text, use the URL title/snippet and switch to another source or search query. "
            "Treat browser failure as missing evidence, not as evidence for a negative answer."
        ),
        "avoid": "Do not infer that a feature/person does not exist only because one page failed.",
    },
]


def select_skills(task: dict[str, Any], k: int = 4) -> list[dict[str, Any]]:
    task_type = classify_task(task)
    scored = []
    for skill in BASE_SKILLS:
        applies = set(skill.get("applies_to") or [])
        score = float(skill.get("priority", 0.0))
        if task_type in applies or "all" in applies:
            score += 1.0
        if score > 0:
            scored.append((score, skill))
    return [skill for _, skill in sorted(scored, key=lambda item: -item[0])[:k]]


def format_skills(skills: list[dict[str, Any]]) -> str:
    if not skills:
        return ""
    lines = ["Reusable skills for this task:"]
    for skill in skills:
        lines.append(f"- {skill.get('name')}: {skill.get('instruction')}")
        avoid = skill.get("avoid")
        if avoid:
            lines.append(f"  Avoid: {avoid}")
    return "\n".join(lines)
