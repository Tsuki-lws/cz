"""Lightweight skill memory for prompt-time strategy injection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def detect_task_type(question: str) -> str:
    q = question.lower()
    if re.search(r'哪一年|年份|year|when|date', q):
        return 'date_year'
    if re.search(r'country|region|国家|地区|originated|from where', q):
        return 'country_region'
    if re.search(r'bridge|fortress|church|temple|landmark|地点|地标|城市|位于|where', q):
        return 'location_landmark'
    if re.search(r'name|叫什么|名称|名字|哪位|who|人物|作者|scientist|actor|singer|president', q):
        return 'entity_name'
    if re.search(r'颜色|color|left|right|side|位置|哪边|how many|数量|几个', q):
        return 'visual_attribute'
    if re.search(r'what is|是什么|类别|类型|关系|运动|style|family|disease', q):
        return 'type_concept'
    return 'other'


class SkillMemory:
    def __init__(self, skills: list[dict[str, Any]]):
        self.skills = skills

    @classmethod
    def load(cls, path: str | None) -> 'SkillMemory | None':
        if not path:
            return None
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f'skill memory not found: {path}')
        payload = json.loads(resolved.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            skills = payload.get('skills', [])
        elif isinstance(payload, list):
            skills = payload
        else:
            skills = []
        return cls([skill for skill in skills if isinstance(skill, dict)])

    def retrieve(self, question: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        task_type = detect_task_type(question)
        scored: list[tuple[float, dict[str, Any]]] = []
        for skill in self.skills:
            applies_to = set(skill.get('applies_to') or [])
            score = float(skill.get('priority', 0.0))
            if task_type in applies_to:
                score += 10.0
            if 'all' in applies_to:
                score += 2.0
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]


def format_skill_instructions(skills: list[dict[str, Any]]) -> str:
    if not skills:
        return ''
    lines = ['Reusable lessons from previous failures:']
    for index, skill in enumerate(skills, start=1):
        name = skill.get('name', f'skill_{index}')
        instruction = skill.get('instruction') or skill.get('better_strategy') or ''
        avoid = skill.get('avoid') or ''
        if avoid:
            lines.append(f'{index}. {name}: {instruction} Avoid: {avoid}')
        else:
            lines.append(f'{index}. {name}: {instruction}')
    lines.append('Apply only the relevant lesson; do not mention these lessons in the final answer.')
    return '\n'.join(lines)

