"""Heuristic failure taxonomy and structured reflections.

The goal is not to replace human review. It gives every failed example a stable
machine-readable label so prompt, memory, and teacher-routing changes can be
compared across runs.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


FAILURE_TYPES = [
    'CORRECT',
    'FORMAT_ERROR',
    'PARTIAL_MATCH',
    'EMPTY_ANSWER',
    'TOO_VERBOSE',
    'LOOPING',
    'TOOL_PROTOCOL_ERROR',
    'IMAGE_MISREAD',
    'ENTITY_MISIDENTIFIED',
    'KNOWLEDGE_ERROR',
    'COUNTRY_REGION_ERROR',
    'DATE_YEAR_ERROR',
    'CONCEPT_ERROR',
    'HALLUCINATION',
]

TASK_TYPES = [
    'date_year',
    'country_region',
    'entity_name',
    'location_landmark',
    'type_concept',
    'visual_attribute',
    'other',
]

SKILL_BY_FAILURE = {
    'FORMAT_ERROR': 'answer_normalization',
    'PARTIAL_MATCH': 'answer_normalization',
    'EMPTY_ANSWER': 'force_final_answer',
    'TOO_VERBOSE': 'concise_answer',
    'LOOPING': 'loop_breaker',
    'TOOL_PROTOCOL_ERROR': 'tool_protocol_guard',
    'IMAGE_MISREAD': 'image_first_then_verify',
    'ENTITY_MISIDENTIFIED': 'entity_identification_then_verify',
    'KNOWLEDGE_ERROR': 'verify_before_final',
    'COUNTRY_REGION_ERROR': 'normalize_country_region',
    'DATE_YEAR_ERROR': 'verify_date_year',
    'CONCEPT_ERROR': 'concept_disambiguation',
    'HALLUCINATION': 'evidence_grounding',
}


@dataclass(slots=True)
class Reflection:
    failure_type: str
    task_type: str
    root_cause: str
    wrong_assumption: str
    missing_information: str
    better_strategy: str
    tool_suggestion: str
    activated_skill: str
    confidence: float


def normalize_answer(text: Any) -> str:
    value = unicodedata.normalize('NFKC', str(text or '').strip().lower())
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'\([^)]*\)', ' ', value)
    value = re.sub(r'（[^）]*）', ' ', value)
    value = re.sub(r'[`*_"“”‘’.,!?;:，。！？；：、\[\]{}]', ' ', value)
    value = re.sub(r'\b(the|a|an)\b', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def exact_match(gold: Any, pred: Any) -> bool:
    return str(gold or '').strip().lower() == str(pred or '').strip().lower()


def normalized_match(gold: Any, pred: Any) -> bool:
    return normalize_answer(gold) == normalize_answer(pred)


def loose_match(gold: Any, pred: Any) -> bool:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    if not gold_norm or not pred_norm:
        return False
    if gold_norm == pred_norm:
        return True
    if len(pred_norm) <= max(len(gold_norm) * 4, len(gold_norm) + 30) and gold_norm in pred_norm:
        return True
    return len(gold_norm) <= max(len(pred_norm) * 4, len(pred_norm) + 30) and pred_norm in gold_norm


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


def has_repetition(text: str) -> bool:
    if not text:
        return False
    chunks = re.findall(r'[^。.!?\n]{8,80}', text)
    counts = Counter(chunk.strip().lower() for chunk in chunks if chunk.strip())
    return any(count >= 4 for count in counts.values())


def classify_failure(record: dict[str, Any], trajectory: dict[str, Any] | None = None) -> str:
    gold = record.get('answer', '')
    pred = record.get('pred', '')
    question = record.get('instruction') or record.get('question') or ''
    pred_text = str(pred or '').strip()
    pred_lower = pred_text.lower()

    if exact_match(gold, pred):
        return 'CORRECT'
    if normalized_match(gold, pred):
        return 'FORMAT_ERROR'
    if loose_match(gold, pred):
        return 'PARTIAL_MATCH'
    if not pred_text:
        return 'EMPTY_ANSWER'
    if 'tool_code' in pred_lower or 'google_search' in pred_lower or '<tool_call>' in pred_lower:
        return 'TOOL_PROTOCOL_ERROR'
    if has_repetition(pred_text):
        return 'LOOPING'
    if len(pred_text) > 120 or '\n' in pred_text:
        return 'TOO_VERBOSE'

    task_type = detect_task_type(question)
    if task_type == 'date_year':
        return 'DATE_YEAR_ERROR'
    if task_type == 'country_region':
        return 'COUNTRY_REGION_ERROR'
    if task_type == 'entity_name':
        return 'ENTITY_MISIDENTIFIED'
    if task_type == 'location_landmark':
        return 'IMAGE_MISREAD'
    if task_type in {'type_concept', 'visual_attribute'}:
        return 'CONCEPT_ERROR'

    if trajectory:
        records = trajectory.get('records') or []
        saw_image = any('image_url' in str(item.get('content', '')) for item in records)
        if saw_image:
            return 'IMAGE_MISREAD'
    return 'HALLUCINATION'


def build_reflection(record: dict[str, Any], trajectory: dict[str, Any] | None = None) -> Reflection:
    failure_type = classify_failure(record, trajectory)
    task_type = detect_task_type(record.get('instruction') or record.get('question') or '')
    skill = SKILL_BY_FAILURE.get(failure_type, 'none')

    root_causes = {
        'CORRECT': 'The prediction matches the reference answer.',
        'FORMAT_ERROR': 'The answer is semantically close but formatted differently from the reference.',
        'PARTIAL_MATCH': 'The prediction contains the key answer but includes extra or less specific text.',
        'EMPTY_ANSWER': 'The model failed to emit a final answer.',
        'TOO_VERBOSE': 'The model did not follow the concise final-answer format.',
        'LOOPING': 'The model repeated the same reasoning pattern instead of finalizing.',
        'TOOL_PROTOCOL_ERROR': 'The model emitted pseudo tool code or search commands as text.',
        'IMAGE_MISREAD': 'The visual entity or landmark was likely misread.',
        'ENTITY_MISIDENTIFIED': 'The model identified the wrong person/object/entity.',
        'KNOWLEDGE_ERROR': 'The visual entity may be right, but the required fact is wrong.',
        'COUNTRY_REGION_ERROR': 'The model confused the origin, country, or region.',
        'DATE_YEAR_ERROR': 'The model returned the wrong date or year.',
        'CONCEPT_ERROR': 'The model returned the wrong category, relation, style, or concept.',
        'HALLUCINATION': 'The prediction is unsupported or contradicts the reference answer.',
    }
    strategies = {
        'FORMAT_ERROR': 'Normalize aliases and strip units/punctuation before scoring or training.',
        'PARTIAL_MATCH': 'Train the student to output only the minimal answer span.',
        'EMPTY_ANSWER': 'Inject a force-answer control message and cap reasoning output.',
        'TOO_VERBOSE': 'Use direct answer mode and reduce max_tokens for benchmark inference.',
        'LOOPING': 'Detect repeated text/actions and switch to forced finalization.',
        'TOOL_PROTOCOL_ERROR': 'Forbid pseudo tool calls; only use registered tool_call objects.',
        'IMAGE_MISREAD': 'Route to the vision teacher first, then verify the named entity.',
        'ENTITY_MISIDENTIFIED': 'Ask the vision teacher for top-k entity candidates before answering.',
        'KNOWLEDGE_ERROR': 'Use the text teacher to verify the fact after entity recognition.',
        'COUNTRY_REGION_ERROR': 'Verify country/region aliases and normalize before final answer.',
        'DATE_YEAR_ERROR': 'Verify date/year facts with text teacher and answer with only the year/date.',
        'CONCEPT_ERROR': 'Disambiguate the visual concept before finalizing.',
        'HALLUCINATION': 'Require evidence from image or teacher verification before final answer.',
        'CORRECT': 'Keep this trajectory as a positive example.',
    }

    tool_suggestion = 'none'
    if failure_type in {'IMAGE_MISREAD', 'ENTITY_MISIDENTIFIED'}:
        tool_suggestion = 'vision_teacher'
    elif failure_type in {'KNOWLEDGE_ERROR', 'COUNTRY_REGION_ERROR', 'DATE_YEAR_ERROR'}:
        tool_suggestion = 'text_teacher'
    elif failure_type in {'TOOL_PROTOCOL_ERROR', 'LOOPING'}:
        tool_suggestion = 'harness_control'

    return Reflection(
        failure_type=failure_type,
        task_type=task_type,
        root_cause=root_causes[failure_type],
        wrong_assumption='Prediction can be accepted without verification.' if failure_type != 'CORRECT' else '',
        missing_information='Reliable visual/entity/fact evidence for the requested answer.' if failure_type != 'CORRECT' else '',
        better_strategy=strategies[failure_type],
        tool_suggestion=tool_suggestion,
        activated_skill=skill,
        confidence=0.85 if failure_type in {'CORRECT', 'FORMAT_ERROR', 'PARTIAL_MATCH', 'EMPTY_ANSWER', 'TOO_VERBOSE', 'TOOL_PROTOCOL_ERROR', 'LOOPING'} else 0.65,
    )


def reflection_dict(record: dict[str, Any], trajectory: dict[str, Any] | None = None) -> dict[str, Any]:
    return asdict(build_reflection(record, trajectory))

