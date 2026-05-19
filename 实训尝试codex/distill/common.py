# Common helpers.

from __future__ import annotations

import json
import string
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def read_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    resolved = ensure_parent(path)
    with resolved.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    resolved = ensure_parent(path)
    with resolved.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_json(path: str | Path, payload: Any) -> None:
    resolved = ensure_parent(path)
    with resolved.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(text: Any, *, lower: bool = True, strip_punctuation: bool = True) -> str:
    normalized = '' if text is None else str(text)
    if lower:
        normalized = normalized.lower()
    normalized = ' '.join(normalized.split())
    if strip_punctuation:
        translation = str.maketrans({char: ' ' for char in string.punctuation})
        normalized = normalized.translate(translation)
        normalized = ' '.join(normalized.split())
    return normalized.strip()


def as_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [as_text(item) for item in value]
        return ' | '.join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ('text', 'answer', 'value', 'label'):
            if key in value:
                return as_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_question_field(record: dict[str, Any]) -> str:
    for key in ('instruction', 'question', 'query', 'prompt'):
        if key in record:
            return as_text(record[key])
    return ''


def fuzzy_ratio(a: str, b: str) -> float:
    from thefuzz import fuzz

    return fuzz.ratio(a, b) / 100.0
