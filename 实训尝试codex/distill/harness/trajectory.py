# Trajectory helpers and task-aligned jsonl serialization.

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAJECTORY_FIELDS = (
    'timestamp',
    'step_id',
    'role',
    'content',
    'tool_call_id',
    'tool_calls',
    'reasoning_content',
    'total_tokens',
    'fn_name',
    'fn_args',
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def ordered_record(**values: Any) -> dict[str, Any]:
    return {field: values.get(field) for field in TRAJECTORY_FIELDS}


def sanitize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    normalized: list[dict[str, Any]] = []
    for item in tool_calls:
        if hasattr(item, 'model_dump'):
            item = item.model_dump()
        function_block = item.get('function', {}) if isinstance(item, dict) else {}
        normalized.append(
            {
                'id': item.get('id', ''),
                'type': item.get('type', 'function'),
                'function': {
                    'name': function_block.get('name', ''),
                    'arguments': function_block.get('arguments', '{}'),
                },
            }
        )
    return normalized


@dataclass
class TrajectoryWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        payload = ordered_record(**record)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def write_all(self, records: Iterable[dict[str, Any]]) -> None:
        with self.path.open('w', encoding='utf-8') as handle:
            for record in records:
                payload = ordered_record(**record)
                handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
