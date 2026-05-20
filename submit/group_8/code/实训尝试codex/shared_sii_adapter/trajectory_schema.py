from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


TRAJECTORY_FIELDS = (
    "timestamp",
    "step_id",
    "role",
    "content",
    "tool_call_id",
    "tool_calls",
    "reasoning_content",
    "total_tokens",
    "fn_name",
    "fn_args",
)


def ordered_record(**values: Any) -> dict[str, Any]:
    return {field: values.get(field) for field in TRAJECTORY_FIELDS}


def make_record(
    role: str,
    content: Any,
    *,
    step_id: int | None = None,
    tool_call_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "timestamp": time.time(),
        "step_id": step_id,
        "role": role,
        "content": content,
        "tool_call_id": tool_call_id,
    }
    payload.update(extra)
    return ordered_record(**payload)


def validate(record: dict[str, Any]) -> None:
    missing = [field for field in ("timestamp", "step_id", "role", "content", "tool_call_id") if field not in record]
    if missing:
        raise ValueError(f"trajectory record missing fields: {missing}")


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = ordered_record(**record)
            validate(payload)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def to_messages(records: list[dict[str, Any]], *, max_tool_chars: int = 6000) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in records:
        role = entry["role"]
        content = entry.get("content") or ""
        if role == "tool" and isinstance(content, str) and len(content) > max_tool_chars:
            content = content[:max_tool_chars] + f"\n\n...[truncated at {max_tool_chars} chars]"
        msg: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and entry.get("tool_calls"):
            msg["tool_calls"] = entry["tool_calls"]
        if entry.get("tool_call_id"):
            msg["tool_call_id"] = entry["tool_call_id"]
        messages.append(msg)
    return messages

