from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    resolved = ensure_parent(path)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_table(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if resolved.suffix.lower() == ".jsonl":
        return read_jsonl(resolved)
    if resolved.suffix.lower() == ".json":
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        raise ValueError(f"unsupported json dataset shape: {resolved}")
    csv.field_size_limit(sys.maxsize)
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    resolved = ensure_parent(path)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
