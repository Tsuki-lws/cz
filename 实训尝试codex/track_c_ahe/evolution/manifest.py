from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_manifest(record: dict[str, Any], path: str = "track_c_ahe/runs/manifest.jsonl") -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

