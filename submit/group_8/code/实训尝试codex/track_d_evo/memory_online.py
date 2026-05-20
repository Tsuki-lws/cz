from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class OnlineMemory:
    def __init__(self, path: str = "experiments/track_d/online_memory.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def retrieve(self, k: int = 3) -> str:
        if not self.path.exists():
            return ""
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return "\n".join(str(row.get("lesson", "")) for row in rows[-k:])

    def update(self, lesson: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(lesson, ensure_ascii=False) + "\n")

