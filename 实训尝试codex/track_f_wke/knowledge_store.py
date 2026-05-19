from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class KnowledgeStore:
    def __init__(self, path: str = "track_f_wke/knowledge_store/world_knowledge.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        terms = {token.lower() for token in str(query).split() if len(token) > 2}
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        scored = []
        for row in rows:
            text = json.dumps(row, ensure_ascii=False).lower()
            score = sum(1 for term in terms if term in text)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for score, row in scored[:k] if score > 0] or rows[-k:]


def summarize_hits(hits: list[dict[str, Any]]) -> str:
    lines = []
    for item in hits:
        topic = item.get("topic", "unknown")
        lesson = item.get("lesson", "")
        lines.append(f"- {topic}: {lesson}")
    return "\n".join(lines)

