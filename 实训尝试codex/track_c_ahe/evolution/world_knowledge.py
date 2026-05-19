from __future__ import annotations

from pathlib import Path


WORLD_KNOWLEDGE_PATH = Path("track_c_ahe/harness_components/world_knowledge.md")


def load_world_knowledge() -> str:
    if not WORLD_KNOWLEDGE_PATH.exists():
        return ""
    return WORLD_KNOWLEDGE_PATH.read_text(encoding="utf-8")


def append_world_knowledge(note: str, *, allow_update: bool) -> None:
    if not allow_update:
        return
    WORLD_KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WORLD_KNOWLEDGE_PATH.open("a", encoding="utf-8") as handle:
        handle.write("\n\n" + note.strip() + "\n")

