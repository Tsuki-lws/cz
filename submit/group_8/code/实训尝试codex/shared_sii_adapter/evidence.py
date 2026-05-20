from __future__ import annotations

import json
from typing import Any


def compact_text(text: Any, max_chars: int = 4000) -> str:
    if isinstance(text, (dict, list)):
        text = json.dumps(text, ensure_ascii=False)
    text = "" if text is None else str(text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated at {max_chars} chars]"
    return text


def compact_tool_result(result: Any, max_chars: int = 8000) -> str:
    return compact_text(result, max_chars=max_chars)

