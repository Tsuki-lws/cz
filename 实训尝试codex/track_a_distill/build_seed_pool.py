from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def as_seed(row: dict[str, Any], source: str, idx: int) -> dict[str, Any]:
    return {
        "id": f"{source}_{idx}",
        "source": source,
        "task_type": row.get("task_type", "unknown"),
        "question": row.get("question") or row.get("instruction") or row.get("query") or "",
        "image": row.get("image", ""),
        "image_url": row.get("image_url", ""),
        "answer": row.get("answer", ""),
        "evidence": row.get("evidence", []),
        "metadata": {k: v for k, v in row.items() if k not in {"question", "instruction", "query", "image", "image_url", "answer", "evidence"}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize local jsonl datasets into Track A seed schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if line.strip():
                rows.append(as_seed(json.loads(line), args.source, idx))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

