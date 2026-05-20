from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def normalize_2wiki(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "index": f"2wiki_{idx}",
        "source_dataset": "2wiki",
        "question": row.get("question") or "",
        "instruction": row.get("question") or "",
        "context": row.get("context") or "",
        "answer": row.get("answer") or "",
    }


def normalize_simplevqa(row: dict[str, Any], idx: int, simplevqa_root: Path) -> dict[str, Any]:
    image = str(row.get("image") or "")
    local_image = simplevqa_root / image
    out = {
        "index": f"simplevqa_{idx}",
        "source_dataset": "simpleVQA",
        "question": row.get("question") or "",
        "instruction": row.get("question") or "",
        "image": image,
        "image_url": row.get("image_url") or "",
        "answer": row.get("answer") or "",
    }
    if local_image.exists():
        out["image_path"] = str(local_image.resolve())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge local 2wiki and SimpleVQA files for Track-D runs.")
    parser.add_argument("--datasets-root", default="/inspire/qb-ilm2/project/26summer-camp-01/26210300/datasets")
    parser.add_argument("--output", default="/inspire/qb-ilm2/project/26summer-camp-01/26210300/datasets/trackd_2wiki_simplevqa.jsonl")
    parser.add_argument("--limit-2wiki", type=int, default=0)
    parser.add_argument("--limit-simplevqa", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.datasets_root)
    wiki_path = root / "2wiki.jsonl"
    simple_path = root / "simpleVQA" / "SimpleVQA.jsonl"
    simple_root = root / "simpleVQA"

    rows: list[dict[str, Any]] = []
    wiki_rows = read_jsonl(wiki_path)
    simple_rows = read_jsonl(simple_path)
    if args.limit_2wiki:
        wiki_rows = wiki_rows[: args.limit_2wiki]
    if args.limit_simplevqa:
        simple_rows = simple_rows[: args.limit_simplevqa]
    rows.extend(normalize_2wiki(row, idx) for idx, row in enumerate(wiki_rows))
    rows.extend(normalize_simplevqa(row, idx, simple_root) for idx, row in enumerate(simple_rows))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "rows": len(rows), "2wiki": len(wiki_rows), "simpleVQA": len(simple_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
