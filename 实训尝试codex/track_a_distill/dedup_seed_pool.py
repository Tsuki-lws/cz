from __future__ import annotations

import argparse
import json
from pathlib import Path


def norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def load_questions(path: str) -> set[str]:
    if not path:
        return set()
    out: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out.add(norm(row.get("question") or row.get("instruction") or ""))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple exact-question dedup for Track A seeds.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--against", action="append", default=[])
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    blocked: set[str] = set()
    for path in args.against:
        blocked |= load_questions(path)
    kept = []
    removed = []
    seen: set[str] = set()
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = norm(row.get("question", ""))
            if not key or key in seen or key in blocked:
                removed.append(row)
                continue
            seen.add(key)
            kept.append(row)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(json.dumps({"kept": len(kept), "removed": len(removed)}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

