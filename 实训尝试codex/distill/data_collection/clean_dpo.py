from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BAD_CHOSEN_PATTERNS = [
    r"\bI\s+(?:am\s+)?sorry\b",
    r"\bI\s+can(?:not|'t)\b",
    r"\bunable to\b",
    r"\bplease provide\b",
    r"\bgold answer\b",
    r"\bcorrect final\b",
    r"\bas an ai\b",
    r"```",
    r"<answer>",
]

BAD_INSTRUCTION_PATTERNS = [
    r"\bpdf\b",
    r"\bspeech\b",
    r"\bessay\b",
    r"\blyrics?\b",
    r"\bwords of the song\b",
    r"\blist of\b",
    r"\bpresent it in the class\b",
]

PREFIX_PATTERNS = [
    r"^The correct answer is:\s*",
    r"^The answer is:\s*",
    r"^Correct answer:\s*",
    r"^Answer:\s*",
]


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    for pattern in PREFIX_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text


def has_bad_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def clean_item(item: dict[str, Any], *, max_chosen_chars: int, max_rejected_chars: int) -> tuple[dict[str, Any] | None, str]:
    instruction = normalize_text(item.get("instruction", ""))
    chosen = normalize_text(item.get("chosen", ""))
    rejected = normalize_text(item.get("rejected", ""))

    if not instruction or not chosen or not rejected:
        return None, "empty_field"
    if chosen.casefold() == rejected.casefold():
        return None, "same_chosen_rejected"
    if len(chosen) > max_chosen_chars:
        return None, "chosen_too_long"
    if len(rejected) > max_rejected_chars:
        return None, "rejected_too_long"
    if has_bad_pattern(chosen, BAD_CHOSEN_PATTERNS):
        return None, "bad_chosen_pattern"
    if has_bad_pattern(instruction, BAD_INSTRUCTION_PATTERNS):
        return None, "bad_instruction_pattern"
    serialized = json.dumps(item, ensure_ascii=False)
    if "data:image" in serialized or "base64" in serialized:
        return None, "embedded_image_payload"

    cleaned = {
        "instruction": instruction,
        "input": normalize_text(item.get("input", "")),
        "chosen": chosen,
        "rejected": rejected,
    }
    images = item.get("images")
    if images:
        cleaned["images"] = images
    return cleaned, "kept"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean LLaMA-Factory DPO JSON data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-chosen-chars", type=int, default=1200)
    parser.add_argument("--max-rejected-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(Path(args.input).read_text())
    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for item in items:
        cleaned, reason = clean_item(
            item,
            max_chosen_chars=args.max_chosen_chars,
            max_rejected_chars=args.max_rejected_chars,
        )
        reasons[reason] += 1
        if cleaned is not None:
            kept.append(cleaned)

    report = {
        "input": args.input,
        "output": args.output,
        "input_samples": len(items),
        "kept_samples": len(kept),
        "removed_samples": len(items) - len(kept),
        "image_samples": sum(1 for item in kept if item.get("images")),
        "reasons": dict(reasons),
        "max_chosen_chars": args.max_chosen_chars,
        "max_rejected_chars": args.max_rejected_chars,
    }

    Path(args.output).write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
