from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from distill.common import write_json
from distill.data_collection.format_trackd_sft import has_tool_trace, to_alpaca


ALLOWED_TOOLS = {
    "search_text",
    "search_image",
    "browser_navigate",
    "browser_get_text",
    "browser_parallel",
    "browser_click",
    "browser_type",
}

BAD_OUTPUT_PATTERNS = [
    r"data:image",
    r";base64,",
    r"<tool_call>",
    r"<function=",
    r"\[SKIPPED\] Unknown tool",
]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tool_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in row.get("trajectory") or [] if record.get("role") == "tool"]


def assistant_tool_call_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for record in row.get("trajectory") or []:
        for call in record.get("tool_calls") or []:
            names.append(str((call.get("function") or {}).get("name") or ""))
    return names


def tool_is_error(record: dict[str, Any]) -> bool:
    content = str(record.get("content") or "")
    return bool(record.get("ok") is False or content.startswith("[ERROR]") or content.startswith("[SKIPPED]"))


def bad_prediction(pred: Any) -> bool:
    text = normalize_text(pred)
    if not text:
        return True
    if text.endswith((":", "：")):
        return True
    if len(text) > 240:
        return True
    bad_prefixes = (
        "现在我已经确认",
        "根据搜索结果",
        "based on the search",
        "based on the information",
    )
    return text.lower().startswith(bad_prefixes)


def should_drop(row: dict[str, Any], *, max_error_rate: float, max_error_count: int, max_output_chars: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not has_tool_trace(row):
        reasons.append("no_tool_trace")

    tools = tool_records(row)
    if not tools:
        reasons.append("no_tool_observation")
    unknown_tools = [name for name in assistant_tool_call_names(row) + [str(r.get("fn_name") or "") for r in tools] if name and name not in ALLOWED_TOOLS]
    if unknown_tools:
        reasons.append("unknown_tool")

    error_count = sum(1 for record in tools if tool_is_error(record))
    if tools and error_count / len(tools) > max_error_rate:
        reasons.append("high_tool_error_rate")
    if error_count > max_error_count:
        reasons.append("too_many_tool_errors")

    for record in tools:
        if str(record.get("fn_name") or "") == "search_text":
            args = record.get("fn_args") or {}
            if not normalize_text(args.get("query")):
                reasons.append("empty_search_query")
                break

    if bad_prediction(row.get("pred")):
        reasons.append("bad_prediction")

    item = to_alpaca(row)
    output = item.get("output", "")
    if not normalize_text(item.get("instruction")) or not normalize_text(output):
        reasons.append("empty_sft_field")
    if len(output) > max_output_chars:
        reasons.append("output_too_long")
    if any(re.search(pattern, output, flags=re.IGNORECASE) for pattern in BAD_OUTPUT_PATTERNS):
        reasons.append("bad_output_pattern")

    return bool(reasons), sorted(set(reasons))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Track-D SFT data from raw trajectory JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dropped-output", default="")
    parser.add_argument("--max-error-rate", type=float, default=0.4)
    parser.add_argument("--max-error-count", type=int, default=1)
    parser.add_argument("--max-output-chars", type=int, default=18000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    reasons_counter: Counter[str] = Counter()

    for row in rows:
        drop, reasons = should_drop(
            row,
            max_error_rate=args.max_error_rate,
            max_error_count=args.max_error_count,
            max_output_chars=args.max_output_chars,
        )
        if drop:
            for reason in reasons:
                reasons_counter[reason] += 1
            dropped.append({"index": row.get("index"), "pred": row.get("pred"), "reasons": reasons})
            continue
        kept.append(to_alpaca(row))

    write_json(args.output, kept)
    if args.dropped_output:
        write_json(args.dropped_output, dropped)

    report = {
        "input": args.input,
        "output": args.output,
        "input_samples": len(rows),
        "kept_samples": len(kept),
        "removed_samples": len(dropped),
        "tool_trace_samples": sum(1 for row in rows if has_tool_trace(row)),
        "reasons": dict(reasons_counter),
        "max_error_rate": args.max_error_rate,
        "max_error_count": args.max_error_count,
        "max_output_chars": args.max_output_chars,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
