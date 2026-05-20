from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from distill.common import write_json


_FINAL_ANSWER_RE = re.compile(r'"final_answer"\s*:\s*"(?P<answer>[^"\n{}]{1,300})', re.S)
_XML_FINAL_ANSWER_RE = re.compile(
    r"<function=final_answer>\s*(?P<body>.*?)\s*</function>",
    re.S,
)
_XML_PARAM_RE = re.compile(r"<parameter=(?:answer|final_answer)>\s*(?P<answer>.*?)\s*</parameter>", re.S)
_XML_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
_XML_FUNCTION_BLOCK_RE = re.compile(r"<function=[\w_]+>.*?</function>", re.S)


def sanitize_text(value: Any, *, max_chars: int = 4000) -> str:
    text = str(value or "").strip()
    text = re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", "[image]", text)
    text = _XML_TOOL_CALL_BLOCK_RE.sub("[tool call emitted via XML fallback; parsed separately]", text)
    text = _XML_FUNCTION_BLOCK_RE.sub("[tool call emitted via XML fallback; parsed separately]", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[:max_chars] + " ... [truncated]"
    return text


def parse_tool_calls(value: Any) -> str:
    if not value:
        return ""
    return sanitize_text(json.dumps(value, ensure_ascii=False), max_chars=2000)


def extract_final_from_reasoning(text: str) -> str:
    if not text:
        return ""
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return sanitize_text(match.group("answer"), max_chars=300)
    for block in _XML_FINAL_ANSWER_RE.finditer(text):
        param = _XML_PARAM_RE.search(block.group("body"))
        if param:
            return sanitize_text(param.group("answer"), max_chars=300)
    return ""


def extract_final_answer(row: dict[str, Any]) -> str:
    trajectory = row.get("trajectory") or []
    for record in reversed(trajectory):
        reasoning = str(record.get("reasoning_content") or "")
        answer = extract_final_from_reasoning(reasoning)
        if answer:
            return answer
    pred = sanitize_text(row.get("pred"), max_chars=300)
    if pred and not pred.endswith(("：", ":")):
        return pred
    for record in reversed(trajectory):
        content = sanitize_text(record.get("content"), max_chars=300)
        if content and not content.endswith(("：", ":")):
            return content
    return pred


def format_trajectory(trajectory: list[dict[str, Any]], *, max_records: int = 24) -> str:
    lines: list[str] = []
    for record in trajectory:
        role = str(record.get("role") or "")
        if role == "system":
            continue
        if role == "assistant" and record.get("tool_calls"):
            tool_calls = parse_tool_calls(record.get("tool_calls"))
            reasoning = sanitize_text(record.get("reasoning_content"), max_chars=1200)
            if reasoning:
                lines.append(f"[assistant reasoning] {reasoning}")
            lines.append(f"[assistant tool_calls] {tool_calls}")
        elif role == "tool":
            fn_name = str(record.get("fn_name") or "tool")
            content = sanitize_text(record.get("content"), max_chars=1400)
            ok = record.get("ok")
            ok_suffix = "" if ok is None else f" ok={bool(ok)}"
            lines.append(f"[tool:{fn_name}{ok_suffix}] {content}")
        elif role in {"user", "assistant"}:
            content = sanitize_text(record.get("content"), max_chars=1200)
            reasoning = sanitize_text(record.get("reasoning_content"), max_chars=1200)
            if content:
                lines.append(f"[{role}] {content}")
            elif role == "assistant" and reasoning:
                lines.append(f"[assistant reasoning] {reasoning}")
        if len(lines) >= max_records:
            break
    return "\n".join(lines)


def has_tool_trace(row: dict[str, Any]) -> bool:
    trajectory = row.get("trajectory") or []
    return any(record.get("tool_calls") or record.get("role") == "tool" for record in trajectory)


def to_alpaca(row: dict[str, Any]) -> dict[str, Any]:
    final_answer = extract_final_answer(row)
    trace = format_trajectory(row.get("trajectory") or [])
    output_parts = []
    if trace:
        output_parts.append("Track-D tool trajectory:\n" + trace)
    if final_answer:
        output_parts.append("Final answer:\n" + final_answer)
    return {
        "instruction": str(row.get("instruction") or "").strip(),
        "input": "",
        "output": "\n\n".join(output_parts).strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format Track-D trajectory JSONL into Alpaca SFT data with tool traces.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--require-tools", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    items = []
    skipped_no_tools = 0
    for row in rows:
        if args.require_tools and not has_tool_trace(row):
            skipped_no_tools += 1
            continue
        item = to_alpaca(row)
        if item["instruction"] and item["output"]:
            items.append(item)

    write_json(args.output, items)
    report = {
        "input": args.input,
        "output": args.output,
        "input_samples": len(rows),
        "output_samples": len(items),
        "skipped_no_tools": skipped_no_tools,
        "tool_trace_samples": sum(1 for row in rows if has_tool_trace(row)),
    }
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
