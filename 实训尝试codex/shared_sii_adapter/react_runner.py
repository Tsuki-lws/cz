from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

from .llm_client import build_client
from .compliance import assert_no_gold_payload
from .tools import FINAL_ANSWER_TOOL_NAMES, TOOL_FN_MAP, TOOLS_SCHEMA, dispatch_tool, normalize_tool_name, parse_tool_args
from .trajectory_schema import make_record, to_messages, write_records
from .types import AgentRunResult, RuntimeConfig


DEFAULT_SYSTEM_PROMPT = """你是一个高效、严谨的任务执行 Agent，运行在配备搜索和沙盒浏览器工具的自动化框架中。

核心要求：
1. 只能使用系统提供的工具：search_text、search_image、browser_navigate、browser_get_text、browser_parallel、browser_click、browser_type。
2. 工具只能通过系统提供的原生 tool call 机制调用；不要在文本里写 <tool_call>、<function=...>、JSON 工具调用或任何伪造函数名。
3. 每一步只能二选一：要么调用真实工具，要么直接给出最终答案；不要输出空内容、占位符、半截 JSON 或只有格式没有答案的内容。
4. 当你已经得到足够证据时，立即停止调用工具并回答；避免重复同一个搜索 query、重复访问同一个 URL。
5. 最终答案应直接回答问题所问，保留必要限定词；不要输出推理过程、证据列表、来源说明或工具调用文本。
6. 如果收尾阶段要求 JSON，只返回 {"final_answer":"答案"}，不要附加任何其他文本。

工具使用准则：
1. search_text 返回 [{rank,title,url,snippet,content}]，适合查实体、年份、国籍、地点、两跳事实和候选网页。
2. search_image 是图搜文，输入优先使用当前样本本地图片路径或当前图片本身；不要依赖可能过期的临时图片 URL。
3. browser_navigate 打开候选 URL 并返回文本预览；browser_get_text 获取当前页正文。
4. browser_parallel 可并发打开多个候选 URL，用于比较多个搜索结果，单个 URL 失败不影响其他。
5. 若工具返回错误，先分析 error；同类操作最多重试 1 次，仍失败就换 search_text query 或浏览其他候选页面。
"""

_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[\w_]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>",
    re.S,
)
_XML_PARAM_RE = re.compile(r"<parameter=(?P<key>[\w_]+)>\s*(?P<value>.*?)\s*</parameter>", re.S)
_PARTIAL_FINAL_ANSWER_RE = re.compile(r'"final_answer"\s*:\s*"(?P<answer>[^"\n{}]{1,160})', re.S)


def extract_final_answer(content: str) -> str:
    content = content or ""
    stripped = content.strip()
    if not stripped or stripped.startswith("[HARNESS]"):
        return ""
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                answer = payload.get("final_answer")
                if answer is not None:
                    return str(answer).strip()
        except json.JSONDecodeError:
            pass
    partial = _PARTIAL_FINAL_ANSWER_RE.search(stripped)
    if partial:
        return partial.group("answer").strip()
    if "<answer>" in content and "</answer>" in content:
        return content.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    bold_matches = re.findall(r"\*\*([^*\n]{1,160})\*\*", stripped)
    if bold_matches:
        return bold_matches[-1].strip().strip(" .。；;，,")
    patterns = [
        re.compile(r"(?:final answer|answer is|answer|答案是|答案为|最终答案)[:：]\s*([^\n。]+)", re.I),
        re.compile(r"(?:so|therefore),?\s+the answer is\s+([^\n。]+)", re.I),
        re.compile(r"(?:死于|死因是|死因：|死因为)\s*([^，,。；;\n]{1,80})", re.I),
        re.compile(r"(?:起源于|源于|来自|from)\s*([^，,。；;\n]{1,80})", re.I),
        re.compile(r"(?:全长|长度为|长约)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:千米|公里|km|kilometers?)?", re.I),
    ]
    for pattern in patterns:
        matches = pattern.findall(stripped)
        if matches:
            answer = matches[-1].strip().strip(" .。；;，,")
            if answer:
                return answer
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    if first.startswith("根据") and len(lines) > 1:
        return lines[-1].strip()
    return first.strip()


def is_raw_tool_call_text(content: str) -> bool:
    return "<tool_call>" in (content or "") or "<function=" in (content or "")


def is_direct_answer_text(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
            return isinstance(payload, dict) and isinstance(payload.get("final_answer"), str)
        except json.JSONDecodeError:
            return False
    if "<answer>" in text and "</answer>" in text:
        return True
    if is_raw_tool_call_text(text):
        return False
    return len(extract_final_answer(text)) <= 160 and "\n" not in text.strip("\n")


def select_previous_answer(records: list[dict[str, Any]]) -> str:
    for record in reversed(records):
        if record.get("role") != "assistant":
            continue
        content = str(record.get("content") or "").strip()
        if content and not is_raw_tool_call_text(content):
            return content
    for record in reversed(records):
        if record.get("role") != "assistant":
            continue
        answer = extract_answer_from_reasoning(str(record.get("reasoning_content") or ""))
        if answer:
            return answer
    return ""


def extract_answer_from_reasoning(reasoning: str) -> str:
    text = str(reasoning or "").strip()
    if not text:
        return ""
    answer = extract_structured_answer_from_reasoning(text)
    if answer:
        return answer
    patterns = [
        r"最终答案[：:]\s*(?P<answer>[^\n。；;]+)",
        r"答案[：:]\s*(?P<answer>[^\n。；;]+)",
        r"这个图片(?:显示的)?是\s*(?P<answer>[^\n。；;]+)",
        r"这张图片(?:显示的)?是\s*(?P<answer>[^\n。；;]+)",
        r"this image (?:is|shows|depicts)\s*(?P<answer>[^\n.]+)",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.I))
        if not matches:
            continue
        answer = matches[-1].group("answer").strip(" *，,。.;；:：\"'`")
        if 0 < len(answer) <= 160:
            return answer
    return ""


def extract_structured_answer_from_reasoning(reasoning: str) -> str:
    text = str(reasoning or "").strip()
    if "final_answer" not in text:
        return ""
    return extract_final_answer(text)


def _coerce_xml_value(value: str) -> Any:
    text = value.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        return int(text)
    except ValueError:
        return text


def extract_xml_tool_calls(content: str, reasoning_content: str = "") -> list[dict[str, Any]]:
    text = "\n".join(part for part in [content or "", reasoning_content or ""] if part)
    calls: list[dict[str, Any]] = []
    for idx, match in enumerate(_XML_TOOL_CALL_RE.finditer(text)):
        args = {}
        for param in _XML_PARAM_RE.finditer(match.group("body")):
            args[param.group("key").strip()] = _coerce_xml_value(param.group("value"))
        calls.append(
            {
                "id": f"xml_tool_call_{idx}",
                "type": "function",
                "function": {
                    "name": match.group("name").strip(),
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return calls


def final_answer_from_tool_call(call: dict[str, Any], content: str = "") -> str:
    args = parse_tool_args((call.get("function") or {}).get("arguments"))
    if args.get("first_name") and args.get("last_name"):
        return f"{args['first_name']} {args['last_name']}".strip()
    for key in ("final_answer", "answer", "result", "message", "value", "text"):
        value = args.get(key)
        if value:
            return str(value).strip()
    if args and len(args) == 1:
        key, value = next(iter(args.items()))
        if value and str(key).lower() not in {"query", "url", "image"}:
            return str(value).strip()
        if key and str(key).lower() not in {"query", "url", "image"}:
            return str(key).strip()
    return extract_final_answer(content)


def compress_image_b64(image_b64: str, *, max_side: int = 1280, quality: int = 88) -> str:
    return image_b64


def compact_records_for_model(records: list[dict[str, Any]], *, max_chars: int = 120000) -> list[dict[str, Any]]:
    if not records:
        return records
    compacted = list(records)
    total = sum(len(str(record.get("content") or "")) + len(str(record.get("reasoning_content") or "")) for record in compacted)
    if total <= max_chars:
        return compacted
    protected = compacted[:2]
    tail = compacted[2:]
    kept: list[dict[str, Any]] = []
    budget = max_chars - sum(len(str(record.get("content") or "")) for record in protected)
    for record in reversed(tail):
        size = len(str(record.get("content") or "")) + len(str(record.get("reasoning_content") or ""))
        if kept and budget - size < 0:
            break
        kept.append(record)
        budget -= size
    return protected + list(reversed(kept))


def build_user_content(task: dict[str, Any]) -> Any:
    instruction = task.get("instruction") or task.get("question") or task.get("query") or ""
    image_b64 = task.get("image_b64")
    image_url = task.get("image_url") or task.get("image")
    image_url_for_text = str(image_url) if str(image_url or "").startswith(("http://", "https://")) else ""
    image_path = task.get("image_path")
    if image_b64:
        image_b64 = compress_image_b64(str(image_b64))
        parts: list[dict[str, Any]] = [{"type": "text", "text": str(instruction)}]
        if image_path:
            parts[0]["text"] += f"\nlocal_image_path: {image_path}"
        if image_url_for_text:
            parts[0]["text"] += f"\nimage_url: {image_url_for_text}"
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        return parts
    if image_path and Path(image_path).exists():
        data = compress_image_b64(base64.b64encode(Path(image_path).read_bytes()).decode("utf-8"))
        parts = [{"type": "text", "text": str(instruction)}]
        parts[0]["text"] += f"\nlocal_image_path: {image_path}"
        if image_url_for_text:
            parts[0]["text"] += f"\nimage_url: {image_url_for_text}"
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        return parts
    if image_url:
        return f"{instruction}\nimage_url: {image_url}"
    return str(instruction)


def resolve_search_image_arg(args: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    args = dict(args or {})
    image_path = str(task.get("image_path") or "").strip()
    image_b64 = str(task.get("image_b64") or "").strip()
    if image_path and Path(image_path).exists():
        args["image"] = image_path
        args.pop("image_url", None)
        return args
    if image_b64:
        args["image"] = image_b64
        args.pop("image_url", None)
    return args


def run_react_task(
    task: dict[str, Any],
    runtime: RuntimeConfig,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    memory_context: str = "",
    reflection_context: str = "",
    track_name: str | None = None,
) -> AgentRunResult:
    assert_no_gold_payload(task)
    started = time.perf_counter()
    client = build_client(runtime)
    index = str(task.get("index") or task.get("id") or "")
    instruction = str(task.get("instruction") or task.get("question") or task.get("query") or "")
    image = str(task.get("image") or "")
    image_url = str(task.get("image_url") or "")
    prompt = system_prompt
    if runtime.disable_tools:
        prompt += "\n\n当前运行已关闭工具。你不能调用工具，也不能在文本中写任何工具调用；每一步必须直接给出最终答案。"
    if memory_context:
        prompt += "\n\n可参考的历史经验：\n" + memory_context
    if reflection_context:
        prompt += "\n\n本轮修正策略：\n" + reflection_context

    records = [
        make_record("system", prompt, step_id=0),
        make_record("user", build_user_content(task), step_id=0),
    ]

    final_content = ""
    final_step_reason = ""
    total_tokens = 0
    tool_calls_count = 0
    assistant_turns = 0
    search_text_queries: dict[str, int] = {}
    tool_name_counts: dict[str, int] = {}

    for step in range(1, runtime.max_steps + 1):
        response = client.chat(
            to_messages(compact_records_for_model(records)),
            tools=None if runtime.disable_tools else TOOLS_SCHEMA,
            max_tokens=runtime.max_tokens,
            temperature=runtime.temperature,
            enable_thinking=runtime.enable_thinking,
        )
        content = response["content"] or ""
        reasoning_answer = extract_structured_answer_from_reasoning(response.get("reasoning_content") or "")
        tool_calls = response["tool_calls"] or []
        if not content.strip() and reasoning_answer:
            content = reasoning_answer
            tool_calls = []
        elif runtime.enable_xml_tool_fallback and not tool_calls:
            tool_calls = extract_xml_tool_calls(content, response.get("reasoning_content") or "")
        if not content.strip() and not tool_calls:
            extracted = extract_answer_from_reasoning(response.get("reasoning_content") or "")
            content = extracted or ""
        total_tokens += int(response.get("total_tokens") or 0)
        assistant_turns += 1
        records.append(
            make_record(
                "assistant",
                content,
                step_id=step,
                tool_calls=tool_calls or None,
                reasoning_content=response.get("reasoning_content") or None,
                total_tokens=response.get("total_tokens") or None,
            )
        )
        if tool_calls:
            filtered_tool_calls = []
            for call in tool_calls:
                fn = normalize_tool_name((call.get("function") or {}).get("name", ""))
                if fn in FINAL_ANSWER_TOOL_NAMES:
                    final_content = final_answer_from_tool_call(call, content)
                    filtered_tool_calls = []
                    break
                filtered_tool_calls.append(call)
            tool_calls = filtered_tool_calls
            if final_content and not tool_calls:
                records[-1]["tool_calls"] = None
        if not tool_calls and content and not is_raw_tool_call_text(content):
            final_content = content
            final_step_reason = "assistant_finished_without_tool_calls"
            break
        if not tool_calls and is_raw_tool_call_text(content):
            final_step_reason = "raw_tool_call_text_without_native_call"
            final_content = ""
            break
        if not tool_calls and not content.strip():
            final_step_reason = "empty_assistant_message"
            break
        if not tool_calls:
            continue

        for call in tool_calls:
            fn = (call.get("function") or {}).get("name", "")
            normalized_fn = normalize_tool_name(fn)
            args = parse_tool_args((call.get("function") or {}).get("arguments"))
            if normalized_fn in FINAL_ANSWER_TOOL_NAMES:
                final_content = final_answer_from_tool_call(call, content)
                final_step_reason = "final_answer_tool_call"
                break
            if normalized_fn not in TOOL_FN_MAP:
                records.append(
                    make_record(
                        "tool",
                        f"[SKIPPED] Unknown tool '{fn}'. Use only the provided tools.",
                        step_id=step,
                        tool_call_id=call.get("id"),
                        fn_name=fn,
                        fn_args=args,
                        ok=False,
                    )
                )
                continue
            if normalized_fn == "search_image":
                args = resolve_search_image_arg(args, task)
            if normalized_fn == "search_text":
                query = " ".join(str(args.get("query") or "").lower().split())
                search_text_queries[query] = search_text_queries.get(query, 0) + 1
                tool_name_counts[normalized_fn] = tool_name_counts.get(normalized_fn, 0) + 1
                if search_text_queries[query] > 3:
                    result = {
                        "ok": False,
                        "content": "[SKIPPED] Duplicate search_text query. Use existing evidence or try a meaningfully different query.",
                        "latency_ms": 0,
                    }
                elif tool_name_counts[normalized_fn] > 24:
                    result = {
                        "ok": False,
                        "content": "[SKIPPED] search_text budget exhausted for this task. Use gathered evidence and answer.",
                        "latency_ms": 0,
                    }
                else:
                    result = dispatch_tool(fn, args)
            else:
                tool_name_counts[normalized_fn] = tool_name_counts.get(normalized_fn, 0) + 1
                result = dispatch_tool(fn, args)
            tool_calls_count += 1
            records.append(
                make_record(
                    "tool",
                    result["content"],
                    step_id=step,
                    tool_call_id=call.get("id"),
                    fn_name=fn,
                    fn_args=args,
                    total_tokens=None,
                    tool_calls=None,
                    reasoning_content=None,
                    latency_ms=result.get("latency_ms"),
                    ok=result.get("ok"),
                )
            )

    needs_final_answer_step = bool(records) and (
        records[-1].get("role") == "tool"
        or not final_content
        or (final_step_reason == "assistant_finished_without_tool_calls" and not is_direct_answer_text(final_content))
    )
    if runtime.structured_final_answer and needs_final_answer_step:
        correction = ""
        if final_step_reason == "raw_tool_call_text_without_native_call":
            correction = "上一条输出是无效的伪工具调用，必须忽略它。"
        elif final_step_reason == "empty_assistant_message":
            correction = "上一条输出为空，必须重新给出非空答案。"
        records.append(
            make_record(
                "user",
                correction
                + "请基于题目和图片直接给出最终答案。不要再调用工具，不要写工具调用文本。只返回 JSON：{\"final_answer\":\"你的答案\"}。答案字段只写最短答案，不要解释、不要 Markdown、不要 <answer> 标签。",
                step_id=runtime.max_steps + 1,
            )
        )
        response = client.chat(
            to_messages(compact_records_for_model(records)),
            tools=None,
            max_tokens=min(runtime.max_tokens, 512),
            temperature=0.1,
            enable_thinking=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "final_answer_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "final_answer": {"type": "string"}
                        },
                        "required": ["final_answer"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        final_content = response["content"] or ""
        if not final_content.strip():
            final_content = extract_answer_from_reasoning(response.get("reasoning_content") or "")
        if is_raw_tool_call_text(final_content):
            final_content = ""
        total_tokens += int(response.get("total_tokens") or 0)
        assistant_turns += 1
        records.append(
            make_record(
                "assistant",
                final_content,
                step_id=runtime.max_steps + 1,
                reasoning_content=response.get("reasoning_content") or None,
                total_tokens=response.get("total_tokens") or None,
            )
        )
    if not final_content:
        records.append(
            make_record(
                "user",
                "请基于以上已有证据直接给出最终答案。不要再调用工具，只输出答案本身；如果证据不完整，也必须给出最可能的简短答案。",
                step_id=runtime.max_steps + 1,
            )
        )
        response = client.chat(
            to_messages(compact_records_for_model(records)),
            tools=None,
            max_tokens=min(runtime.max_tokens, 256),
            temperature=0.1,
            enable_thinking=False,
        )
        final_content = response["content"] or ""
        if is_raw_tool_call_text(final_content):
            final_content = ""
        total_tokens += int(response.get("total_tokens") or 0)
        assistant_turns += 1
        records.append(
            make_record(
                "assistant",
                final_content,
                step_id=runtime.max_steps + 1,
                reasoning_content=response.get("reasoning_content") or None,
                total_tokens=response.get("total_tokens") or None,
            )
        )
        if not final_content:
            final_content = select_previous_answer(records)
        if not final_content:
            final_content = next(
                (
                    extract_answer_from_reasoning(str(r.get("reasoning_content") or ""))
                    for r in reversed(records)
                    if r.get("role") == "assistant"
                    and extract_answer_from_reasoning(str(r.get("reasoning_content") or ""))
                ),
                "",
            )
    pred = extract_final_answer(final_content)
    if is_raw_tool_call_text(pred):
        pred = extract_final_answer(select_previous_answer(records))
    elapsed = time.perf_counter() - started
    out_dir = Path(runtime.output_dir) / (track_name or runtime.track_name) / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"{index or int(started * 1000)}.jsonl"
    write_records(traj_path, records)
    return AgentRunResult(
        index=index,
        instruction=instruction,
        image=image,
        image_url=image_url,
        pred=pred,
        trajectory=records,
        metrics={
            "tokens": total_tokens,
            "turns": assistant_turns,
            "tool_calls": tool_calls_count,
            "latency": elapsed,
            "trajectory_path": str(traj_path),
        },
        debug={"track": track_name or runtime.track_name},
    )
