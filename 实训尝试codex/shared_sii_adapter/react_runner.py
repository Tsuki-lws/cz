from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from .llm_client import build_client
from .compliance import assert_no_gold_payload
from .tools import TOOLS_SCHEMA, dispatch_tool, parse_tool_args
from .trajectory_schema import make_record, to_messages, write_records
from .types import AgentRunResult, RuntimeConfig


DEFAULT_SYSTEM_PROMPT = """你是一个高效、严谨的任务执行 Agent，运行在配备搜索和浏览器工具的自动化框架中。

要求：
1. 需要证据时主动调用搜索或浏览器工具。
2. 图像相关任务优先使用输入图像在线链接调用 search_image；如果没有在线链接，再基于图像内容谨慎推理。
3. 避免重复同一个搜索 query 或重复访问同一个 URL。
4. 最终答案必须简洁，并尽量用 <answer>...</answer> 包裹。
"""


def extract_final_answer(content: str) -> str:
    content = content or ""
    if "<answer>" in content and "</answer>" in content:
        return content.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return content.strip()


def build_user_content(task: dict[str, Any]) -> Any:
    instruction = task.get("instruction") or task.get("question") or task.get("query") or ""
    image_b64 = task.get("image_b64")
    image_url = task.get("image_url") or task.get("image")
    if image_b64:
        parts: list[dict[str, Any]] = [{"type": "text", "text": str(instruction)}]
        if image_url:
            parts[0]["text"] += f"\nimage_url: {image_url}"
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        return parts
    image_path = task.get("image_path")
    if image_path and Path(image_path).exists():
        data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        parts = [{"type": "text", "text": str(instruction)}]
        if image_url:
            parts[0]["text"] += f"\nimage_url: {image_url}"
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        return parts
    if image_url:
        return f"{instruction}\nimage_url: {image_url}"
    return str(instruction)


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
    if memory_context:
        prompt += "\n\n可参考的历史经验：\n" + memory_context
    if reflection_context:
        prompt += "\n\n本轮修正策略：\n" + reflection_context

    records = [
        make_record("system", prompt, step_id=0),
        make_record("user", build_user_content(task), step_id=0),
    ]

    final_content = ""
    total_tokens = 0
    tool_calls_count = 0
    assistant_turns = 0

    for step in range(1, runtime.max_steps + 1):
        response = client.chat(
            to_messages(records),
            tools=None if runtime.disable_tools else TOOLS_SCHEMA,
            max_tokens=runtime.max_tokens,
            temperature=runtime.temperature,
            enable_thinking=runtime.enable_thinking,
        )
        content = response["content"] or ""
        tool_calls = response["tool_calls"] or []
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
        if not tool_calls and content:
            final_content = content
            break
        if not tool_calls:
            continue

        for call in tool_calls:
            fn = (call.get("function") or {}).get("name", "")
            args = parse_tool_args((call.get("function") or {}).get("arguments"))
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

    if not final_content:
        if records and records[-1].get("role") == "tool":
            records.append(
                make_record(
                    "user",
                    "请基于以上已有证据直接给出最终答案。不要再调用工具，答案必须简洁，并用 <answer>...</answer> 包裹。",
                    step_id=runtime.max_steps + 1,
                )
            )
            response = client.chat(
                to_messages(records),
                tools=None,
                max_tokens=min(runtime.max_tokens, 1024),
                temperature=runtime.temperature,
                enable_thinking=runtime.enable_thinking,
            )
            final_content = response["content"] or ""
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
            final_content = next((str(r.get("content") or "") for r in reversed(records) if r.get("role") == "assistant"), "")
    pred = extract_final_answer(final_content)
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
