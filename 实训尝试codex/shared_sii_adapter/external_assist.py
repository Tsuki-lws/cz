from __future__ import annotations

import json
import re
from typing import Any

from .compliance import assert_teacher_model_size_le_32b
from .llm_client import build_judge_client
from .types import AgentRunResult, RuntimeConfig


def _json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        payload = json.loads(match.group(0))
    return payload if isinstance(payload, dict) else {}


def external_assist_enabled(runtime: RuntimeConfig) -> bool:
    return bool(runtime.enable_external_assist and not runtime.benchmark_mode and not runtime.disable_reflection)


def organize_memory_with_external_model(
    *,
    runtime: RuntimeConfig,
    track_name: str,
    task: dict[str, Any],
    result: AgentRunResult,
    local_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not external_assist_enabled(runtime):
        return {}
    assert_teacher_model_size_le_32b(runtime.judge_model_name)
    compact_trace = []
    for row in result.trajectory[-8:]:
        compact_trace.append(
            {
                "role": row.get("role"),
                "tool": row.get("fn_name"),
                "content": str(row.get("content", ""))[:700],
            }
        )
    prompt = {
        "track": track_name,
        "question": task.get("instruction") or task.get("question") or task.get("query") or "",
        "prediction": result.pred,
        "local_signal": local_signal or {},
        "trajectory_tail": compact_trace,
        "instruction": (
            "Without using any gold answer, organize a reusable lesson for future similar tasks. "
            "Do not include case-specific hidden answers. Return JSON only with keys: "
            "useful (boolean), lesson (short string), retry_hint (short string), risk (short string)."
        ),
    }
    try:
        client = build_judge_client(runtime)
        resp = client.chat(
            [
                {"role": "system", "content": "You are a <=32B reflection and memory organizer. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            tools=None,
            max_tokens=runtime.external_assist_max_tokens,
            temperature=0.1,
            enable_thinking=False,
            response_format={"type": "json_object"},
        )
        payload = _json_from_text(resp.get("content", ""))
        if payload:
            payload["model"] = runtime.judge_model_name
        return payload
    except Exception as exc:  # noqa: BLE001
        return {"useful": False, "lesson": "", "retry_hint": "", "risk": "external_assist_unavailable", "error": str(exc)}


def build_reflection_hint_with_external_model(
    *,
    runtime: RuntimeConfig,
    track_name: str,
    task: dict[str, Any],
    result: AgentRunResult,
    local_signal: dict[str, Any],
) -> str:
    payload = organize_memory_with_external_model(
        runtime=runtime,
        track_name=track_name,
        task=task,
        result=result,
        local_signal=local_signal,
    )
    retry_hint = str(payload.get("retry_hint") or "").strip()
    if retry_hint:
        return retry_hint
    return str(payload.get("lesson") or "").strip()
