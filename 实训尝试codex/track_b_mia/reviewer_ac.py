from __future__ import annotations

import json
from typing import Any

from shared_sii_adapter.llm_client import build_judge_client
from shared_sii_adapter.types import RuntimeConfig


def review_without_gold(task: dict[str, Any], pred: str, trajectory: list[dict[str, Any]], runtime: RuntimeConfig) -> dict[str, Any]:
    if not pred.strip():
        return {"pass": False, "confidence": 0.0, "failure_type": "no_answer", "rationale": "empty answer"}
    client = build_judge_client(runtime)
    compact_trace = "\n".join(
        f"{r.get('role')}:{str(r.get('content'))[:500]}" for r in trajectory[-8:]
    )
    prompt = (
        "You are a no-gold answer confidence reviewer. Do not use or infer any hidden ground truth.\n"
        "Judge whether the answer is sufficiently supported by the trajectory evidence.\n"
        "Return JSON: {\"pass\": bool, \"confidence\": number, \"failure_type\": string, \"rationale\": string}.\n\n"
        f"Question: {task.get('instruction') or task.get('question')}\n"
        f"Prediction: {pred}\n"
        f"Recent trajectory:\n{compact_trace}"
    )
    try:
        resp = client.chat(
            [{"role": "user", "content": prompt}],
            tools=None,
            max_tokens=512,
            temperature=0.1,
            enable_thinking=False,
        )
        content = resp["content"].strip()
        return json.loads(content)
    except Exception as exc:  # noqa: BLE001
        return {"pass": True, "confidence": 0.5, "failure_type": "review_unavailable", "rationale": str(exc)}

