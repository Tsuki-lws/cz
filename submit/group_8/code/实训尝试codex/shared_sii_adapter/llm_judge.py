from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from statistics import mean
from typing import Any

from .compliance import assert_teacher_model_size_le_32b
from .llm_client import build_judge_client
from .types import AgentRunResult, RuntimeConfig


JUDGE_SYSTEM_PROMPT = """You are a careful answer equivalence judge for VQA/open-domain QA.

Decide whether the prediction correctly answers the question using the gold answer as reference.
Allow semantically equivalent wording, translations, aliases, abbreviations, and harmless formatting differences.
Accept date/year variants such as "1934" and "1934年".
Mark incorrect if the prediction is missing, ambiguous, partially correct, broader/narrower than the gold, or a different entity.

Return only JSON:
{"correct": true/false, "confidence": 0.0-1.0, "rationale": "short reason"}"""


def _json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty judge response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("judge response is not an object")
    return payload


def judge_prediction(
    *,
    runtime: RuntimeConfig,
    question: str,
    gold: str,
    pred: str,
    max_tokens: int = 512,
) -> dict[str, Any]:
    if not str(pred or "").strip():
        return {"correct": False, "confidence": 1.0, "rationale": "empty prediction", "error": ""}
    assert_teacher_model_size_le_32b(runtime.judge_model_name)
    client = build_judge_client(runtime)
    prompt = f"Question:\n{question}\n\nGold answer:\n{gold}\n\nPrediction:\n{pred}\n"
    try:
        resp = client.chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            max_tokens=max_tokens,
            temperature=0.0,
            enable_thinking=False,
            response_format={"type": "json_object"},
        )
        payload = _json_from_text(resp.get("content", ""))
        return {
            "correct": bool(payload.get("correct")),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "rationale": str(payload.get("rationale", "")),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"correct": False, "confidence": 0.0, "rationale": "", "error": str(exc)}


def summarize_with_llm_judge(
    results: list[AgentRunResult],
    gold_rows_by_index: dict[str, dict[str, Any]],
    runtime: RuntimeConfig,
    *,
    concurrency: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [result for result in results if result.index in gold_rows_by_index and gold_rows_by_index[result.index].get("answer")]
    judged: list[dict[str, Any] | None] = [None] * len(rows)

    def work(result: AgentRunResult) -> dict[str, Any]:
        gold_row = gold_rows_by_index[result.index]
        judge = judge_prediction(
            runtime=runtime,
            question=str(gold_row.get("instruction") or gold_row.get("question") or ""),
            gold=str(gold_row.get("answer") or ""),
            pred=result.pred,
        )
        return {
            "index": result.index,
            "instruction": result.instruction,
            "answer": str(gold_row.get("answer") or ""),
            "pred": result.pred,
            "llm_correct": judge["correct"],
            "llm_confidence": judge["confidence"],
            "llm_rationale": judge["rationale"],
            "judge_error": judge["error"],
        }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        future_to_idx = {pool.submit(work, result): idx for idx, result in enumerate(rows)}
        for future in as_completed(future_to_idx):
            judged[future_to_idx[future]] = future.result()

    judge_rows = [row for row in judged if row is not None]
    ok_rows = [row for row in judge_rows if not row.get("judge_error")]
    summary = {
        "llm_judge_count": len(judge_rows),
        "llm_judge_error_count": len(judge_rows) - len(ok_rows),
        "llm_accuracy": mean([bool(row.get("llm_correct")) for row in ok_rows]) if ok_rows else None,
        "llm_avg_confidence": mean([float(row.get("llm_confidence") or 0.0) for row in ok_rows]) if ok_rows else 0.0,
        "llm_judge_model": runtime.judge_model_name,
    }
    return summary, judge_rows
