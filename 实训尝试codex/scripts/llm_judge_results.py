from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
from statistics import mean
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared_sii_adapter.io_utils import read_jsonl, read_table, write_json, write_jsonl
from shared_sii_adapter.llm_client import SGLangClient
from shared_sii_adapter.run_dataset import DEFAULT_QWEN32B_BASE_URL, normalize_task


TRACKS = ["track_a", "track_b", "track_c", "track_d", "track_e", "track_f"]


JUDGE_SYSTEM_PROMPT = """You are a careful answer equivalence judge for VQA/open-domain QA.

You will receive a question, the gold answer, and a model prediction.
Decide whether the prediction correctly answers the question.

Rules:
- Use the gold answer as the reference, but allow semantically equivalent wording.
- Accept harmless formatting differences, articles, punctuation, whitespace, and casing.
- Accept date/year formatting variants, such as "1934" and "1934年".
- Accept common aliases, translations, abbreviations, and names in another language only when they clearly refer to the same entity.
- Mark incorrect if the prediction is missing, ambiguous, only partially correct, broader/narrower than the gold answer, or names a different entity.
- Do not reward extra explanation if the final answer itself is wrong.

Return only JSON:
{"correct": true/false, "confidence": 0.0-1.0, "rationale": "short reason"}"""


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").lower().strip().split())


def strict_correct(pred: Any, gold: Any) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty judge response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_one(client: SGLangClient, row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    question = str(row.get("question") or row.get("instruction") or "")
    gold = str(row.get("answer") or "")
    pred = str(row.get("pred") or "")
    if not pred.strip():
        return {
            **row,
            "strict_correct": strict_correct(pred, gold),
            "llm_correct": False,
            "llm_confidence": 1.0,
            "llm_rationale": "empty prediction",
            "judge_raw": "",
            "judge_error": "",
        }
    prompt = (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Prediction:\n{pred}\n"
    )
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
        raw = (resp.get("content") or "").strip()
        payload = extract_json_object(raw)
        return {
            **row,
            "strict_correct": strict_correct(pred, gold),
            "llm_correct": bool(payload.get("correct")),
            "llm_confidence": float(payload.get("confidence", 0.0) or 0.0),
            "llm_rationale": str(payload.get("rationale", "")),
            "judge_raw": raw,
            "judge_error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **row,
            "strict_correct": strict_correct(pred, gold),
            "llm_correct": False,
            "llm_confidence": 0.0,
            "llm_rationale": "",
            "judge_raw": "",
            "judge_error": str(exc),
        }


def load_gold_rows(dataset_path: Path) -> dict[str, dict[str, Any]]:
    rows = read_table(dataset_path)
    gold: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        task = normalize_task(row, idx)
        index = str(task["index"])
        gold[index] = {
            "index": index,
            "question": task.get("instruction") or row.get("question") or row.get("query") or "",
            "answer": row.get("answer", ""),
            "image": row.get("image", ""),
            "image_url": row.get("image_url", ""),
        }
    return gold


def load_track_rows(results_path: Path, gold_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = read_jsonl(results_path)
    rows: list[dict[str, Any]] = []
    for pred_row in predictions:
        index = str(pred_row.get("index", ""))
        gold = gold_rows.get(index, {"index": index, "question": pred_row.get("instruction", ""), "answer": ""})
        rows.append(
            {
                "index": index,
                "question": gold.get("question") or pred_row.get("instruction", ""),
                "answer": gold.get("answer", ""),
                "pred": pred_row.get("pred", ""),
                "image": gold.get("image") or pred_row.get("image", ""),
                "image_url": gold.get("image_url", ""),
            }
        )
    return rows


def load_direct_rows(results_path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(results_path):
        rows.append(
            {
                "index": str(row.get("index", "")),
                "question": row.get("question") or row.get("instruction") or row.get("problem") or "",
                "answer": row.get("answer", ""),
                "pred": row.get("pred") or row.get("prediction") or "",
                "image": row.get("image", ""),
                "image_url": row.get("image_url", ""),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if not row.get("judge_error")]
    errors = [row for row in rows if row.get("judge_error")]
    return {
        "count": len(rows),
        "judged_count": len(judged),
        "error_count": len(errors),
        "strict_accuracy": mean([bool(row.get("strict_correct")) for row in rows]) if rows else None,
        "llm_accuracy": mean([bool(row.get("llm_correct")) for row in judged]) if judged else None,
        "avg_confidence": mean([float(row.get("llm_confidence", 0.0) or 0.0) for row in judged]) if judged else 0.0,
    }


def judge_track(
    *,
    track: str,
    results_root: Path,
    output_root: Path,
    gold_rows: dict[str, dict[str, Any]],
    client: SGLangClient,
    concurrency: int,
    max_tokens: int,
    limit: int,
) -> dict[str, Any]:
    results_path = results_root / "simpleVQA" / track / track / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"missing results: {results_path}")
    rows = load_track_rows(results_path, gold_rows)
    if limit:
        rows = rows[:limit]
    judged: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        future_to_idx = {pool.submit(judge_one, client, row, max_tokens): idx for idx, row in enumerate(rows)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            judged[idx] = future.result()
    final_rows = [row for row in judged if row is not None]
    track_dir = output_root / "simpleVQA" / track
    write_jsonl(track_dir / "llm_judge_results.jsonl", final_rows)
    summary = summarize(final_rows)
    write_json(track_dir / "llm_judge_summary.json", summary)
    return {"track": track, **summary}


def judge_named_results(
    *,
    name: str,
    results_path: Path,
    output_root: Path,
    client: SGLangClient,
    concurrency: int,
    max_tokens: int,
    limit: int,
) -> dict[str, Any]:
    rows = load_direct_rows(results_path)
    if limit:
        rows = rows[:limit]
    judged: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        future_to_idx = {pool.submit(judge_one, client, row, max_tokens): idx for idx, row in enumerate(rows)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            judged[idx] = future.result()
    final_rows = [row for row in judged if row is not None]
    run_dir = output_root / "simpleVQA" / name
    write_jsonl(run_dir / "llm_judge_results.jsonl", final_rows)
    summary = summarize(final_rows)
    write_json(run_dir / "llm_judge_summary.json", summary)
    return {"track": name, **summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score saved predictions with a Qwen judge model.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--results-root", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tracks", default="all", help="Comma-separated tracks or all.")
    parser.add_argument(
        "--results-jsonl",
        action="append",
        default=[],
        help="Direct result file to judge, as name=/path/to/results.jsonl. Can be repeated.",
    )
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL") or DEFAULT_QWEN32B_BASE_URL)
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME", "Qwen3-32B"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    client = SGLangClient(args.judge_base_url, args.judge_model)
    summaries = []

    if args.results_jsonl:
        for item in args.results_jsonl:
            if "=" not in item:
                raise ValueError("--results-jsonl must use name=/path/to/results.jsonl")
            name, path = item.split("=", 1)
            print(f"[judge] results={name}", flush=True)
            summary = judge_named_results(
                name=name,
                results_path=Path(path),
                output_root=output_root,
                client=client,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                limit=args.limit,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    else:
        if not args.dataset or not args.results_root:
            raise ValueError("--dataset and --results-root are required unless --results-jsonl is used")
        selected_tracks = TRACKS if args.tracks == "all" else [item.strip() for item in args.tracks.split(",") if item.strip()]
        gold_rows = load_gold_rows(Path(args.dataset))
        for track in selected_tracks:
            print(f"[judge] track={track}", flush=True)
            summary = judge_track(
                track=track,
                results_root=Path(args.results_root),
                output_root=output_root,
                gold_rows=gold_rows,
                client=client,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                limit=args.limit,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    summaries.sort(key=lambda item: (item.get("llm_accuracy") is not None, item.get("llm_accuracy") or 0), reverse=True)
    write_json(output_root / "simpleVQA" / "llm_judge_overall_summary.json", summaries)
    print(f"[done] wrote {output_root / 'simpleVQA' / 'llm_judge_overall_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
