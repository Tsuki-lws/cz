from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import json
import os
from pathlib import Path

from .eval_metrics import summarize
from .io_utils import read_table, write_json, write_jsonl
from .llm_judge import summarize_with_llm_judge
from .submit_writer import write_submission
from .types import AgentRunResult, RuntimeConfig
from .compliance import assert_harness_base_model, assert_no_benchmark_evolution, assert_no_gold_payload, assert_teacher_model_size_le_32b


TRACK_MODULES = {
    "baseline": "track_e_engineering.agent",
    "track_a": "track_a_distill.agent",
    "track_b": "track_b_mia.agent",
    "track_c": "track_c_ahe.agent",
    "track_d": "track_d_ttl.agent",
    "track_e": "track_e_engineering.agent",
    "track_f": "track_f_wke.agent",
}

DEFAULT_QWEN32B_BASE_URL = (
    "https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/"
    "project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/"
    "user-b260c9e2-91ae-48ff-bfce-dcfd887a0358/"
    "vscode/aace7e69-939d-426f-944d-8d2e148bdb2a/"
    "926b48b6-abf2-4e2b-b2ff-4dea116721c0/proxy/30000"
)


def load_runner(track: str):
    module_name = TRACK_MODULES.get(track, track)
    module = importlib.import_module(module_name)
    return module.run_one


def normalize_task(row: dict, idx: int) -> dict:
    task = dict(row)
    task.setdefault("index", row.get("index") or row.get("id") or str(idx))
    task.setdefault("instruction", row.get("instruction") or row.get("question") or row.get("query") or row.get("problem") or "")
    if row.get("problem") and "problem" not in task:
        task["problem"] = row.get("problem")
    return task


def strip_gold_fields(task: dict) -> dict:
    forbidden = {"answer", "gold", "ground_truth", "label"}
    return {key: value for key, value in task.items() if str(key).lower() not in forbidden}


def resolve_local_image(task: dict, dataset_path: Path) -> dict:
    existing_image_path = str(task.get("image_path") or "")
    if existing_image_path and Path(existing_image_path).expanduser().exists():
        return task
    image = str(task.get("image") or "")
    if image and not image.startswith(("http://", "https://", "data:")) and len(image) > 1000:
        task = dict(task)
        task["image_b64"] = image
        return task
    if not image or image.startswith(("http://", "https://", "data:")):
        return task
    image_path = Path(image)
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        image_path,
        dataset_path.parent / image_path,
        dataset_path.parent.parent / image_path,
        project_root / "datasets" / image_path,
        project_root / "datasets" / "simpleVQA" / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            task = dict(task)
            task["image_path"] = str(candidate.resolve())
            return task
    return task


def run_dataset(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    rows = read_table(input_path)
    if args.limit:
        rows = rows[: args.limit]
    runtime = RuntimeConfig(
        llm_base_url=args.llm_base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        model_name=args.model or os.getenv("MODEL_NAME", "Qwen3.5-9B"),
        judge_base_url=args.judge_base_url or os.getenv("JUDGE_BASE_URL") or DEFAULT_QWEN32B_BASE_URL,
        judge_model_name=args.judge_model or os.getenv("JUDGE_MODEL_NAME", "Qwen3-32B"),
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        disable_tools=args.disable_tools,
        disable_reflection=args.disable_reflection,
        disable_memory=args.disable_memory,
        enable_xml_tool_fallback=not args.disable_xml_tool_fallback,
        structured_final_answer=not args.disable_structured_final_answer,
        output_dir=args.output_dir,
        track_name=args.track,
        group_id=args.group_id,
        run_mode=args.run_mode,
        allow_evolution_updates=not args.disable_evolution_updates,
        enable_external_assist=args.enable_external_assist,
        external_assist_max_tokens=args.external_assist_max_tokens,
    )
    if runtime.benchmark_mode and not args.disable_evolution_updates:
        runtime.allow_evolution_updates = False
    assert_harness_base_model(runtime.model_name)
    assert_teacher_model_size_le_32b(runtime.judge_model_name)
    assert_no_benchmark_evolution(runtime.run_mode, runtime.allow_evolution_updates)
    run_one = load_runner(args.track)
    results: list[AgentRunResult] = []
    out_dir = Path(args.output_dir) / args.track
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.jsonl"
    trajectory_path = out_dir / "trajectories.jsonl"
    existing_results: dict[str, AgentRunResult] = {}
    if args.resume and result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            result = AgentRunResult(
                index=str(row.get("index", "")),
                instruction=str(row.get("instruction", "")),
                image=str(row.get("image", "")),
                pred=str(row.get("pred", "")),
            )
            existing_results[result.index] = result
    tasks = []
    for idx, row in enumerate(rows):
        task = strip_gold_fields(normalize_task(row, idx))
        if str(task.get("index")) in existing_results:
            continue
        task = resolve_local_image(task, input_path)
        assert_no_gold_payload(task)
        tasks.append(task)
    effective_concurrency = max(1, args.concurrency)
    stateful_tracks = {"track_b", "track_c", "track_d", "track_f"}
    if args.track in stateful_tracks and runtime.allow_evolution_updates:
        effective_concurrency = 1
    if runtime.benchmark_mode and runtime.allow_evolution_updates:
        effective_concurrency = 1

    def write_progress(completed_results: list[AgentRunResult]) -> None:
        write_jsonl(result_path, [r.to_result_row() for r in completed_results])
        write_jsonl(
            trajectory_path,
            [
                {
                    "index": r.index,
                    "instruction": r.instruction,
                    "pred": r.pred,
                    "trajectory": r.trajectory,
                    "metrics": r.metrics,
                    "debug": r.debug,
                }
                for r in completed_results
            ],
        )

    results = list(existing_results.values())
    if tasks:
        if effective_concurrency == 1:
            for task in tasks:
                results.append(run_one(task, runtime))
                write_progress(results)
        else:
            ordered: list[AgentRunResult | None] = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=effective_concurrency) as pool:
                future_to_idx = {pool.submit(run_one, task, runtime): idx for idx, task in enumerate(tasks)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    ordered[idx] = future.result()
                    completed = list(existing_results.values()) + [r for r in ordered if r is not None]
                    write_progress(completed)
            results.extend(r for r in ordered if r is not None)
    gold = {str(normalize_task(row, idx)["index"]): str(row.get("answer", "")) for idx, row in enumerate(rows) if row.get("answer")}
    summary = summarize(results, gold)
    if args.score_mode == "llm":
        gold_rows = {str(normalize_task(row, idx)["index"]): normalize_task(row, idx) for idx, row in enumerate(rows) if row.get("answer")}
        llm_summary, judge_rows = summarize_with_llm_judge(
            results,
            gold_rows,
            runtime,
            concurrency=args.judge_concurrency,
        )
        summary.update(llm_summary)
        write_jsonl(out_dir / "llm_judge_results.jsonl", judge_rows)
    write_json(out_dir / "summary.json", summary)
    if args.write_submission:
        write_submission(results, Path(args.output_dir) / "submit", args.group_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one exploration track on a dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--track", default="track_e", choices=sorted(TRACK_MODULES))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "runs" / "eval"))
    parser.add_argument("--group-id", default="0")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--score-mode", default="llm", choices=["strict", "llm"])
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--enable-external-assist", action="store_true", help="Allow <=32B judge model to organize reflection/memory, never as Harness base model.")
    parser.add_argument("--external-assist-max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--disable-tools", action="store_true")
    parser.add_argument("--disable-reflection", action="store_true")
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--disable-xml-tool-fallback", action="store_true")
    parser.add_argument("--disable-structured-final-answer", action="store_true")
    parser.add_argument("--write-submission", action="store_true")
    parser.add_argument("--run-mode", default="eval", choices=["train", "dev", "eval", "benchmark"])
    parser.add_argument("--disable-evolution-updates", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel samples for stateless runs. Stateful evolution tracks are forced to 1 when updates are enabled.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N rows. Intended for smoke tests and dev checks.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing results.jsonl rows and only run missing indexes.")
    return parser.parse_args()


if __name__ == "__main__":
    run_dataset(parse_args())
