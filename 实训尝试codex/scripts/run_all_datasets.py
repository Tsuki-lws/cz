from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

DATASETS = {
    "2wiki": PROJECT_ROOT / "datasets" / "2wiki.jsonl",
    "simpleVQA": PROJECT_ROOT / "datasets" / "simpleVQA" / "SimpleVQA.jsonl",
    "benchmark": PROJECT_ROOT / "benchmark.csv",
}
TRACKS = ["track_a", "track_b", "track_c", "track_d", "track_e", "track_f"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "eval"

DEFAULT_LLM_BASE_URL = (
    "https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/"
    "project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/"
    "user-b1acf6ce-25a4-4cb6-b428-f427f4a59686/"
    "vscode/b2aa27b1-e0f7-425d-b208-acbd7f40ef68/"
    "68f1224c-8cc9-4e87-8701-523c6e59db1f/proxy/8000/v1"
)


def split_csv(value: str, allowed: list[str]) -> list[str]:
    if value == "all":
        return allowed
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise SystemExit(f"unknown values: {unknown}; allowed={allowed}")
    return selected


def run_one(dataset: str, track: str, args: argparse.Namespace, env: dict[str, str]) -> int:
    dataset_path = DATASETS[dataset]
    out_dir = Path(args.output_dir).resolve() / dataset / track
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    cmd = [
        sys.executable,
        "-m",
        "shared_sii_adapter.run_dataset",
        "--input",
        str(dataset_path),
        "--track",
        track,
        "--output-dir",
        str(out_dir),
        "--run-mode",
        args.run_mode,
        "--disable-evolution-updates",
        "--concurrency",
        str(args.concurrency),
        "--max-steps",
        str(args.max_steps),
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.score_mode:
        cmd.extend(["--score-mode", args.score_mode])
    if args.judge_concurrency:
        cmd.extend(["--judge-concurrency", str(args.judge_concurrency)])
    if args.enable_external_assist:
        cmd.append("--enable-external-assist")
        cmd.extend(["--external-assist-max-tokens", str(args.external_assist_max_tokens)])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.write_submission:
        cmd.append("--write-submission")
    print(f"[start] dataset={dataset} track={track} out={out_dir}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"[done] dataset={dataset} track={track} rc={proc.returncode} log={log_path}", flush=True)
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all uploaded tracks on local datasets.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or all.")
    parser.add_argument("--tracks", default="all", help="Comma-separated track names or all.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-mode", default="dev", choices=["train", "dev", "eval", "benchmark"])
    parser.add_argument("--score-mode", default="llm", choices=["strict", "llm"])
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--enable-external-assist", action="store_true")
    parser.add_argument("--external-assist-max-tokens", type=int, default=512)
    parser.add_argument("--write-submission", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = split_csv(args.datasets, list(DATASETS))
    tracks = split_csv(args.tracks, TRACKS)
    missing = [str(DATASETS[name]) for name in datasets if not DATASETS[name].exists()]
    if missing:
        raise SystemExit(f"missing dataset files: {missing}")

    env = os.environ.copy()
    env.setdefault("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    env.setdefault("MODEL_NAME", "Qwen3.5-9B")
    # Empty SEARCH_PROXY_URL selects direct online Serper/Jina mode in
    # harness-sii/tools/search_tool.py. Users can still export a proxy URL
    # explicitly when they want to route through the shared search service.
    env.setdefault("SEARCH_PROXY_URL", "")
    env.setdefault("SANDBOX_BASE_URL", "http://127.0.0.1:8080")

    failures = 0
    for dataset in datasets:
        for track in tracks:
            failures += int(run_one(dataset, track, args, env) != 0)
    if failures:
        print(f"[finish] failures={failures}", flush=True)
        return 1
    print("[finish] all runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
