from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def _load_benchmark(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text.startswith("[HARNESS]"):
        return ""
    if "<answer>" in text.lower() and "</answer>" in text.lower():
        lower = text.lower()
        start = lower.find("<answer>")
        end = lower.find("</answer>")
        if start != -1 and end != -1 and end > start:
            return text[start + len("<answer>"):end].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    if first.startswith("根据") and len(lines) > 1:
        return lines[-1]
    return first


_ANSWER_PATTERNS = [
    re.compile(r"(?:final answer|answer is|answer|答案是|答案为|最终答案)[:：]\s*([^\n。]+)", re.IGNORECASE),
    re.compile(r"(?:so|therefore),?\s+the answer is\s+([^\n。]+)", re.IGNORECASE),
]


def _extract_explicit_answer(text: str) -> str:
    text = (text or "").strip()
    if not text or text.startswith("[HARNESS]"):
        return ""
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            ans = matches[-1].strip().strip(" .。；;，,")
            if ans and not ans.lower().startswith(("unknown", "unclear", "not enough")):
                return ans
    return ""


def _extract_answer_from_traj(traj_path: Path) -> str:
    if not traj_path.exists():
        return ""
    rows = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in reversed(rows):
        if row.get("role") != "assistant":
            continue
        content = (row.get("content") or "").strip()
        if content:
            answer = _normalize_answer(content)
            if answer:
                return answer
        reasoning = row.get("reasoning_content") or ""
        answer = _extract_explicit_answer(reasoning)
        if answer:
            return answer
    return ""


def _run_one_case(case_file: Path, max_steps: int, timeout_sec: int) -> tuple[str, str]:
    cmd = [
        "conda",
        "run",
        "-n",
        "pegp",
        "python",
        "run_one_case.py",
        "--case-file",
        str(case_file),
        "--max-steps",
        str(max_steps),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent,
            timeout=timeout_sec,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout_sec}s"

    if proc.returncode != 0:
        return "", f"worker exit {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}"

    try:
        payload = json.loads(case_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "", f"read case file failed: {exc}"

    traj_path = Path(payload["trajectory_path"])
    answer = _normalize_answer(payload.get("answer", ""))
    if not answer or answer.startswith("[HARNESS]"):
        answer = _extract_answer_from_traj(traj_path)
    return answer, ""


def main() -> None:
    p = argparse.ArgumentParser(description="Run benchmark.csv and generate submission files.")
    p.add_argument("--benchmark", default="/inspire/qb-ilm2/project/26summer-camp-01/public/benchmark.csv")
    p.add_argument("--group", required=True)
    p.add_argument("--traj-dir", default="trajectories_submit")
    p.add_argument("--work-dir", default="cases_submit")
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--timeout-sec", type=int, default=180)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="Reuse existing case/trajectory outputs when available.")
    args = p.parse_args()

    benchmark_path = Path(args.benchmark)
    rows = _load_benchmark(benchmark_path)
    start = max(0, args.start)
    end = len(rows) if args.limit <= 0 else min(len(rows), start + args.limit)
    selected = rows[start:end]

    out_csv = Path(f"group_{args.group}.csv")
    out_json = Path(f"group_{args.group}.json")
    work_dir = Path(args.work_dir)
    traj_dir = Path(args.traj_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    answers: list[dict] = []
    trajectories: list[dict] = []

    def flush_outputs() -> None:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["problem", "image", "answer"])
            writer.writeheader()
            writer.writerows(answers)
        out_json.write_text(json.dumps(trajectories, ensure_ascii=False, indent=2), encoding="utf-8")

    for global_idx, row in enumerate(selected, start=start):
        task_id = f"benchmark_{global_idx:03d}"
        case_file = work_dir / f"{task_id}.json"
        payload = {
            "id": task_id,
            "instruction": row["problem"],
            "image_b64": ((row.get("image") or "").strip() or None),
            "traj_dir": str(traj_dir),
            "trajectory_path": str(traj_dir / f"{task_id}.jsonl"),
            "answer": "",
        }
        if not (args.resume and case_file.exists()):
            case_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        if args.resume and case_file.exists():
            existing = json.loads(case_file.read_text(encoding="utf-8"))
            existing_answer = (existing.get("answer", "") or "").strip()
            if existing_answer and not existing_answer.startswith("[HARNESS]"):
                answer = existing.get("answer", "")
                error = existing.get("error", "")
            else:
                answer, error = _run_one_case(case_file, max_steps=args.max_steps, timeout_sec=args.timeout_sec)
        else:
            answer, error = _run_one_case(case_file, max_steps=args.max_steps, timeout_sec=args.timeout_sec)

        answers.append(
            {
                "problem": row.get("problem", ""),
                "image": row.get("image", ""),
                "answer": answer,
            }
        )

        traj_path = traj_dir / f"{task_id}.jsonl"
        turns = []
        if traj_path.exists():
            turns = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        trajectories.append(
            {
                "index": global_idx,
                "task_id": task_id,
                "problem": row.get("problem", ""),
                "answer": answer,
                "error": error,
                "trajectory": turns,
            }
        )

        print(f"[{global_idx + 1}/{len(rows)}] done: {task_id} -> {answer[:120]}")
        flush_outputs()

    flush_outputs()
    print(f"CSV:  {out_csv}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
