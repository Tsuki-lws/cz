from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import re
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_32B_BASE_URL = (
    "https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/"
    "project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/"
    "user-b260c9e2-91ae-48ff-bfce-dcfd887a0358/"
    "vscode/aace7e69-939d-426f-944d-8d2e148bdb2a/"
    "926b48b6-abf2-4e2b-b2ff-4dea116721c0/proxy/30000/v1"
)
DEFAULT_32B_MODEL = "/inspire/qb-ilm2/project/26summer-camp-01/26210300/Qwen3-32B"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_think(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<think>" in text:
        # Some SGLang/Qwen responses ignore enable_thinking=False and emit an
        # unterminated reasoning block. For Judge JSON we should not treat the
        # hidden chain as a valid answer or reason.
        text = text.split("<think>", 1)[0]
    return text.strip()


def normalize_answer(text: str) -> str:
    text = _strip_think(text).strip()
    if not text or text.startswith("[HARNESS]"):
        return ""
    lower = text.lower()
    if "<answer>" in lower and "</answer>" in lower:
        start = lower.find("<answer>") + len("<answer>")
        end = lower.find("</answer>", start)
        return text[start:end].strip()
    markers = ["最终答案", "答案是", "答案为", "answer is", "final answer", "answer"]
    for line in reversed([x.strip() for x in text.splitlines() if x.strip()]):
        low = line.lower()
        for marker in markers:
            if marker in low:
                ans = line[low.rfind(marker) + len(marker):].lstrip(":：").strip()
                return ans.strip(" .。；;，,")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) == 1:
        return lines[0].strip(" .。；;，,")
    return ""


def score_exact(pred: str, gold: str) -> bool:
    alias = {
        "uk": "unitedkingdom",
        "u.k.": "unitedkingdom",
        "英国": "unitedkingdom",
        "unitedkingdom": "unitedkingdom",
        "usa": "unitedstates",
        "u.s.a.": "unitedstates",
        "us": "unitedstates",
        "u.s.": "unitedstates",
        "美国": "unitedstates",
        "france": "france",
        "法国": "france",
        "ctscan": "医学ct图像",
        "computedtomography": "医学ct图像",
    }

    def canon(s: str) -> str:
        s = normalize_answer(s).lower()
        s = re.sub(r"\s+", "", s)
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = re.sub(r"[，。！？；：“”‘’、（）《》【】\[\]{}<>]", "", s)
        s = alias.get(s, s)
        return s

    p = canon(pred)
    g = canon(gold)
    if not p or not g:
        return False
    return p == g or p in g or g in p


def load_dataset(name: str, path: Path) -> list[dict[str, Any]]:
    if name == "2wiki":
        return _read_jsonl(path)
    if name == "simplevqa":
        return _read_jsonl(path)
    raise ValueError(f"unknown dataset: {name}")


def build_instruction(name: str, row: dict[str, Any], memory: list[str], evolved: bool) -> str:
    mem = "\n".join(f"- {m}" for m in memory[-8:])
    memory_block = f"\n\n可用经验记忆（只作为策略提示，不是事实来源）：\n{mem}\n" if evolved and mem else ""
    if name == "2wiki":
        titles = row.get("context", {}).get("title", [])
        sentences = row.get("context", {}).get("sentences", [])
        chunks = []
        for title, sents in zip(titles, sentences):
            text = " ".join(str(x) for x in sents)
            chunks.append(f"[{title}] {text}")
        context = "\n".join(chunks)
        return (
            "You are answering a 2Wiki multihop QA item. Use only the provided context. "
            "Return only the final answer, with no explanation.\n"
            f"{memory_block}\nQuestion: {row.get('question', '')}\n\nContext:\n{context}"
        )
    return (
        "Answer this visual question. First identify the visual entity from the image, "
        "then answer using concise world knowledge when needed. Return only the final answer.\n"
        f"{memory_block}\nQuestion: {row.get('question', '')}"
    )


def image_b64_for(name: str, row: dict[str, Any], dataset_path: Path) -> str | None:
    if name != "simplevqa":
        return None
    image = row.get("image")
    if not image:
        return None
    path = dataset_path.parent / image
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def write_case(
    case_path: Path,
    task_id: str,
    instruction: str,
    image_b64: str | None,
    image_url: str | None,
    traj_dir: Path,
) -> None:
    payload = {
        "id": task_id,
        "instruction": instruction,
        "image_b64": image_b64,
        "image_url": image_url,
        "traj_dir": str(traj_dir),
        "trajectory_path": str(traj_dir / f"{task_id}.jsonl"),
        "answer": "",
    }
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run_case(case_path: Path, max_steps: int, timeout_sec: int, disable_tools: bool, max_tokens: int) -> tuple[str, str]:
    env = os.environ.copy()
    env["MAX_TOKENS"] = str(max_tokens)
    if disable_tools:
        env["DISABLE_TOOLS"] = "1"
    cmd = [
        "conda",
        "run",
        "-n",
        "pegp",
        "python",
        "run_one_case.py",
        "--case-file",
        str(case_path),
        "--max-steps",
        str(max_steps),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout_sec}s"
    if proc.returncode != 0:
        return "", (proc.stderr or proc.stdout)[-1000:]
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    answer = normalize_answer(payload.get("answer", ""))
    if not answer:
        traj_path = Path(payload["trajectory_path"])
        for turn in reversed(_read_jsonl(traj_path)):
            if turn.get("role") == "assistant":
                answer = normalize_answer(turn.get("content", ""))
                if answer:
                    break
    return answer, ""


def judge_with_32b(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    pred: str,
    traj_tail: list[dict[str, Any]],
    memory: list[str],
) -> dict[str, Any]:
    compact_traj = [
        {
            "role": t.get("role"),
            "content": str(t.get("content", ""))[:800],
            "reasoning_content": str(t.get("reasoning_content", ""))[:800],
        }
        for t in traj_tail[-6:]
    ]
    prompt = {
        "question": row.get("question") or row.get("atomic_question"),
        "prediction": pred,
        "trajectory_tail": compact_traj,
        "memory": memory[-8:],
        "instruction": (
            "Judge whether the prediction is likely sufficient without using any gold answer. "
            "Return JSON only: {\"pass\": boolean, \"reason\": string, "
            "\"memory\": string, \"retry_hint\": string}. The memory should be a reusable "
            "strategy lesson, not this case's gold answer."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict LLM-as-a-Judge and memory organizer. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            max_tokens=512,
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        text = _strip_think(resp.choices[0].message.content or "")
    except Exception as exc:  # noqa: BLE001
        return {"pass": bool(pred), "reason": f"judge_error: {exc}", "memory": "", "retry_hint": ""}
    raw_content = resp.choices[0].message.content or "" if "resp" in locals() else ""
    raw_reasoning = getattr(resp.choices[0].message, "reasoning_content", "") or "" if "resp" in locals() else ""
    candidate = text or _strip_think(raw_reasoning)
    match = re.search(r"\{.*\}", candidate, flags=re.S)
    if not match:
        reason = candidate[:300] or ("empty prediction" if not pred else "judge returned no JSON")
        return {"pass": bool(pred), "reason": reason, "memory": "", "retry_hint": ""}
    try:
        obj = json.loads(match.group(0))
        return {
            "pass": bool(obj.get("pass")),
            "reason": str(obj.get("reason", ""))[:500],
            "memory": str(obj.get("memory", ""))[:500],
            "retry_hint": str(obj.get("retry_hint", ""))[:500],
        }
    except Exception as exc:  # noqa: BLE001
        return {"pass": bool(pred), "reason": f"judge_parse_error: {exc}: {text[:300]}", "memory": "", "retry_hint": ""}


def consolidate_trajectory(
    out_path: Path,
    dataset: str,
    index: int,
    task_id: str,
    attempt: int,
    traj_path: Path,
    judge: dict[str, Any] | None,
) -> None:
    for turn in _read_jsonl(traj_path):
        turn = dict(turn)
        turn.update({"dataset": dataset, "index": index, "task_id": task_id, "attempt": attempt})
        _append_jsonl(out_path, turn)
    if judge is not None:
        _append_jsonl(
            out_path,
            {
                "timestamp": time.time(),
                "step_id": "judge",
                "role": "judge",
                "content": judge,
                "dataset": dataset,
                "index": index,
                "task_id": task_id,
                "attempt": attempt,
            },
        )


def run_attempt(
    args: argparse.Namespace,
    dataset_path: Path,
    case_dir: Path,
    traj_dir: Path,
    dataset: str,
    index: int,
    row: dict[str, Any],
    attempt: int,
    memory_snapshot: list[str],
    retry_hint: str = "",
) -> dict[str, Any]:
    task_id = f"{dataset}_{args.mode}_{index:03d}_a{attempt}"
    instruction = build_instruction(dataset, row, memory_snapshot, args.mode == "evolved")
    if retry_hint:
        instruction += "\n\nJudge retry hint, without gold answer:\n" + retry_hint
    case_path = case_dir / f"{task_id}.json"
    write_case(
        case_path,
        task_id,
        instruction,
        image_b64_for(dataset, row, dataset_path),
        row.get("image_url") if dataset == "simplevqa" else None,
        traj_dir,
    )
    pred, error = run_case(case_path, args.max_steps, args.timeout_sec, args.disable_tools, args.max_tokens)
    return {
        "index": index,
        "row": row,
        "attempt": attempt,
        "task_id": task_id,
        "pred": pred,
        "error": error,
        "traj_path": traj_dir / f"{task_id}.jsonl",
    }


def process_attempt_result(
    args: argparse.Namespace,
    result_path: Path,
    traj_all_path: Path,
    memory_path: Path,
    judge_client: OpenAI | None,
    memory: list[str],
    dataset: str,
    attempt_result: dict[str, Any],
) -> dict[str, Any]:
    index = int(attempt_result["index"])
    row = attempt_result["row"]
    pred = attempt_result["pred"]
    judge_obj = None
    if args.judge and judge_client is not None:
        judge_obj = judge_with_32b(
            judge_client,
            args.judge_model,
            row,
            pred,
            _read_jsonl(attempt_result["traj_path"]),
            memory,
        )
        mem = (judge_obj.get("memory") or "").strip()
        if mem and mem not in memory:
            memory.append(mem)
            memory[:] = memory[-80:]
            memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")

    consolidate_trajectory(
        traj_all_path,
        dataset,
        index,
        attempt_result["task_id"],
        int(attempt_result["attempt"]),
        attempt_result["traj_path"],
        judge_obj,
    )

    answer = row.get("answer", "")
    output = {
        "index": index,
        "instruction": row.get("question", ""),
        "image": row.get("image", ""),
        "answer": answer,
        "pred": pred,
        "error": attempt_result["error"],
        "judge": judge_obj,
    }
    _append_jsonl(result_path, output)
    correct = score_exact(pred, answer) if answer else None
    print(f"[done] {dataset} {index}: pred={pred[:80]!r} correct={correct}")
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Run validation datasets with 9B base and optional 32B judge/memory.")
    p.add_argument("--dataset", choices=["2wiki", "simplevqa"], required=True)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--mode", choices=["baseline", "evolved"], default="baseline")
    p.add_argument("--out-dir", type=Path, default=ROOT / "validation_outputs")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument("--timeout-sec", type=int, default=150)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--disable-tools", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--workers", type=int, default=1, help="Parallel 9B case workers. Judge/memory writes stay in main process.")
    p.add_argument("--batch-size", type=int, default=0, help="Parallel scheduling batch size; default is workers.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL", DEFAULT_32B_BASE_URL))
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", DEFAULT_32B_MODEL))
    args = p.parse_args()

    data_path = args.data_path
    if data_path is None:
        data_path = PROJECT_ROOT / ("datasets/2wiki.jsonl" if args.dataset == "2wiki" else "datasets/simpleVQA/SimpleVQA.jsonl")
    rows = load_dataset(args.dataset, data_path)
    end = len(rows) if args.limit <= 0 else min(len(rows), args.start + args.limit)
    selected = list(enumerate(rows[args.start:end], start=args.start))

    run_name = f"{args.dataset}_{args.mode}"
    out_dir = args.out_dir / run_name
    case_dir = out_dir / "cases"
    traj_dir = out_dir / "trajectories"
    result_path = out_dir / f"{run_name}_results.jsonl"
    traj_all_path = out_dir / f"{run_name}_trajectories.jsonl"
    memory_path = out_dir / f"{run_name}_memory.json"
    metrics_path = out_dir / f"{run_name}_metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    done: dict[int, dict[str, Any]] = {}
    if args.resume and result_path.exists():
        for item in _read_jsonl(result_path):
            done[int(item["index"])] = item
    memory: list[str] = []
    if args.resume and memory_path.exists():
        memory = json.loads(memory_path.read_text(encoding="utf-8"))

    judge_client = OpenAI(base_url=args.judge_base_url, api_key="EMPTY") if args.judge else None

    pending = [(index, row) for index, row in selected if index not in done]
    if args.workers > 1 and args.attempts > 1:
        print("[warn] --workers > 1 currently runs one attempt per sample; set --attempts 1 for parallel runs.")
        args.attempts = 1

    if args.workers > 1 and pending:
        batch_size = args.batch_size if args.batch_size > 0 else args.workers
        completed_count = len(done)
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset:offset + batch_size]
            memory_snapshot = list(memory)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(
                        run_attempt,
                        args,
                        data_path,
                        case_dir,
                        traj_dir,
                        args.dataset,
                        index,
                        row,
                        1,
                        memory_snapshot,
                        "",
                    )
                    for index, row in chunk
                ]
                attempt_results = []
                for fut in as_completed(futures):
                    attempt_results.append(fut.result())
            for attempt_result in sorted(attempt_results, key=lambda x: int(x["index"])):
                output = process_attempt_result(
                    args,
                    result_path,
                    traj_all_path,
                    memory_path,
                    judge_client,
                    memory,
                    args.dataset,
                    attempt_result,
                )
                done[int(output["index"])] = output
                completed_count += 1
                print(f"[{completed_count}/{len(selected)}] {args.dataset} {output['index']}")
    else:
        for index, row in selected:
            if index in done:
                print(f"[skip] {args.dataset} {index}")
                continue
            best_pred = ""
            error = ""
            judge_obj = None
            for attempt in range(1, max(1, args.attempts) + 1):
                attempt_result = run_attempt(
                    args,
                    data_path,
                    case_dir,
                    traj_dir,
                    args.dataset,
                    index,
                    row,
                    attempt,
                    memory,
                    str(judge_obj["retry_hint"]) if judge_obj and judge_obj.get("retry_hint") else "",
                )
                best_pred = attempt_result["pred"]
                error = attempt_result["error"]
                judge_obj = None
                if args.judge and judge_client is not None:
                    judge_obj = judge_with_32b(
                        judge_client,
                        args.judge_model,
                        row,
                        best_pred,
                        _read_jsonl(attempt_result["traj_path"]),
                        memory,
                    )
                    mem = (judge_obj.get("memory") or "").strip()
                    if mem and mem not in memory:
                        memory.append(mem)
                        memory = memory[-80:]
                        memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
                consolidate_trajectory(
                    traj_all_path,
                    args.dataset,
                    index,
                    attempt_result["task_id"],
                    attempt,
                    attempt_result["traj_path"],
                    judge_obj,
                )
                if not args.judge or (judge_obj and judge_obj.get("pass")):
                    break

            answer = row.get("answer", "")
            result = {
                "index": index,
                "instruction": row.get("question", ""),
                "image": row.get("image", ""),
                "answer": answer,
                "pred": best_pred,
                "error": error,
                "judge": judge_obj,
            }
            _append_jsonl(result_path, result)
            done[index] = result
            correct = score_exact(best_pred, answer) if answer else None
            print(f"[{len(done)}/{len(selected)}] {args.dataset} {index}: pred={best_pred[:80]!r} correct={correct}")

    scored = [x for x in done.values() if str(x.get("answer", "")).strip()]
    correct_count = sum(1 for x in scored if score_exact(x.get("pred", ""), x.get("answer", "")))
    metrics = {
        "dataset": args.dataset,
        "mode": args.mode,
        "completed": len(done),
        "scored": len(scored),
        "correct": correct_count,
        "accuracy": (correct_count / len(scored)) if scored else None,
        "result_path": str(result_path),
        "trajectory_path": str(traj_all_path),
        "memory_path": str(memory_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
