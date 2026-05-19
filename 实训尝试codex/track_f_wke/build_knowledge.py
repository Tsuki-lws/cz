from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared_sii_adapter.types import RuntimeConfig
from shared_sii_adapter.react_runner import run_react_task

from .exploration.explorer import build_exploration_prompt, make_lesson
from .knowledge_store import KnowledgeStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reward-free world knowledge memory from public seeds.")
    parser.add_argument("--seeds", required=True, help="JSONL with topic/question/instruction fields")
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    runtime = RuntimeConfig(llm_base_url=args.llm_base_url, model_name=args.model, run_mode="dev", allow_evolution_updates=True, track_name="track_f")
    store = KnowledgeStore()
    with Path(args.seeds).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= args.limit:
                break
            if not line.strip():
                continue
            seed = json.loads(line)
            task = {"index": f"wke_{idx}", "instruction": build_exploration_prompt(seed)}
            result = run_react_task(task, runtime, track_name="track_f_wke_build")
            store.add(make_lesson(seed, result.pred, int(result.metrics.get("tool_calls", 0))))


if __name__ == "__main__":
    main()

