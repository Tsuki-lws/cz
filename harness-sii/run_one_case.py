from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_runner import run_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case-file", required=True)
    p.add_argument("--max-steps", type=int, default=4)
    args = p.parse_args()

    case_path = Path(args.case_file)
    payload = json.loads(case_path.read_text(encoding="utf-8"))

    task = {
        "id": payload["id"],
        "instruction": payload["instruction"],
        "image_b64": payload.get("image_b64"),
        "image_url": payload.get("image_url"),
    }

    result = run_task(task, max_steps=args.max_steps, trajectory_dir=payload["traj_dir"])
    payload["answer"] = result["answer"]
    payload["trajectory_path"] = result["trajectory_path"]
    payload["summary"] = result["summary"]
    case_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
