from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .io_utils import write_csv, write_json
from .types import AgentRunResult


def write_submission(results: list[AgentRunResult], output_dir: str | Path, group_id: str) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"group_{group_id}.csv"
    json_path = out / f"group_{group_id}.json"
    zip_path = out / f"group_{group_id}.zip"
    rows = [{"index": r.index, "answer": r.pred} for r in results]
    write_csv(csv_path, rows, fieldnames=["index", "answer"])
    payload = [
        {
            "index": r.index,
            "instruction": r.instruction,
            "answer": r.pred,
            "trajectory": r.trajectory,
            "metrics": r.metrics,
            "debug": r.debug,
        }
        for r in results
    ]
    write_json(json_path, payload)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
        zf.write(json_path, arcname=json_path.name)
    return {"csv": str(csv_path), "json": str(json_path), "zip": str(zip_path)}

