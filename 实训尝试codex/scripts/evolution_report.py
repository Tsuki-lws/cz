from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_run(run_dir: Path, track: str) -> dict[str, Any]:
    track_dir = run_dir / track
    results = read_jsonl(track_dir / "results.jsonl")
    trajectories = read_jsonl(track_dir / "trajectories.jsonl")
    summary_path = track_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    traj_by_index = {str(row.get("index")): row for row in trajectories}
    metrics = []
    failure_types = Counter()
    reflection_count = 0
    memory_hit_count = 0
    tool_error_count = 0
    for row in results:
        idx = str(row.get("index"))
        traj = traj_by_index.get(idx, {})
        metric = traj.get("metrics") or {}
        debug = traj.get("debug") or {}
        metrics.append(metric)
        if debug.get("reflection_triggered"):
            reflection_count += 1
        if debug.get("memory_hits"):
            memory_hit_count += 1
        for key in ["initial_judge", "final_judge"]:
            signal = debug.get(key) or {}
            if signal.get("failure_type"):
                failure_types[str(signal.get("failure_type"))] += 1
        for event in traj.get("trajectory") or []:
            if event.get("role") == "tool" and event.get("ok") is False:
                tool_error_count += 1
    def avg(name: str) -> float:
        values = [float(m.get(name, 0) or 0) for m in metrics]
        return mean(values) if values else 0.0
    return {
        "run_dir": str(run_dir),
        "track": track,
        "count": len(results),
        "summary": summary,
        "avg_tokens": avg("tokens"),
        "avg_turns": avg("turns"),
        "avg_tool_calls": avg("tool_calls"),
        "avg_latency": avg("latency"),
        "tool_error_count": tool_error_count,
        "reflection_count": reflection_count,
        "memory_hit_count": memory_hit_count,
        "failure_types": dict(failure_types),
    }


def pct_delta(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / before


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare no-evolution and self-evolution runs.")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--evo-run", required=True)
    parser.add_argument("--track-baseline", default="track_d_evo")
    parser.add_argument("--track-evo", default="track_d_evo")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_run(Path(args.baseline_run), args.track_baseline)
    evo = load_run(Path(args.evo_run), args.track_evo)
    comparisons = {}
    for key in ["avg_tokens", "avg_turns", "avg_tool_calls", "avg_latency"]:
        comparisons[key] = {
            "baseline": baseline[key],
            "evo": evo[key],
            "relative_delta": pct_delta(float(baseline[key]), float(evo[key])),
        }
    for key in ["strict_accuracy", "llm_accuracy", "llm_judge_accuracy"]:
        before = baseline["summary"].get(key)
        after = evo["summary"].get(key)
        if before is not None or after is not None:
            comparisons[key] = {"baseline": before, "evo": after}

    report = {"baseline": baseline, "evo": evo, "comparisons": comparisons}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
