# Parallel teacher trajectory collection with resume support.

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from distill.common import append_jsonl, load_yaml, read_jsonl
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig
from track_g_distill_ttl.agent import TRACK_NAME, run_one as run_track_g


def has_image(seed: dict[str, Any]) -> bool:
    return any(str(seed.get(key) or "").strip() for key in ("image", "image_path", "image_url", "image_b64"))


def build_runtime(
    config_path: str,
    *,
    teacher_name: str,
    args: argparse.Namespace,
) -> tuple[str, RuntimeConfig]:
    cfg = load_yaml(config_path)
    name = teacher_name or str(cfg["model"]).replace("/", "_")
    runtime = RuntimeConfig(
        llm_base_url=str(cfg["base_url"]),
        model_name=str(cfg["model"]),
        max_steps=args.max_steps,
        max_tokens=args.max_tokens if args.max_tokens else int(cfg.get("max_tokens", 4096)),
        temperature=float(cfg.get("temperature", 0.2)),
        enable_thinking=False,
        disable_tools=args.disable_tools,
        disable_reflection=args.disable_reflection,
        disable_memory=args.disable_memory,
        enable_xml_tool_fallback=True,
        structured_final_answer=True,
        output_dir=args.output_dir,
        track_name=TRACK_NAME,
        run_mode="train",
        allow_evolution_updates=not args.disable_memory,
        enable_external_assist=False,
    )
    return name, runtime


def seed_to_task(seed: dict[str, Any]) -> dict[str, Any]:
    task = {
        "index": seed["id"],
        "instruction": seed["question"],
        "image": seed.get("image", ""),
        "image_path": seed.get("image_path", ""),
        "image_url": seed.get("image_url", ""),
        "image_b64": seed.get("image_b64", ""),
    }
    return {key: value for key, value in task.items() if value not in (None, "")}


def build_episode(seed: dict[str, Any], result: AgentRunResult, teacher_name: str) -> dict[str, Any]:
    return {
        'id': seed['id'],
        'teacher': teacher_name,
        'question': seed['question'],
        'answer': seed.get('answer', ''),
        'image': seed.get('image', ''),
        'image_path': seed.get('image_path', ''),
        'image_url': seed.get('image_url', ''),
        'image_b64': seed.get('image_b64', ''),
        'prediction': result.pred,
        'elapsed_seconds': result.metrics.get('latency', 0),
        'turns': result.metrics.get('turns', 0),
        'total_tokens': result.metrics.get('tokens', 0),
        'tool_call_count': result.metrics.get('tool_calls', 0),
        'records': result.trajectory,
        'metrics': result.metrics,
        'debug': result.debug,
    }


def collect_one(seed: dict[str, Any], runtime: RuntimeConfig, teacher_name: str) -> dict[str, Any]:
    result = run_track_g(seed_to_task(seed), runtime)
    return build_episode(seed, result, teacher_name)


def run_collection(args: argparse.Namespace) -> None:
    seeds = read_jsonl(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_output = output_dir / 'episodes.jsonl'
    records_output = output_dir / 'trajectories.jsonl'

    existing_ids: set[str] = set()
    if trajectory_output.exists():
        existing_ids = {row['id'] for row in read_jsonl(trajectory_output)}

    text_teacher_name, text_runtime = build_runtime(args.config, teacher_name=args.teacher_name, args=args)
    vision_teacher_name = ""
    vision_runtime = None
    if args.vision_config:
        vision_teacher_name, vision_runtime = build_runtime(
            args.vision_config,
            teacher_name=args.vision_teacher_name,
            args=args,
        )

    def guarded(seed: dict[str, Any]) -> dict[str, Any] | None:
        if seed['id'] in existing_ids:
            return None
        if vision_runtime and has_image(seed):
            return collect_one(seed, vision_runtime, vision_teacher_name)
        return collect_one(seed, text_runtime, text_teacher_name)

    pending = [seed for seed in seeds if seed['id'] not in existing_ids]
    if args.concurrency <= 1:
        for seed in tqdm(pending, desc='collect'):
            episode = guarded(seed)
            if not episode:
                continue
            append_episode(trajectory_output, records_output, episode)
        return

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(guarded, seed) for seed in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc='collect'):
            episode = future.result()
            if not episode:
                continue
            append_episode(trajectory_output, records_output, episode)


def append_episode(trajectory_output: Path, records_output: Path, episode: dict[str, Any]) -> None:
    append_jsonl(trajectory_output, episode)
    for record in episode['records']:
        append_jsonl(records_output, {'question_id': episode['id'], 'teacher': episode['teacher'], **record})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect teacher trajectories with resume support.')
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--vision-config', default='')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--teacher-name', default='')
    parser.add_argument('--vision-teacher-name', default='')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--disable-tools', action='store_true')
    parser.add_argument('--disable-reflection', action='store_true')
    parser.add_argument('--disable-memory', action='store_true', default=True)
    return parser.parse_args()


def main() -> None:
    run_collection(parse_args())


if __name__ == '__main__':
    main()
