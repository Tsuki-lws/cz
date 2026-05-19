# Parallel teacher trajectory collection with resume support.

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from tqdm import tqdm

from distill.common import append_jsonl, read_jsonl
from distill.harness.agent import AgentConfig, ReActAgent
from distill.harness.llm_client import AsyncLLMClient, load_backend_config
from distill.harness.teacher_router import TeacherRouter
from distill.harness.tools import ToolRegistry


def build_episode(seed: dict[str, Any], result: Any, teacher_name: str) -> dict[str, Any]:
    return {
        'id': seed['id'],
        'teacher': teacher_name,
        'question': seed['question'],
        'answer': seed.get('answer', ''),
        'image': seed.get('image', ''),
        'image_path': seed.get('image_path', ''),
        'image_url': seed.get('image_url', ''),
        'image_b64': seed.get('image_b64', ''),
        'prediction': result.prediction,
        'elapsed_seconds': result.elapsed_seconds,
        'turns': result.turns,
        'total_tokens': result.total_tokens,
        'tool_call_count': result.tool_call_count,
        'records': result.records,
    }


async def collect_one(seed: dict[str, Any], agent: ReActAgent, teacher_name: str) -> dict[str, Any]:
    result = await agent.run(
        question_id=seed['id'],
        question=seed['question'],
        image=seed.get('image') or None,
        image_path=seed.get('image_path') or None,
        image_url=seed.get('image_url') or None,
        image_b64=seed.get('image_b64') or None,
        metadata={'answer': seed.get('answer', '')},
    )
    return build_episode(seed, result, teacher_name)


async def run_collection(args: argparse.Namespace) -> None:
    seeds = read_jsonl(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_output = output_dir / 'episodes.jsonl'
    records_output = output_dir / 'trajectories.jsonl'

    existing_ids: set[str] = set()
    if trajectory_output.exists():
        existing_ids = {row['id'] for row in read_jsonl(trajectory_output)}

    text_config = load_backend_config(args.config)
    text_teacher_name = args.teacher_name or text_config.model.replace('/', '_')
    text_client = AsyncLLMClient(text_config)
    vision_config = load_backend_config(args.vision_config) if args.vision_config else None
    vision_client = AsyncLLMClient(vision_config) if vision_config else None
    vision_teacher_name = args.vision_teacher_name or (vision_config.model.replace('/', '_') if vision_config else '')
    router = TeacherRouter(
        text_teacher_name=text_teacher_name,
        text_teacher_client=text_client,
        text_teacher_config=text_config,
        vision_teacher_name=vision_teacher_name or None,
        vision_teacher_client=vision_client,
        vision_teacher_config=vision_config,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(seed: dict[str, Any]) -> dict[str, Any] | None:
        if seed['id'] in existing_ids:
            return None
        async with semaphore:
            routed = router.choose(seed)
            agent = ReActAgent(routed.client, ToolRegistry.default(), AgentConfig(max_steps=args.max_steps, allow_multimodal=True))
            return await collect_one(seed, agent, routed.name)

    tasks = [asyncio.create_task(guarded(seed)) for seed in seeds]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='collect'):
        episode = await task
        if not episode:
            continue
        append_jsonl(trajectory_output, episode)
        for record in episode['records']:
            append_jsonl(records_output, {'question_id': episode['id'], 'teacher': teacher_name, **record})


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
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_collection(parse_args()))


if __name__ == '__main__':
    main()
