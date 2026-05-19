# BOPD-style on-policy collection for SFT and DPO.

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from tqdm import tqdm

from distill.common import append_jsonl, read_jsonl
from distill.harness.agent import AgentConfig, ReActAgent
from distill.harness.llm_client import AsyncLLMClient, load_backend_config
from distill.harness.prompts import build_judge_prompt, build_teacher_critique_prompt
from distill.harness.tools import ToolRegistry


def trajectory_to_text(records: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for record in records:
        role = record.get('role', 'unknown')
        content = record.get('content', '')
        parts.append(f'[{role}] {content}')
    return ' '.join(parts)


async def judge_prediction(client: AsyncLLMClient, seed: dict[str, Any], prediction: str) -> dict[str, Any]:
    prompt = build_judge_prompt(seed['question'], seed.get('answer', ''), prediction)
    response = await client.chat_completion(
        messages=[{'role': 'system', 'content': 'Return valid JSON only.'}, {'role': 'user', 'content': prompt}],
        tools=None,
        max_tokens=512,
    )
    content = response['content'].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {'correct': False, 'rationale': content}


async def rewrite_with_teacher(client: AsyncLLMClient, seed: dict[str, Any], failed_records: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = build_teacher_critique_prompt(seed['question'], seed.get('answer', ''), trajectory_to_text(failed_records))
    critique = await client.chat_completion(
        messages=[{'role': 'system', 'content': 'You critique student traces.'}, {'role': 'user', 'content': prompt}],
        tools=None,
        max_tokens=800,
    )
    rewritten = await client.chat_completion(
        messages=[
            {'role': 'system', 'content': 'Rewrite the full trajectory as a correct assistant solution ending with <answer>...</answer>.'},
            {'role': 'user', 'content': seed['question']},
        ],
        tools=None,
        max_tokens=2048,
    )
    return {'critique': critique['content'], 'teacher_rewrite': rewritten['content']}


async def process_seed(seed: dict[str, Any], student_agent: ReActAgent, judge_client: AsyncLLMClient, teacher_client: AsyncLLMClient) -> dict[str, Any]:
    student_result = await student_agent.run(
        question_id=seed['id'],
        question=seed['question'],
        image=seed.get('image') or None,
        metadata={'answer': seed.get('answer', '')},
    )
    judge = await judge_prediction(judge_client, seed, student_result.prediction)
    if judge.get('correct', False):
        return {}
    rewrite = await rewrite_with_teacher(teacher_client, seed, student_result.records)
    return {
        'id': seed['id'],
        'question': seed['question'],
        'answer': seed.get('answer', ''),
        'student_failed': {'prediction': student_result.prediction, 'records': student_result.records},
        'judge': judge,
        'teacher_rewritten': rewrite['teacher_rewrite'],
        'teacher_critique': rewrite['critique'],
        'dpo_pair': {'chosen': rewrite['teacher_rewrite'], 'rejected': trajectory_to_text(student_result.records)},
    }


async def run_collection(args: argparse.Namespace) -> None:
    seeds = read_jsonl(args.seeds)
    student_client = AsyncLLMClient(load_backend_config(args.student_config))
    teacher_client = AsyncLLMClient(load_backend_config(args.teacher_config))
    judge_client = AsyncLLMClient(load_backend_config(args.judge_config or args.teacher_config))
    student_agent = ReActAgent(student_client, ToolRegistry.default(), AgentConfig(max_steps=args.max_steps))
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(seed: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await process_seed(seed, student_agent, judge_client, teacher_client)

    tasks = [asyncio.create_task(guarded(seed)) for seed in seeds]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='on_policy'):
        item = await task
        if item:
            append_jsonl(args.output, item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect on-policy BOPD-style failed trajectories and teacher rewrites.')
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--student-config', required=True)
    parser.add_argument('--teacher-config', required=True)
    parser.add_argument('--judge-config', default='')
    parser.add_argument('--output', required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=8)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_collection(parse_args()))


if __name__ == '__main__':
    main()
