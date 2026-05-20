# BOPD-style on-policy collection for SFT and DPO.

from __future__ import annotations

import argparse
import asyncio
import json
import re
from typing import Any

from tqdm import tqdm

from distill.common import append_jsonl, read_jsonl
from distill.harness.agent import AgentConfig, ReActAgent
from distill.harness.llm_client import AsyncLLMClient, load_backend_config
from distill.harness.teacher_router import TeacherRouter
from distill.harness.prompts import build_judge_prompt, build_teacher_critique_prompt
from distill.harness.tools import ToolRegistry


def trajectory_to_text(records: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for record in records:
        role = record.get('role', 'unknown')
        content = record.get('content', '')
        if isinstance(content, str) and 'data:image' in content:
            content = '[multimodal user input omitted]'
        if isinstance(content, str) and len(content) > 2000:
            content = content[:2000] + ' ... [truncated]'
        tool_calls = record.get('tool_calls')
        fn_name = record.get('fn_name')
        if tool_calls:
            parts.append(f'[{role} tool_calls] {json.dumps(tool_calls, ensure_ascii=False)[:2000]}')
        elif role == 'tool' and fn_name:
            parts.append(f'[{role}:{fn_name}] {content}')
        else:
            parts.append(f'[{role}] {content}')
    return ' '.join(parts)


def parse_judge_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped)
        stripped = re.sub(r'\s*```$', '', stripped)
    match = re.search(r'\{.*\}', stripped, flags=re.DOTALL)
    if match:
        stripped = match.group(0)
    try:
        parsed = json.loads(stripped)
        return {'correct': bool(parsed.get('correct', False)), 'rationale': str(parsed.get('rationale', '')).strip()}
    except json.JSONDecodeError:
        return {'correct': False, 'rationale': content}


async def judge_prediction(client: AsyncLLMClient, seed: dict[str, Any], prediction: str) -> dict[str, Any]:
    prompt = build_judge_prompt(seed['question'], seed.get('answer', ''), prediction)
    response = await client.chat_completion(
        messages=[{'role': 'system', 'content': 'Return valid JSON only.'}, {'role': 'user', 'content': prompt}],
        tools=None,
        max_tokens=512,
    )
    content = response['content'].strip()
    return parse_judge_json(content)


async def rewrite_with_teacher(client: AsyncLLMClient, seed: dict[str, Any], failed_records: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = build_teacher_critique_prompt(seed['question'], seed.get('answer', ''), trajectory_to_text(failed_records))
    critique = await client.chat_completion(
        messages=[{'role': 'system', 'content': 'You critique student traces.'}, {'role': 'user', 'content': prompt}],
        tools=None,
        max_tokens=800,
    )
    user_content: Any = seed['question']
    if seed.get('image_url'):
        user_content = [
            {'type': 'text', 'text': seed['question']},
            {'type': 'image_url', 'image_url': {'url': seed['image_url']}},
        ]
    rewritten = await client.chat_completion(
        messages=[
            {
                'role': 'system',
                'content': (
                    'Rewrite the failed answer into a correct final assistant answer. '
                    f'The gold answer is: {seed.get("answer", "")}. '
                    'Return only the corrected answer, with no Markdown, explanations, citations, or <answer> tags.'
                ),
            },
            {'role': 'user', 'content': user_content},
        ],
        tools=None,
        max_tokens=2048,
    )
    return {'critique': critique['content'], 'teacher_rewrite': rewritten['content']}


async def process_seed(
    seed: dict[str, Any],
    student_agent: ReActAgent,
    judge_client: AsyncLLMClient,
    teacher_router: TeacherRouter,
) -> dict[str, Any]:
    student_result = await student_agent.run(
        question_id=seed['id'],
        question=seed['question'],
        image=seed.get('image') or None,
        image_path=seed.get('image_path') or None,
        image_url=seed.get('image_url') or None,
        image_b64=seed.get('image_b64') or None,
        metadata={'answer': seed.get('answer', '')},
    )
    judge = await judge_prediction(judge_client, seed, student_result.prediction)
    if judge.get('correct', False):
        return {}
    routed_teacher = teacher_router.choose(seed)
    rewrite = await rewrite_with_teacher(routed_teacher.client, seed, student_result.records)
    return {
        'id': seed['id'],
        'question': seed['question'],
        'answer': seed.get('answer', ''),
        'teacher': routed_teacher.name,
        'image': seed.get('image', ''),
        'image_path': seed.get('image_path', ''),
        'image_url': seed.get('image_url', ''),
        'image_b64': seed.get('image_b64', ''),
        'student_failed': {'prediction': student_result.prediction, 'records': student_result.records},
        'judge': judge,
        'teacher_rewritten': rewrite['teacher_rewrite'],
        'teacher_critique': rewrite['critique'],
        'dpo_pair': {'chosen': rewrite['teacher_rewrite'], 'rejected': student_result.prediction or trajectory_to_text(student_result.records)},
    }


async def run_collection(args: argparse.Namespace) -> None:
    seeds = read_jsonl(args.seeds)
    student_client = AsyncLLMClient(load_backend_config(args.student_config))
    text_teacher_config = load_backend_config(args.teacher_config)
    text_teacher_client = AsyncLLMClient(text_teacher_config)
    vision_teacher_config = load_backend_config(args.vision_teacher_config) if args.vision_teacher_config else None
    vision_teacher_client = AsyncLLMClient(vision_teacher_config) if vision_teacher_config else None
    judge_client = AsyncLLMClient(load_backend_config(args.judge_config or args.teacher_config))
    teacher_router = TeacherRouter(
        text_teacher_name=args.teacher_name or text_teacher_config.model.replace('/', '_'),
        text_teacher_client=text_teacher_client,
        text_teacher_config=text_teacher_config,
        vision_teacher_name=args.vision_teacher_name or (vision_teacher_config.model.replace('/', '_') if vision_teacher_config else ''),
        vision_teacher_client=vision_teacher_client,
        vision_teacher_config=vision_teacher_config,
    )
    student_agent = ReActAgent(
        student_client,
        ToolRegistry.default(),
        AgentConfig(
            max_steps=args.max_steps,
            allow_multimodal=True,
            disable_tools=args.disable_tools,
            prompt_mode=args.prompt_mode,
        ),
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(seed: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await process_seed(seed, student_agent, judge_client, teacher_router)

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
    parser.add_argument('--vision-teacher-config', default='')
    parser.add_argument('--judge-config', default='')
    parser.add_argument('--teacher-name', default='')
    parser.add_argument('--vision-teacher-name', default='')
    parser.add_argument('--output', required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--disable-tools', action='store_true')
    parser.add_argument('--prompt-mode', choices=['direct_vqa', 'react'], default='direct_vqa')
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_collection(parse_args()))


if __name__ == '__main__':
    main()
