# Minimal ReAct harness that emits task-aligned trajectories.

from __future__ import annotations

import time
from dataclasses import dataclass
import re
from typing import Any

from distill.harness.llm_client import AsyncLLMClient
from distill.harness.media import build_user_content
from distill.harness.memory import SkillMemory, format_skill_instructions
from distill.harness.prompts import FORCE_ANSWER_PROMPT, build_system_prompt
from distill.harness.tools import ToolRegistry
from distill.harness.trajectory import sanitize_tool_calls, utc_timestamp


@dataclass(slots=True)
class AgentConfig:
    max_steps: int = 8
    tool_choice: str = 'auto'
    extra_system_instruction: str | None = None
    allow_multimodal: bool = True
    disable_tools: bool = False
    prompt_mode: str = 'direct_vqa'
    skill_memory_path: str | None = None
    max_memory_skills: int = 3


@dataclass(slots=True)
class AgentResult:
    question_id: str
    question: str
    answer: str
    prediction: str
    records: list[dict[str, Any]]
    elapsed_seconds: float
    turns: int
    total_tokens: int
    tool_call_count: int


class ReActAgent:
    def __init__(self, client: AsyncLLMClient, tools: ToolRegistry, config: AgentConfig | None = None):
        self.client = client
        self.tools = tools
        self.config = config or AgentConfig()
        self.skill_memory = SkillMemory.load(self.config.skill_memory_path)

    async def run(
        self,
        *,
        question_id: str,
        question: str,
        image: str | None = None,
        image_path: str | None = None,
        image_url: str | None = None,
        image_b64: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        start_time = time.perf_counter()
        memory_skills = self.skill_memory.retrieve(question, top_k=self.config.max_memory_skills) if self.skill_memory else []
        memory_instruction = format_skill_instructions(memory_skills)
        extra_instruction = self.config.extra_system_instruction or ''
        if memory_instruction:
            extra_instruction = (extra_instruction + '\n\n' + memory_instruction).strip()
        system_prompt = build_system_prompt(extra_instruction or None, mode=self.config.prompt_mode)
        user_content = build_user_content(
            question,
            image=image or '',
            image_path=image_path or '',
            image_url=image_url or '',
            image_b64=image_b64 or '',
            allow_multimodal=self.config.allow_multimodal,
        )

        messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ]
        records: list[dict[str, Any]] = [
            {
                'timestamp': utc_timestamp(),
                'step_id': 0,
                'role': 'user',
                'content': user_content if isinstance(user_content, str) else str(user_content),
                'tool_call_id': None,
                'tool_calls': None,
                'reasoning_content': None,
                'total_tokens': None,
                'fn_name': None,
                'fn_args': None,
                'memory_skills': [skill.get('name') for skill in memory_skills],
            }
        ]

        total_tokens = 0
        tool_call_count = 0
        step_id = 1
        final_content = ''

        for _ in range(self.config.max_steps):
            if step_id >= self.config.max_steps:
                messages.append({'role': 'system', 'content': FORCE_ANSWER_PROMPT})
            response = await self.client.chat_completion(
                messages=messages,
                tools=None if self.config.disable_tools else self.tools.openai_tools(),
                tool_choice=self.config.tool_choice,
            )
            content = response['content'] or ''
            tool_calls = sanitize_tool_calls(response['tool_calls'])
            reasoning_content = response.get('reasoning_content')
            fallback_content = content or (reasoning_content or '')
            used_tokens = response.get('total_tokens') or 0
            total_tokens += used_tokens

            records.append(
                {
                    'timestamp': utc_timestamp(),
                    'step_id': step_id,
                    'role': 'assistant',
                    'content': content,
                    'tool_call_id': None,
                    'tool_calls': tool_calls or None,
                    'reasoning_content': reasoning_content,
                    'total_tokens': used_tokens,
                    'fn_name': None,
                    'fn_args': None,
                }
            )
            step_id += 1

            assistant_message: dict[str, Any] = {'role': 'assistant', 'content': fallback_content}
            if tool_calls:
                assistant_message['tool_calls'] = tool_calls
            messages.append(assistant_message)

            if not tool_calls:
                final_content = fallback_content
                break

            for tool_call in tool_calls:
                tool_call_count += 1
                result = await self.tools.run_tool_call(tool_call)
                records.append(
                    {
                        'timestamp': utc_timestamp(),
                        'step_id': step_id,
                        'role': 'tool',
                        'content': result['content'],
                        'tool_call_id': tool_call.get('id'),
                        'tool_calls': None,
                        'reasoning_content': None,
                        'total_tokens': None,
                        'fn_name': result['tool_name'],
                        'fn_args': result['arguments'],
                    }
                )
                step_id += 1
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': tool_call.get('id'),
                        'content': result['content'],
                    }
                )

        if not final_content and records:
            final_content = records[-1]['content'] or ''

        prediction = extract_answer(final_content)
        elapsed = time.perf_counter() - start_time
        return AgentResult(
            question_id=question_id,
            question=question,
            answer=(metadata or {}).get('answer', ''),
            prediction=prediction,
            records=records,
            elapsed_seconds=elapsed,
            turns=sum(1 for record in records if record['role'] == 'assistant'),
            total_tokens=total_tokens,
            tool_call_count=tool_call_count,
        )


def extract_answer(content: str) -> str:
    content = (content or '').strip()
    if not content:
        return ''
    if '<answer>' in content and '</answer>' in content:
        return content.split('<answer>', 1)[1].split('</answer>', 1)[0].strip()
    if '<answer>' in content:
        tail = content.split('<answer>', 1)[1]
        return tail.splitlines()[0].strip()

    explicit_patterns = [
        re.compile(r'(?:final answer|answer is|answer|答案是|答案为|最终答案)[:：]\s*([^\n。]+)', re.IGNORECASE),
        re.compile(r'(?:so|therefore),?\s+the answer is\s+([^\n。]+)', re.IGNORECASE),
    ]
    for pattern in explicit_patterns:
        matches = pattern.findall(content)
        if matches:
            answer = matches[-1].strip().strip(' .。；;，,')
            if answer:
                return answer

    bold_matches = re.findall(r'\*\*([^*\n]{1,120})\*\*', content)
    if bold_matches:
        return bold_matches[-1].strip().strip(' .。；;，,')

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ''
    bad_prefixes = (
        'based on',
        'the key',
        'key features',
        'the image',
        'this image',
        'let me',
    )
    concise_lines = [
        line
        for line in lines
        if len(line) <= 120
        and not line.startswith(('-', '*'))
        and not line.endswith((':', '：'))
        and not line.lower().startswith(bad_prefixes)
    ]
    if concise_lines:
        return concise_lines[-1].strip().strip(' .。；;，,')
    return lines[0].strip()
