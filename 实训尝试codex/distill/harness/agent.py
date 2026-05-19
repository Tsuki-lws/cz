# Minimal ReAct harness that emits task-aligned trajectories.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from distill.harness.llm_client import AsyncLLMClient
from distill.harness.prompts import build_system_prompt
from distill.harness.tools import ToolRegistry
from distill.harness.trajectory import sanitize_tool_calls, utc_timestamp


@dataclass(slots=True)
class AgentConfig:
    max_steps: int = 8
    tool_choice: str = 'auto'
    extra_system_instruction: str | None = None


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

    async def run(
        self,
        *,
        question_id: str,
        question: str,
        image: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        start_time = time.perf_counter()
        system_prompt = build_system_prompt(self.config.extra_system_instruction)
        user_content = question if not image else f'Question: {question}\nImage: {image}'

        messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ]
        records: list[dict[str, Any]] = [
            {
                'timestamp': utc_timestamp(),
                'step_id': 0,
                'role': 'user',
                'content': user_content,
                'tool_call_id': None,
                'tool_calls': None,
                'reasoning_content': None,
                'total_tokens': None,
                'fn_name': None,
                'fn_args': None,
            }
        ]

        total_tokens = 0
        tool_call_count = 0
        step_id = 1
        final_content = ''

        for _ in range(self.config.max_steps):
            response = await self.client.chat_completion(
                messages=messages,
                tools=self.tools.openai_tools(),
                tool_choice=self.config.tool_choice,
            )
            content = response['content'] or ''
            tool_calls = sanitize_tool_calls(response['tool_calls'])
            reasoning_content = response.get('reasoning_content')
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

            assistant_message: dict[str, Any] = {'role': 'assistant', 'content': content}
            if tool_calls:
                assistant_message['tool_calls'] = tool_calls
            messages.append(assistant_message)

            if not tool_calls:
                final_content = content
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
    if '<answer>' in content and '</answer>' in content:
        return content.split('<answer>', 1)[1].split('</answer>', 1)[0].strip()
    return content.strip()
