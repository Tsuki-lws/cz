# OpenAI-compatible async client wrapper.

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from distill.common import load_yaml


@dataclass(slots=True)
class LLMBackendConfig:
    backend: str
    model: str
    base_url: str
    api_key_env: str = 'OPENAI_API_KEY'
    timeout: int = 180
    max_retries: int = 5
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 4096
    enable_thinking: bool = False

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, 'EMPTY')


class AsyncLLMClient:
    def __init__(self, config: LLMBackendConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,
        )

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = 'auto',
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'model': self.config.model,
            'messages': messages,
            'temperature': self.config.temperature if temperature is None else temperature,
            'top_p': self.config.top_p if top_p is None else top_p,
            'max_tokens': self.config.max_tokens if max_tokens is None else max_tokens,
            'extra_body': {
                'enable_thinking': self.config.enable_thinking,
                'chat_template_kwargs': {'enable_thinking': self.config.enable_thinking},
            },
        }
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = tool_choice
        if response_format:
            payload['response_format'] = response_format

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(**payload)
                choice = response.choices[0]
                message = choice.message
                return {
                    'content': self._extract_content(message),
                    'tool_calls': self._extract_tool_calls(message),
                    'reasoning_content': getattr(message, 'reasoning_content', None),
                    'finish_reason': choice.finish_reason,
                    'usage': getattr(response, 'usage', None),
                    'total_tokens': getattr(getattr(response, 'usage', None), 'total_tokens', None),
                    'raw': response,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 == self.config.max_retries:
                    break
                await asyncio.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f'chat completion failed after retries: {last_error}') from last_error

    @staticmethod
    def _extract_content(message: Any) -> str:
        content = getattr(message, 'content', '')
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: list[str] = []
            for block in content:
                if hasattr(block, 'model_dump'):
                    block = block.model_dump()
                if isinstance(block, dict) and block.get('type') == 'text':
                    fragments.append(block.get('text', ''))
            return '\n'.join(fragment for fragment in fragments if fragment)
        return str(content or '')

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
        tool_calls = getattr(message, 'tool_calls', None) or []
        normalized: list[dict[str, Any]] = []
        for call in tool_calls:
            if hasattr(call, 'model_dump'):
                call = call.model_dump()
            normalized.append(call)
        return normalized


def load_backend_config(path: str) -> LLMBackendConfig:
    data = load_yaml(path)
    return LLMBackendConfig(**data)
