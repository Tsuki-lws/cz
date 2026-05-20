# Tool registry for OpenAI function calling.

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

ToolFn = Callable[..., Any] | Callable[..., Awaitable[Any]]


async def browser_get(url: str, max_chars: int = 4000) -> dict[str, Any]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
    return {
        'url': str(response.url),
        'status_code': response.status_code,
        'content': response.text[:max_chars],
    }


async def wiki_search(query: str, limit: int = 5) -> dict[str, Any]:
    params = {
        'action': 'opensearch',
        'search': query,
        'limit': limit,
        'namespace': 0,
        'format': 'json',
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get('https://en.wikipedia.org/w/api.php', params=params)
        response.raise_for_status()
    payload = response.json()
    results = []
    for title, description, url in zip(payload[1], payload[2], payload[3]):
        results.append({'title': title, 'description': description, 'url': url})
    return {'query': query, 'results': results}


async def web_search(query: str, limit: int = 5) -> dict[str, Any]:
    params = {
        'q': query,
        'format': 'json',
        'no_html': 1,
        'no_redirect': 1,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get('https://api.duckduckgo.com/', params=params)
        response.raise_for_status()
    payload = response.json()
    related = payload.get('RelatedTopics', [])
    results = []
    for item in related:
        if len(results) >= limit:
            break
        if 'Text' in item:
            results.append({'title': item.get('Text', ''), 'url': item.get('FirstURL', '')})
    if payload.get('AbstractText'):
        results.insert(0, {'title': payload.get('Heading', query), 'url': payload.get('AbstractURL', '')})
    return {'query': query, 'results': results[:limit]}


async def image_search(query: str, limit: int = 5) -> dict[str, Any]:
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': query,
        'gsrlimit': limit,
        'prop': 'pageimages|info',
        'pithumbsize': 320,
        'inprop': 'url',
        'format': 'json',
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get('https://commons.wikimedia.org/w/api.php', params=params)
        response.raise_for_status()
    pages = response.json().get('query', {}).get('pages', {})
    results = []
    for page in pages.values():
        results.append(
            {
                'title': page.get('title', ''),
                'url': page.get('fullurl', ''),
                'thumbnail': page.get('thumbnail', {}).get('source', ''),
            }
        )
    return {'query': query, 'results': results[:limit]}


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn

    def openai_schema(self) -> dict[str, Any]:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]):
        self._tools = {tool.name: tool for tool in tools}

    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    async def run_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function_block = tool_call.get('function', {})
        name = function_block.get('name', '')
        if name not in self._tools:
            raise KeyError(f'unknown tool: {name}')
        args_raw = function_block.get('arguments', '{}')
        try:
            args = json.loads(args_raw or '{}')
        except json.JSONDecodeError as exc:
            args = {}
            result = {'error': f'invalid tool arguments: {exc}', 'raw_arguments': args_raw}
        else:
            try:
                result = self._tools[name].fn(**args)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:  # noqa: BLE001
                result = {
                    'error': f'tool execution failed: {type(exc).__name__}: {exc}',
                    'tool_name': name,
                    'arguments': args,
                }
        return {
            'tool_name': name,
            'arguments': args,
            'content': json.dumps(result, ensure_ascii=False),
        }

    @classmethod
    def default(cls) -> 'ToolRegistry':
        return cls(
            [
                ToolSpec(
                    name='web_search',
                    description='Search the web for concise factual leads.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'limit': {'type': 'integer', 'default': 5},
                        },
                        'required': ['query'],
                    },
                    fn=web_search,
                ),
                ToolSpec(
                    name='wiki_search',
                    description='Search Wikipedia for encyclopedic evidence.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'limit': {'type': 'integer', 'default': 5},
                        },
                        'required': ['query'],
                    },
                    fn=wiki_search,
                ),
                ToolSpec(
                    name='browser_get',
                    description='Fetch a web page and return a truncated text snapshot.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'url': {'type': 'string'},
                            'max_chars': {'type': 'integer', 'default': 4000},
                        },
                        'required': ['url'],
                    },
                    fn=browser_get,
                ),
                ToolSpec(
                    name='image_search',
                    description='Search Wikimedia Commons for candidate images.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'limit': {'type': 'integer', 'default': 5},
                        },
                        'required': ['query'],
                    },
                    fn=image_search,
                ),
            ]
        )
