from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .evidence import compact_tool_result

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent


def _find_harness_dir() -> Path:
    candidates = [
        ROOT / "harness-sii",
        PROJECT_ROOT / "harness-sii",
        PROJECT_ROOT / "harness-sii" / "harness-sii",
    ]
    for candidate in candidates:
        if (candidate / "tools" / "search_tool.py").exists() and (candidate / "tools" / "browser_tool.py").exists():
            return candidate
    return candidates[1]


HARNESS_DIR = _find_harness_dir()
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from tools.search_tool import search_image, search_text  # type: ignore  # noqa: E402
from tools.browser_tool import (  # type: ignore  # noqa: E402
    browser_click,
    browser_get_text,
    browser_navigate,
    browser_parallel,
    browser_type,
)


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "联网文字搜索，返回标题、URL、摘要和可选正文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "fetch": {"type": "boolean", "default": True},
                    "max_chars": {"type": "integer", "default": 800},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": "图搜文。输入 http(s) 图片 URL、本地路径、data URL 或 base64 图片，返回相关网页和正文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "图片 URL、本地路径、data:image/...;base64,... 或裸 base64"},
                    "top_k": {"type": "integer", "default": 3},
                    "fetch": {"type": "boolean", "default": True},
                    "max_chars": {"type": "integer", "default": 800},
                },
                "required": ["image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "在远程浏览器中打开 URL，并可返回页面文本预览。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "wait_until": {"type": "string", "default": "domcontentloaded"},
                    "include_text": {"type": "boolean", "default": True},
                    "max_text": {"type": "integer", "default": 2000},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_text",
            "description": "获取当前页面可见文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "default": 5000},
                    "timeout": {"type": "integer", "default": 15},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_parallel",
            "description": "并发打开多个 URL，返回文本预览或正文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "default": "navigate"},
                    "max_chars": {"type": "integer", "default": 2000},
                    "wait_until": {"type": "string", "default": "domcontentloaded"},
                    "max_concurrency": {"type": "integer", "description": "同时打开的标签页数，默认上限由 MAX_BROWSER_PARALLEL_CONCURRENCY 控制", "default": 4},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "点击当前页面 CSS 选择器匹配的元素。",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "nth": {"type": "integer", "default": 0},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "向当前页面 CSS 选择器匹配的输入框输入文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "default": False},
                    "clear": {"type": "boolean", "default": True},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["selector", "text"],
            },
        },
    },
]


TOOL_FN_MAP: dict[str, Callable[..., Any]] = {
    "search_text": search_text,
    "search_image": search_image,
    "browser_navigate": browser_navigate,
    "browser_get_text": browser_get_text,
    "browser_parallel": browser_parallel,
    "browser_click": browser_click,
    "browser_type": browser_type,
}


def _call_search_image(**args: Any) -> Any:
    image = args.pop("image", None) or args.pop("image_url", None)
    return search_image(image=image, **args)


TOOL_FN_MAP["search_image"] = _call_search_image


def parse_tool_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(raw or "{}")
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def dispatch_tool(name: str, args: dict[str, Any], *, max_result_chars: int = 10000) -> dict[str, Any]:
    started = time.perf_counter()
    if name not in TOOL_FN_MAP:
        return {
            "ok": False,
            "content": f"[ERROR] Unknown tool: {name}",
            "latency_ms": 0,
            "raw": None,
        }
    try:
        raw = TOOL_FN_MAP[name](**args)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "content": compact_tool_result(raw, max_chars=max_result_chars),
            "latency_ms": latency_ms,
            "raw": raw,
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "content": f"[ERROR] Tool '{name}' raised: {type(exc).__name__}: {exc}",
            "latency_ms": latency_ms,
            "raw": None,
        }
