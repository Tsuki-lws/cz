from __future__ import annotations

from dataclasses import replace
from typing import Any

from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig

from .adaptive_steps import max_steps_for
from .reflection_lite import build_reflection_hint


ENGINEERING_PROMPT = DEFAULT_SYSTEM_PROMPT + """

工程基线策略：
- 先判断题目是否需要搜索、图搜或浏览器。
- 搜索结果不足时换关键词，不重复同一个 query。
- 浏览多个候选网页时优先使用 browser_parallel。
- 如果工具失败，换工具或缩小查询范围。
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    tuned = replace(runtime, max_steps=max_steps_for(task, runtime.max_steps), track_name="track_e")
    return run_react_task(
        task,
        tuned,
        system_prompt=ENGINEERING_PROMPT,
        reflection_context=build_reflection_hint(task),
        track_name="track_e",
    )
