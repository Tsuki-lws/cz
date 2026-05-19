from __future__ import annotations

from dataclasses import replace
from typing import Any

from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig


DISTILL_PROMPT = DEFAULT_SYSTEM_PROMPT + """

蒸馏路线执行策略：
- 这里代表蒸馏后 student 的推理入口；若尚未部署 SFT 权重，则使用当前 Qwen3.5-9B 基座。
- 保持工具调用和轨迹格式与 teacher 采集一致，方便后续 SFT/DPO/BOPD 数据闭环。
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    return run_react_task(task, replace(runtime, track_name="track_a"), system_prompt=DISTILL_PROMPT, track_name="track_a")

