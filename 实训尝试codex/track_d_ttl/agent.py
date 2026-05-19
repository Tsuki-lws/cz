from __future__ import annotations

from dataclasses import replace
from typing import Any

from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig

from .judge import heuristic_judge
from .memory_online import OnlineMemory
from .refine import refine_hint


TTL_PROMPT = DEFAULT_SYSTEM_PROMPT + """

Test-Time Learning 策略：
- 每个样本只能在当前处理时 refine，不能处理完后回头。
- 根据无 gold judge 信号调整工具预算和计划。
- 经验只能顺序写入 memory，供后续样本参考。
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    memory = OnlineMemory()
    memory_context = "" if runtime.benchmark_mode else memory.retrieve()
    tuned = replace(runtime, track_name="track_d")
    first = run_react_task(task, tuned, system_prompt=TTL_PROMPT, memory_context=memory_context, track_name="track_d")
    judge = heuristic_judge(task, first.pred, first.trajectory)
    if judge.get("pass") or runtime.max_steps <= 4:
        result = first
    else:
        second = run_react_task(
            task,
            replace(tuned, max_steps=max(4, runtime.max_steps // 2)),
            system_prompt=TTL_PROMPT,
            memory_context=memory_context,
            reflection_context=refine_hint(judge),
            track_name="track_d",
        )
        second.trajectory = first.trajectory + second.trajectory
        second.metrics["tokens"] = first.metrics.get("tokens", 0) + second.metrics.get("tokens", 0)
        second.metrics["turns"] = first.metrics.get("turns", 0) + second.metrics.get("turns", 0)
        second.metrics["tool_calls"] = first.metrics.get("tool_calls", 0) + second.metrics.get("tool_calls", 0)
        second.metrics["latency"] = first.metrics.get("latency", 0) + second.metrics.get("latency", 0)
        result = second
    final_judge = heuristic_judge(task, result.pred, result.trajectory)
    result.debug.update({"initial_judge": judge, "final_judge": final_judge})
    if runtime.allow_evolution_updates and not runtime.benchmark_mode:
        memory.update({"index": result.index, "lesson": f"failure={final_judge.get('failure_type')} pred_nonempty={bool(result.pred)}"})
    return result
