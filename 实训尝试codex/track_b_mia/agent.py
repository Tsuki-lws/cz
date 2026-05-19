from __future__ import annotations

from dataclasses import replace
from typing import Any

from shared_sii_adapter.external_assist import organize_memory_with_external_model
from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig

from .manager import classify_task
from .memory.workflow_buffer import WorkflowBuffer
from .planner import build_plan
from .reviewer_ac import review_without_gold


MIA_PROMPT = DEFAULT_SYSTEM_PROMPT + """

MIA 多智能体执行策略：
- Manager 已给出任务类型和预算。
- Planner 已给出计划；严格按计划执行，但如果证据不足可以调整 query。
- Executor 必须使用 Qwen3.5-9B 和工具完成最终回答。
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    manager_state = classify_task(task)
    plan = build_plan(task, manager_state)
    memory = WorkflowBuffer()
    memory_hits = [] if runtime.benchmark_mode else memory.retrieve(task)
    memory_context = "\n".join(str(item.get("summary", item)) for item in memory_hits)
    reflection_context = f"Manager: {manager_state}\nPlan:\n{plan}"
    tuned = replace(runtime, track_name="track_b")
    result = run_react_task(
        task,
        tuned,
        system_prompt=MIA_PROMPT,
        memory_context=memory_context,
        reflection_context=reflection_context,
        track_name="track_b",
    )
    review = review_without_gold(task, result.pred, result.trajectory, runtime)
    assist = organize_memory_with_external_model(
        runtime=runtime,
        track_name="track_b",
        task=task,
        result=result,
        local_signal={"manager": manager_state, "review": review},
    )
    result.debug.update({"manager": manager_state, "plan": plan, "review": review, "memory_hits": memory_hits, "external_assist": assist})
    if runtime.allow_evolution_updates and not runtime.benchmark_mode:
        memory.add(
            {
                "index": result.index,
                "task_type": manager_state["task_type"],
                "summary": assist.get("lesson") or f"type={manager_state['task_type']} pred={result.pred[:100]} pass={review.get('pass')}",
                "review": review,
                "external_assist": assist,
            }
        )
    return result
