from __future__ import annotations

from dataclasses import replace
from typing import Any

from shared_sii_adapter.external_assist import build_reflection_hint_with_external_model, organize_memory_with_external_model
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

Track-D 证据规划：
- 图像地标、建筑、艺术品、书籍封面、人物、动植物、品牌/物体识别：先直接观察图片，再调用 search_image 获取候选；如果候选不唯一，抽取可见文字/风格/地点特征，用 search_text 复查，不要只凭视觉猜实体。
- 图像中的年份、作者、馆藏地、国籍、起源国家等知识题：通常需要 search_image 或 search_text 交叉验证；最终只输出问题要求的字段。
- 2wiki、多跳比较、亲属/导演/出生地/电影年份等纯文本题：先识别题型和需要填的事实槽位（比较早晚、国籍、出生/死亡地点、亲属、导演、电影年份等），按实体关系分解后作答；只有题目给出的信息不足或实体非常生僻时，再用 search_text/browser 工具补证。
- 对简单常识且你非常确定的题可以直接答；但如果题目包含专名、日期、地点、电影、书籍、艺术品或“which/who/where/year/nationality”等事实槽位，至少使用一次搜索或浏览器验证。
- 发散要受控：第一轮最多 1 个图搜或 1-2 个文搜；第二轮只围绕最可信候选补证，不要无限扩大关键词。
- 最终答案直接回答问题所问；需要年份就给年份，需要地点/人物/实体就给对应名称，需要 yes/no 就明确回答并保留必要限定。不要输出 <answer> 标签、证据列表或工具调用文本。
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    memory = OnlineMemory()
    memory_context = "" if runtime.benchmark_mode or runtime.disable_memory else memory.retrieve()
    tuned = replace(runtime, track_name="track_d")
    first = run_react_task(task, tuned, system_prompt=TTL_PROMPT, memory_context=memory_context, track_name="track_d")
    judge = {"pass": True, "confidence": 1.0, "failure_type": "reflection_disabled", "rationale": "reflection disabled"} if runtime.disable_reflection else heuristic_judge(task, first.pred, first.trajectory)
    if runtime.disable_reflection or judge.get("pass") or runtime.max_steps <= 4:
        result = first
    else:
        external_hint = build_reflection_hint_with_external_model(
            runtime=runtime,
            track_name="track_d",
            task=task,
            result=first,
            local_signal=judge,
        )
        hint = refine_hint(judge)
        if external_hint:
            hint += "\n外部反思补充：" + external_hint
        second = run_react_task(
            task,
            replace(tuned, max_steps=max(4, runtime.max_steps // 2)),
            system_prompt=TTL_PROMPT,
            memory_context=memory_context,
            reflection_context=hint,
            track_name="track_d",
        )
        second.trajectory = first.trajectory + second.trajectory
        second.metrics["tokens"] = first.metrics.get("tokens", 0) + second.metrics.get("tokens", 0)
        second.metrics["turns"] = first.metrics.get("turns", 0) + second.metrics.get("turns", 0)
        second.metrics["tool_calls"] = first.metrics.get("tool_calls", 0) + second.metrics.get("tool_calls", 0)
        second.metrics["latency"] = first.metrics.get("latency", 0) + second.metrics.get("latency", 0)
        result = second
    final_judge = judge if runtime.disable_reflection else heuristic_judge(task, result.pred, result.trajectory)
    assist = organize_memory_with_external_model(
        runtime=runtime,
        track_name="track_d",
        task=task,
        result=result,
        local_signal={"initial_judge": judge, "final_judge": final_judge},
    )
    result.debug.update({"initial_judge": judge, "final_judge": final_judge, "external_assist": assist})
    if runtime.allow_evolution_updates and not runtime.benchmark_mode and not runtime.disable_memory:
        memory.update({"index": result.index, "lesson": assist.get("lesson") or f"failure={final_judge.get('failure_type')} pred_nonempty={bool(result.pred)}"})
    return result
