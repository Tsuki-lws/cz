from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig

from .judge import heuristic_judge
from .memory_evo import EvoMemory
from .skills import classify_task, extract_constraints, format_skills, select_skills


TTL_PROMPT = DEFAULT_SYSTEM_PROMPT + """

Lightweight Self-Evolution 策略：
- 本分支只做轻量反思和记忆注入：每题执行一次，结束后根据无 gold 轨迹信号记录经验，供后续题参考。
- 不因为本地 judge 的不确定信号直接二次完整重跑，避免把在线工具波动放大成更差答案。
- 记忆只保存题型、失败类型、工具使用模式和下一题可复用策略；benchmark/test 提交模式不写入跨样本记忆。
- Qwen3-32B 只可作为评测 judge；本分支默认不把它作为 Harness 基座模型或逐题反思重写器。

Track-D 证据规划：
- 图像地标、建筑、艺术品、书籍封面、人物、动植物、品牌/物体识别：先直接观察图片，再调用 search_image 获取候选；如果候选不唯一，抽取可见文字/风格/地点特征，用 search_text 复查，不要只凭视觉猜实体。
- 图像中的年份、作者、馆藏地、国籍、起源国家等知识题：通常需要 search_image 或 search_text 交叉验证；最终只输出问题要求的字段。
- 2wiki、多跳比较、亲属/导演/出生地/电影年份等纯文本题：先识别题型和需要填的事实槽位（比较早晚、国籍、出生/死亡地点、亲属、导演、电影年份等），按实体关系分解后作答；只有题目给出的信息不足或实体非常生僻时，再用 search_text/browser 工具补证。
- 对简单常识且你非常确定的题可以直接答；但如果题目包含专名、日期、地点、电影、书籍、艺术品或“which/who/where/year/nationality”等事实槽位，至少使用一次搜索或浏览器验证。
- 发散要受控：第一轮最多 1 个图搜或 1-2 个文搜；第二轮只围绕最可信候选补证，不要无限扩大关键词。
- 对含 first/before/after/as of/season/shown/current 的题，把时间约束放进搜索 query；不要用后续新闻覆盖题目要求的时间点。
- 对图像题，先锁定图中实体候选，再查目标属性；如果后续搜索出现不同实体，必须用图像线索和题目约束交叉验证后才能切换。
- 最终答案直接回答问题所问；需要年份就给年份，需要地点/人物/实体就给对应名称，需要 yes/no 就明确回答并保留必要限定。不要输出 <answer> 标签、证据列表或工具调用文本。
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    memory_path = Path(runtime.output_dir) / "track_d_evo" / "memory" / "evo_memory.jsonl"
    memory = EvoMemory(str(memory_path))
    evolution_enabled = not runtime.disable_memory or not runtime.disable_reflection
    skills = select_skills(task) if evolution_enabled else []
    skill_context = format_skills(skills) if evolution_enabled else ""
    memory_context = ""
    memory_hits: list[dict[str, Any]] = []
    if evolution_enabled and not runtime.benchmark_mode and not runtime.disable_memory:
        memory_context, memory_hits = memory.retrieve(task)
    merged_memory_context = "\n\n".join(part for part in [skill_context, memory_context] if part.strip())
    tuned = replace(runtime, track_name="track_d_evo")
    result = run_react_task(task, tuned, system_prompt=TTL_PROMPT, memory_context=merged_memory_context, track_name="track_d_evo")
    final_judge = (
        {"pass": True, "confidence": 1.0, "failure_type": "reflection_disabled", "rationale": "reflection disabled"}
        if runtime.disable_reflection
        else heuristic_judge(task, result.pred, result.trajectory)
    )
    result.debug.update(
        {
            "initial_judge": final_judge,
            "final_judge": final_judge,
            "reflection_triggered": not runtime.disable_reflection,
            "reflection_hint": memory.reflection_from_signal(task, result, final_judge),
            "task_type": classify_task(task),
            "constraints": extract_constraints(task),
            "skills": [skill.get("name") for skill in skills],
            "memory_hits": memory_hits,
            "external_assist": {},
        }
    )
    if runtime.allow_evolution_updates and not runtime.benchmark_mode and not runtime.disable_memory:
        memory.update(memory.build_item(task, result, final_judge))
    return result
