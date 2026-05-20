from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig

from .judge import heuristic_judge
from .memory_evo import EvoMemory
from .policy import evidence_state, strategy_from_state
from .refine import refine_hint
from .skills import classify_task, extract_constraints, format_skills, select_skills


TTL_PROMPT = DEFAULT_SYSTEM_PROMPT + """

Lightweight Self-Evolution 策略：
- 本分支做轻量反思和记忆注入：每题执行一次，结束后根据无 gold 轨迹信号记录经验，供后续题参考。
- 不因为本地 judge 的不确定信号直接二次完整重跑，避免把在线工具波动放大成更差答案。
- 记忆只保存失败题型、失败类型、答案槽位错误、工具使用模式和下一题可复用策略；不保存 gold 答案。
- 记忆只用于避免重复错误和改进最终答案槽位选择，不能降低题目的最低证据要求。
- Qwen3-32B 只可作为评测 judge；本分支默认不把它作为 Harness 基座模型或逐题反思重写器。
- 策略层会记录重复搜索、低可信来源、浏览器失败和时间约束遗漏；后续题必须把这些记录当成行为约束，而不是复用旧答案。

Track-D 证据规划：
- 图像地标、建筑、艺术品、书籍封面、人物、动植物、品牌/物体识别：先直接观察图片，再调用 search_image 获取候选；如果候选不唯一，抽取可见文字/风格/地点特征，用 search_text 复查，不要只凭视觉猜实体。
- 图像中的年份、作者、馆藏地、国籍、起源国家等知识题：通常需要 search_image 或 search_text 交叉验证；最终只输出问题要求的字段。
- 图像中的空间位置、场景类别、物体类别、生物关系、材质/现象类型等可直接视觉判断的问题，优先根据图片回答；不要为了形式搜索后被网页热门结果带偏。
- 图像事实题先提取图片锚点：可见文字、标题、logo、人物、物体、地标、海报名或作品名；搜索 query 必须包含该锚点和题目要求的字段。
- 如果 search_image/search_text 给出多个候选，必须用图片锚点排除不匹配项；不能直接选择搜索结果第一名。
- 2wiki、多跳比较、亲属/导演/出生地/电影年份等纯文本题：先识别题型和需要填的事实槽位（比较早晚、国籍、出生/死亡地点、亲属、导演、电影年份等），按实体关系分解后作答；只有题目给出的信息不足或实体非常生僻时，再用 search_text/browser 工具补证。
- 对简单常识且你非常确定的题可以直接答；但如果题目包含专名、日期、地点、电影、书籍、艺术品或“which/who/where/year/nationality”等事实槽位，至少使用一次搜索或浏览器验证。
- 发散要受控：第一轮最多 1 个图搜或 1-2 个文搜；第二轮只围绕最可信候选补证，不要无限扩大关键词。
- 避免同义反复搜索：如果 search_text 已覆盖同一实体/关系，下一步必须增加缺失的时间、地点、关系、候选名或可信域名；否则应停止并回答。
- 低可信源过滤：标题和内容明显不匹配、PDF/答案站/聚合站、搜索片段像拼接语料、或同域多次 403/500 的来源，不应作为事实锚点。
- 对需要链式追踪的题，显式维护事实锚点：实体 -> 文档/页面 -> 目标关系 -> 时间约束 -> 最终答案槽位。
- 对 2wiki/context 已给出足够证据的题，优先使用题目 context 推理，不要为了形式额外搜索；但如果搜索了，最终答案必须回到题目要求的槽位。
- 对含 first/before/after/as of/season/shown/current 的题，把时间约束放进搜索 query；不要用后续新闻覆盖题目要求的时间点。
- 对图像题，先锁定图中实体候选，再查目标属性；如果后续搜索出现不同实体，必须用图像线索和题目约束交叉验证后才能切换。
- 收尾时先判断问题要求的答案槽位：人名、地点、年份、国家、缩写、法案编号、数值或实体名。最终答案只能填这个槽位。
- 年份题只输出年份；数量题只输出数字和必要单位；关系题只输出关系名；类别题只输出类别名；地点题不要输出搜索页面、国家泛称或完整解释。
- 如果隐藏思考或证据里已有短候选答案，且它比当前答案更匹配槽位，优先候选答案；不要把公司、来源页面、工具标签或解释文本当最终答案。
- 最终答案直接回答问题所问；需要年份就给年份，需要地点/人物/实体就给对应名称，需要 yes/no 就明确回答并保留必要限定。不要输出 <answer> 标签、证据列表或工具调用文本。
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    memory_path = Path(runtime.output_dir) / "track_d_slot_evo" / "memory" / "evo_memory.jsonl"
    memory = EvoMemory(str(memory_path))
    evolution_enabled = not runtime.disable_memory or not runtime.disable_reflection
    skills = select_skills(task) if evolution_enabled else []
    skill_context = format_skills(skills) if evolution_enabled else ""
    memory_context = ""
    memory_hits: list[dict[str, Any]] = []
    if evolution_enabled and not runtime.benchmark_mode and not runtime.disable_memory:
        memory_context, memory_hits = memory.retrieve(task, k=1)
    merged_memory_context = "\n\n".join(part for part in [skill_context, memory_context] if part.strip())
    tuned = replace(runtime, track_name="track_d_slot_evo")
    result = run_react_task(
        task,
        tuned,
        system_prompt=TTL_PROMPT,
        memory_context=merged_memory_context,
        reflection_context="",
        track_name="track_d_slot_evo",
        expose_reasoning_candidates=evolution_enabled,
        final_answer_review=evolution_enabled,
    )
    final_judge = (
        {"pass": True, "confidence": 1.0, "failure_type": "reflection_disabled", "rationale": "reflection disabled"}
        if runtime.disable_reflection
        else heuristic_judge(task, result.pred, result.trajectory)
    )
    policy_state = evidence_state(task, result.trajectory)
    policy_strategy = strategy_from_state(policy_state)
    reflection_hint = "" if runtime.disable_reflection else refine_hint(final_judge)
    if reflection_hint and policy_strategy:
        reflection_hint = f"{reflection_hint} {policy_strategy}"
    elif policy_strategy:
        reflection_hint = policy_strategy
    memory_item = memory.build_item(task, result, final_judge, reflection_hint=reflection_hint)
    result.debug.update(
        {
            "initial_judge": final_judge,
            "final_judge": final_judge,
            "reflection_triggered": not runtime.disable_reflection,
            "reflection_hint": memory.reflection_from_signal(task, result, final_judge),
            "repair_strategy": reflection_hint,
            "task_type": classify_task(task),
            "constraints": extract_constraints(task),
            "policy_state": policy_state,
            "skills": [skill.get("name") for skill in skills],
            "memory_hits": memory_hits,
            "memory_update": memory_item if runtime.allow_evolution_updates and not runtime.benchmark_mode and not runtime.disable_memory and not final_judge.get("pass") else {},
            "external_assist": {},
        }
    )
    if runtime.allow_evolution_updates and not runtime.benchmark_mode and not runtime.disable_memory and not final_judge.get("pass"):
        memory.update(memory_item)
    return result
