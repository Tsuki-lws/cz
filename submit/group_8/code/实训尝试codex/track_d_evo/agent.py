from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.external_assist import organize_memory_with_external_model
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig

from .judge import heuristic_judge
from .memory_evo import EvoMemory
from .skills import classify_task, extract_constraints, format_skills, select_skills


TTL_PROMPT = DEFAULT_SYSTEM_PROMPT + """

Track-D lightweight evolution:
- Run once per sample. Use prior memory only as a short planning hint; never copy old answers.
- Plan tool order before calling tools.
- Visual task: image understanding/search_image once -> one targeted search_text for the requested attribute -> answer.
- Text multi-hop/2wiki: identify entity slots first -> use 1-3 focused search_text queries; open pages only when snippets conflict or are insufficient.
- Stop as soon as one plausible answer is supported. Avoid broadening after a candidate matches the question slot.
- If a tool fails or budget is low, use gathered snippets and answer the most likely value instead of looping.
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    memory_path = Path(runtime.output_dir) / "track_d_evo" / "memory" / "evo_memory.jsonl"
    memory = EvoMemory(str(memory_path))
    evolution_enabled = not runtime.disable_memory or not runtime.disable_reflection
    skills = select_skills(task, k=2) if evolution_enabled else []
    skill_context = format_skills(skills) if evolution_enabled else ""
    memory_context = ""
    memory_hits: list[dict[str, Any]] = []
    if evolution_enabled and not runtime.benchmark_mode and not runtime.disable_memory:
        memory_context, memory_hits = memory.retrieve(task)
    merged_memory_context = "\n\n".join(part for part in [skill_context, memory_context] if part.strip())
    tuned = replace(runtime, track_name="track_d_evo", max_steps=runtime.max_steps, max_tokens=runtime.max_tokens)
    result = run_react_task(task, tuned, system_prompt=TTL_PROMPT, memory_context=merged_memory_context, track_name="track_d_evo")
    final_judge = (
        {"pass": True, "confidence": 1.0, "failure_type": "reflection_disabled", "rationale": "reflection disabled"}
        if runtime.disable_reflection
        else heuristic_judge(task, result.pred, result.trajectory)
    )
    reflection_hint = memory.reflection_from_signal(task, result, final_judge)
    external_payload = organize_memory_with_external_model(
        runtime=runtime,
        track_name="track_d_evo",
        task=task,
        result=result,
        local_signal=final_judge,
    )
    if external_payload.get("lesson"):
        reflection_hint = str(external_payload.get("lesson"))
    result.debug.update(
        {
            "initial_judge": final_judge,
            "final_judge": final_judge,
            "reflection_triggered": not runtime.disable_reflection,
            "reflection_hint": reflection_hint,
            "task_type": classify_task(task),
            "constraints": extract_constraints(task),
            "skills": [skill.get("name") for skill in skills],
            "memory_hits": memory_hits,
            "external_assist": external_payload,
        }
    )
    if runtime.allow_evolution_updates and not runtime.benchmark_mode and not runtime.disable_memory:
        memory_item = memory.build_item(task, result, final_judge)
        if external_payload.get("lesson"):
            memory_item["strategy"] = str(external_payload.get("lesson"))
            memory_item["external_assist"] = external_payload
        memory.update(memory_item)
    return result
