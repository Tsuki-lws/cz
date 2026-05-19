from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig

from .evolution.agent_debugger import classify_failure
from .evolution.manifest import write_manifest
from .evolution.world_knowledge import load_world_knowledge


def load_component_prompt() -> str:
    path = Path("track_c_ahe/harness_components/system_prompt.md")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_SYSTEM_PROMPT


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    prompt = load_component_prompt()
    world_knowledge = load_world_knowledge()
    result = run_react_task(
        task,
        replace(runtime, track_name="track_c"),
        system_prompt=prompt,
        memory_context=world_knowledge,
        track_name="track_c",
    )
    failure = classify_failure(result.pred, int(result.metrics.get("tool_calls", 0)))
    manifest = {
        "index": result.index,
        "failure_type": failure,
        "component_prompt": "track_c_ahe/harness_components/system_prompt.md",
        "suggested_change": "If repeated failures accumulate, update tool_policy or system_prompt on dev data only.",
    }
    if runtime.allow_evolution_updates and not runtime.benchmark_mode:
        write_manifest(manifest)
    result.debug.update({"ahe_manifest": manifest})
    return result
