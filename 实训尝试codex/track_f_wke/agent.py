from __future__ import annotations

from dataclasses import replace
from typing import Any

from shared_sii_adapter.react_runner import DEFAULT_SYSTEM_PROMPT, run_react_task
from shared_sii_adapter.types import AgentRunResult, RuntimeConfig

from .knowledge_store import KnowledgeStore, summarize_hits


WKE_PROMPT = DEFAULT_SYSTEM_PROMPT + """

World Knowledge Exploration 策略：
- 在训练/dev 阶段，可利用公开知识探索得到的经验辅助解题。
- 经验只能来自公开数据和历史开发轨迹，不能来自 benchmark。
- benchmark 模式下知识库固定，只读不写，且不做跨样本进化。
- 回答时仍需基于当前样本证据，不要把知识库当作答案库。
"""


def run_one(task: dict[str, Any], runtime: RuntimeConfig) -> AgentRunResult:
    store = KnowledgeStore()
    query = str(task.get("instruction") or task.get("question") or "")
    hits = [] if runtime.benchmark_mode else store.retrieve(query)
    memory_context = summarize_hits(hits)
    result = run_react_task(
        task,
        replace(runtime, track_name="track_f"),
        system_prompt=WKE_PROMPT,
        memory_context=memory_context,
        track_name="track_f",
    )
    result.debug.update({"knowledge_hits": hits, "benchmark_read_only": runtime.benchmark_mode})
    if runtime.allow_evolution_updates and not runtime.benchmark_mode:
        store.add(
            {
                "topic": query[:200],
                "lesson": f"pred={result.pred[:200]} tool_calls={result.metrics.get('tool_calls', 0)}",
                "source": "track_f_run",
            }
        )
    return result

