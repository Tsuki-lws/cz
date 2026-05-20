from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeConfig:
    llm_base_url: str
    model_name: str = "Qwen3.5-9B"
    judge_base_url: str | None = None
    judge_model_name: str = "Qwen3-32B"
    max_steps: int = 20
    max_tokens: int = 16000
    temperature: float = 1.0
    enable_thinking: bool = True
    disable_tools: bool = False
    disable_reflection: bool = False
    disable_memory: bool = False
    enable_xml_tool_fallback: bool = True
    structured_final_answer: bool = True
    output_dir: str = "experiments"
    track_name: str = "baseline"
    group_id: str = "0"
    run_mode: str = "eval"  # train | dev | eval | benchmark
    allow_evolution_updates: bool = True
    enable_external_assist: bool = False
    external_assist_max_tokens: int = 512

    @property
    def effective_judge_base_url(self) -> str:
        return self.judge_base_url or self.llm_base_url

    @property
    def benchmark_mode(self) -> bool:
        return self.run_mode == "benchmark"


@dataclass(slots=True)
class AgentRunResult:
    index: str
    instruction: str
    image: str = ""
    image_url: str = ""
    pred: str = ""
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_result_row(self, include_answer: bool = False, answer: str = "") -> dict[str, Any]:
        row = {
            "index": self.index,
            "instruction": self.instruction,
            "image": self.image,
            "pred": self.pred,
        }
        if include_answer:
            row["answer"] = answer
        return row
