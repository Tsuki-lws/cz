from __future__ import annotations

from typing import Any

FORBIDDEN_RUNTIME_KEYS = {"gold", "ground_truth", "label"}


def detect_forbidden_keys(obj: Any, forbidden: set[str] | None = None, path: str = "") -> list[str]:
    forbidden = forbidden or FORBIDDEN_RUNTIME_KEYS
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            child_path = f"{path}.{key}" if path else str(key)
            if key_l in forbidden:
                hits.append(child_path)
            hits.extend(detect_forbidden_keys(value, forbidden, child_path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(detect_forbidden_keys(value, forbidden, f"{path}[{idx}]"))
    return hits


def assert_no_gold_payload(payload: Any, *, allow_answer: bool = False) -> None:
    forbidden = set(FORBIDDEN_RUNTIME_KEYS)
    if not allow_answer:
        forbidden.add("answer")
    hits = detect_forbidden_keys(payload, forbidden)
    if hits:
        raise ValueError(f"forbidden runtime fields detected: {hits}")


def assert_no_benchmark_evolution(run_mode: str, allow_evolution_updates: bool) -> None:
    if run_mode == "benchmark" and allow_evolution_updates:
        raise ValueError("benchmark mode must disable cross-sample evolution updates")


def assert_harness_base_model(model_name: str) -> None:
    lowered = (model_name or "").lower()
    if "qwen3-32b" in lowered:
        raise ValueError("Qwen3-32B is allowed only as judge/reflection/teacher, not as Harness base model")
    if any(marker in lowered for marker in ["72b", "70b", "65b", "34b", "110b", "100b"]):
        raise ValueError(f"Harness base model exceeds project policy: {model_name}")


def assert_teacher_model_size_le_32b(model_name: str) -> None:
    lowered = (model_name or "").lower()
    markers = ["72b", "70b", "65b", "34b", "110b", "100b"]
    if any(marker in lowered for marker in markers):
        raise ValueError(f"distillation teacher/model exceeds <=32B policy: {model_name}")
