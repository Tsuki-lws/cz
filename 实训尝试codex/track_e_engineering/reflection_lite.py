from __future__ import annotations

from typing import Any


def build_reflection_hint(task: dict[str, Any]) -> str:
    text = str(task.get("instruction") or task.get("question") or "")
    hints = [
        "如果缺少证据，先搜索再作答。",
        "不要重复完全相同的 query 或 URL。",
        "最终答案用 <answer>...</answer> 包裹。",
    ]
    if task.get("image") or task.get("image_url") or task.get("image_b64"):
        hints.append("图像题优先使用 search_image 验证图中实体，再搜索目标事实。")
    if len(text) > 120:
        hints.append("长问题先拆成实体、关系、目标答案三部分。")
    return "\n".join(f"- {hint}" for hint in hints)

