from __future__ import annotations

from typing import Any


def build_plan(task: dict[str, Any], manager_state: dict[str, Any]) -> str:
    task_type = manager_state.get("task_type", "simple_fact")
    if task_type == "image_to_fact":
        return (
            "1. 使用 search_image 或图像线索确认图中实体。\n"
            "2. 使用 search_text 验证目标事实。\n"
            "3. 必要时 browser_parallel 打开候选来源。\n"
            "4. 输出短答案。"
        )
    if task_type == "multi_hop":
        return (
            "1. 拆分问题中的实体和关系。\n"
            "2. 对第一跳实体/关系搜索证据。\n"
            "3. 对第二跳目标事实搜索或浏览验证。\n"
            "4. 合并证据后输出短答案。"
        )
    return "1. 搜索关键实体和目标属性。\n2. 验证来源。\n3. 输出短答案。"

