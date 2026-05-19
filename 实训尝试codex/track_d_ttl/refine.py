from __future__ import annotations


def refine_hint(judge: dict) -> str:
    failure = judge.get("failure_type", "")
    if failure == "no_answer":
        return "上一轮没有得到答案。请重新规划，必要时先搜索证据，最后必须输出 <answer>...</answer>。"
    if failure == "insufficient_evidence":
        return "上一轮证据不足。请至少调用一次 search_text/search_image 或 browser 工具验证答案。"
    return "请检查是否有证据缺口，避免重复工具调用，给出简短最终答案。"

