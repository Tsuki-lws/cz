from __future__ import annotations


def refine_hint(judge: dict) -> str:
    failure = judge.get("failure_type", "")
    if failure == "no_answer":
        return "上一轮没有得到答案。请重新规划，必要时先搜索证据；最终只给出简短答案。"
    if failure == "tool_call_leak":
        return "上一轮把工具调用文本当成了最终答案。请不要输出 <tool_call>；如需查证就重新调用真实工具，最后只输出简短答案，不要使用 <answer> 标签。"
    if failure == "uncertain_answer":
        return "上一轮表示无法确定。请换一组更具体的关键词，优先围绕图像实体、题目目标属性和候选来源继续查证；不要用 Unknown 作为最终答案。"
    if failure == "image_evidence_missing":
        return "上一轮图搜没有得到可用证据。请不要直接相信猜测实体；改用图中可见文字/地标/人物特征生成多个 search_text 查询，必要时打开候选网页验证。"
    if failure == "insufficient_evidence":
        return "上一轮证据不足。请至少调用一次 search_text/search_image 或 browser 工具验证答案。"
    if failure == "overbroad_answer":
        return "上一轮答案过长或没有锁定问题所求。请重新定位题目只问的实体/年份/地点/类别，最终答案只保留最短答案。"
    return "请检查是否有证据缺口，避免重复工具调用，给出简短最终答案。"
