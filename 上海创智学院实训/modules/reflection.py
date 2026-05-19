"""
反思模块
在任务失败后进行结构化反思，提取失败原因和修正策略
支持：即时反思（重试当前任务）和跨任务反思（提取通用规则）
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum

from loguru import logger

from core.llm import LLMClient
from config.prompts import Prompts


class FailureCategory(Enum):
    """失败原因分类"""
    SEARCH_STRATEGY = "search_strategy"  # 搜索策略不当
    REASONING_ERROR = "reasoning_error"  # 推理逻辑错误
    INFORMATION_MISSING = "information_missing"  # 信息获取不足
    TOOL_MISUSE = "tool_misuse"  # 工具使用不当
    LOOP_TRAP = "loop_trap"  # 陷入循环
    UNKNOWN = "unknown"


@dataclass
class ReflectionResult:
    """反思结果"""
    task_id: str
    question: str
    failure_point: str  # 哪一步出了问题
    root_cause: FailureCategory  # 失败根因分类
    correction: str  # 修正策略
    general_rule: str  # 可迁移的通用规则
    confidence: float = 0.8
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "question": self.question[:100],
            "failure_point": self.failure_point,
            "root_cause": self.root_cause.value,
            "correction": self.correction,
            "general_rule": self.general_rule,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    def to_memory_text(self) -> str:
        """转为供记忆模块存储的文本"""
        return (
            f"Rule: {self.general_rule}\n"
            f"Context: When {self.failure_point}\n"
            f"Strategy: {self.correction}"
        )


class ReflectionModule:
    """
    反思模块

    核心能力：
    1. 即时反思 (reflect_on_failure): 分析当前任务的失败并生成修正策略
    2. 构建重试提示 (build_retry_prompt): 用反思结果指导重试
    3. 跨任务规则提取 (extract_general_rules): 从多次经验中提取通用规则
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()
        self._reflection_history: list[ReflectionResult] = []

    def reflect_on_failure(
        self,
        question: str,
        trajectory_text: str,
        expected_answer: str = "",
        agent_answer: str = "",
    ) -> ReflectionResult:
        """
        对失败的任务进行反思

        Args:
            question: 原始问题
            trajectory_text: Agent的推理轨迹文本
            expected_answer: 期望的正确答案（如果有）
            agent_answer: Agent给出的错误答案

        Returns:
            ReflectionResult: 结构化的反思结果
        """
        logger.info(f"Reflecting on failed task: {question[:50]}...")

        # 构建反思prompt
        prompt = Prompts.REFLECTION_PROMPT.format(
            question=question,
            expected_answer=expected_answer or "[Not provided]",
            agent_answer=agent_answer or "[No answer produced]",
            trajectory=trajectory_text[:3000],  # 截断避免超长
        )

        # 调用LLM进行反思
        try:
            response = self.llm.simple_generate(prompt)
            reflection = self._parse_reflection(response, question)
        except Exception as e:
            logger.error(f"Reflection LLM call failed: {e}")
            reflection = ReflectionResult(
                task_id=self._generate_task_id(question),
                question=question,
                failure_point="Unable to analyze (reflection failed)",
                root_cause=FailureCategory.UNKNOWN,
                correction="Try a different approach",
                general_rule="When stuck, try breaking the problem into smaller parts",
            )

        # 记录到历史
        self._reflection_history.append(reflection)
        logger.info(
            f"Reflection complete | cause={reflection.root_cause.value} | "
            f"rule: {reflection.general_rule[:80]}"
        )

        return reflection

    def build_retry_prompt(self, question: str, reflection: ReflectionResult) -> str:
        """
        基于反思结果构建重试提示词

        Args:
            question: 原始问题
            reflection: 反思结果

        Returns:
            str: 带有反思指导的重试提示词
        """
        return Prompts.REFLECTION_RETRY_PROMPT.format(
            failure_point=reflection.failure_point,
            root_cause=reflection.root_cause.value,
            correction=reflection.correction,
            question=question,
        )

    def extract_general_rules(
        self, reflections: list[ReflectionResult], min_confidence: float = 0.5
    ) -> list[str]:
        """
        从多次反思中提取通用规则

        Args:
            reflections: 反思结果列表
            min_confidence: 最低置信度阈值

        Returns:
            list[str]: 提取的通用规则列表
        """
        filtered = [r for r in reflections if r.confidence >= min_confidence]
        rules = [r.general_rule for r in filtered if r.general_rule]

        # 去重：使用简单的相似度判断
        unique_rules = []
        for rule in rules:
            is_duplicate = False
            for existing in unique_rules:
                if self._is_similar_rule(rule, existing):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_rules.append(rule)

        return unique_rules

    def _parse_reflection(self, response: str, question: str) -> ReflectionResult:
        """解析LLM反思输出"""
        failure_point = ""
        root_cause = FailureCategory.UNKNOWN
        correction = ""
        general_rule = ""

        # 逐行解析结构化输出
        for line in response.split("\n"):
            line = line.strip()

            if line.startswith("FAILURE_POINT:"):
                failure_point = line[len("FAILURE_POINT:"):].strip()
            elif line.startswith("ROOT_CAUSE:"):
                cause_str = line[len("ROOT_CAUSE:"):].strip().upper()
                root_cause = self._parse_cause(cause_str)
            elif line.startswith("CORRECTION:"):
                correction = line[len("CORRECTION:"):].strip()
            elif line.startswith("GENERAL_RULE:"):
                general_rule = line[len("GENERAL_RULE:"):].strip()

        # 如果结构化解析失败，尝试从自由文本中提取
        if not failure_point and not correction:
            failure_point, correction, general_rule = self._extract_from_freetext(response)

        return ReflectionResult(
            task_id=self._generate_task_id(question),
            question=question,
            failure_point=failure_point or "Unable to identify specific failure point",
            root_cause=root_cause,
            correction=correction or "Try a different search strategy",
            general_rule=general_rule or "Verify each step before proceeding",
        )

    def _parse_cause(self, cause_str: str) -> FailureCategory:
        """解析失败原因分类"""
        cause_map = {
            "SEARCH_STRATEGY": FailureCategory.SEARCH_STRATEGY,
            "REASONING_ERROR": FailureCategory.REASONING_ERROR,
            "INFORMATION_MISSING": FailureCategory.INFORMATION_MISSING,
            "TOOL_MISUSE": FailureCategory.TOOL_MISUSE,
            "LOOP_TRAP": FailureCategory.LOOP_TRAP,
        }

        for key, value in cause_map.items():
            if key in cause_str:
                return value

        return FailureCategory.UNKNOWN

    def _extract_from_freetext(self, text: str) -> tuple[str, str, str]:
        """从自由文本中提取反思内容"""
        sentences = re.split(r"[.。!！?？\n]", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        failure_point = sentences[0] if sentences else ""
        correction = sentences[1] if len(sentences) > 1 else ""
        general_rule = sentences[-1] if len(sentences) > 2 else ""

        return failure_point, correction, general_rule

    def _is_similar_rule(self, rule1: str, rule2: str) -> bool:
        """简单的规则相似度判断"""
        # 基于词重叠
        words1 = set(rule1.lower().split())
        words2 = set(rule2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap > 0.6

    def _generate_task_id(self, question: str) -> str:
        """生成任务ID"""
        import hashlib
        return hashlib.md5(question.encode()).hexdigest()[:12]

    @property
    def history(self) -> list[ReflectionResult]:
        return self._reflection_history.copy()

    @property
    def failure_stats(self) -> dict:
        """失败原因统计"""
        stats = {}
        for r in self._reflection_history:
            cause = r.root_cause.value
            stats[cause] = stats.get(cause, 0) + 1
        return stats
