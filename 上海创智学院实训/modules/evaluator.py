"""
评估器模块
判断Agent输出是否正确回答了问题
支持：精确匹配、模糊匹配、LLM判断
"""

import re
import string
from typing import Optional
from dataclasses import dataclass

from loguru import logger

from core.llm import LLMClient
from config.prompts import Prompts


@dataclass
class EvalResult:
    """评估结果"""
    is_correct: bool
    confidence: float  # 0-1
    method: str  # 使用的评估方法
    detail: str = ""  # 评估细节

    @property
    def score(self) -> float:
        return 1.0 if self.is_correct else 0.0


class Evaluator:
    """
    评估器

    多层评估策略（按优先级）：
    1. 精确匹配（标准化后）
    2. 包含匹配（答案包含在输出中）
    3. F1 token匹配
    4. LLM辅助判断（当前三种不确定时）
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    def evaluate(
        self,
        question: str,
        predicted: str,
        expected: str,
        use_llm: bool = True,
    ) -> EvalResult:
        """
        评估预测答案是否正确

        Args:
            question: 原始问题
            predicted: Agent的预测答案
            expected: 期望的正确答案
            use_llm: 是否使用LLM辅助判断

        Returns:
            EvalResult: 评估结果
        """
        if not predicted or not expected:
            return EvalResult(
                is_correct=False,
                confidence=1.0,
                method="empty_check",
                detail="Predicted or expected answer is empty",
            )

        # 标准化
        pred_norm = self._normalize(predicted)
        exp_norm = self._normalize(expected)

        # 1. 精确匹配
        if pred_norm == exp_norm:
            return EvalResult(
                is_correct=True,
                confidence=1.0,
                method="exact_match",
            )

        # 2. 包含匹配（期望答案包含在预测中）
        if exp_norm in pred_norm:
            return EvalResult(
                is_correct=True,
                confidence=0.9,
                method="contains_match",
                detail=f"Expected '{expected}' found in predicted answer",
            )

        # 3. F1 Token匹配
        f1 = self._compute_f1(pred_norm, exp_norm)
        if f1 >= 0.8:
            return EvalResult(
                is_correct=True,
                confidence=f1,
                method="f1_match",
                detail=f"F1 score: {f1:.3f}",
            )

        # 4. LLM辅助判断
        if use_llm and self.llm and f1 >= 0.3:
            llm_result = self._llm_evaluate(question, predicted, expected)
            if llm_result is not None:
                return llm_result

        # 默认：不正确
        return EvalResult(
            is_correct=False,
            confidence=1.0 - f1,
            method="no_match",
            detail=f"F1: {f1:.3f}, predicted: '{predicted[:100]}', expected: '{expected}'",
        )

    def _normalize(self, text: str) -> str:
        """标准化文本用于比较"""
        text = text.lower().strip()
        # 移除标点
        text = text.translate(str.maketrans("", "", string.punctuation))
        # 移除多余空格
        text = re.sub(r"\s+", " ", text)
        # 移除冠词
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _compute_f1(self, predicted: str, expected: str) -> float:
        """计算token级别的F1分数"""
        pred_tokens = set(predicted.split())
        exp_tokens = set(expected.split())

        if not pred_tokens or not exp_tokens:
            return 0.0

        common = pred_tokens & exp_tokens
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(exp_tokens)

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _llm_evaluate(
        self, question: str, predicted: str, expected: str
    ) -> Optional[EvalResult]:
        """使用LLM辅助判断"""
        try:
            prompt = Prompts.EVALUATOR_PROMPT.format(
                question=question,
                expected_answer=expected,
                agent_answer=predicted,
            )

            response = self.llm.simple_generate(prompt)
            response_clean = response.strip().upper()

            if "CORRECT" in response_clean and "INCORRECT" not in response_clean:
                return EvalResult(
                    is_correct=True,
                    confidence=0.85,
                    method="llm_judge",
                    detail="LLM judged as correct",
                )
            elif "INCORRECT" in response_clean:
                return EvalResult(
                    is_correct=False,
                    confidence=0.85,
                    method="llm_judge",
                    detail="LLM judged as incorrect",
                )

        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")

        return None

    def batch_evaluate(
        self,
        results: list[dict],
        use_llm: bool = False,
    ) -> dict:
        """
        批量评估

        Args:
            results: [{"question": ..., "predicted": ..., "expected": ...}, ...]
            use_llm: 是否使用LLM辅助判断

        Returns:
            dict: 包含成功率、F1等汇总指标
        """
        total = len(results)
        correct = 0
        f1_scores = []

        for item in results:
            eval_result = self.evaluate(
                question=item["question"],
                predicted=item["predicted"],
                expected=item["expected"],
                use_llm=use_llm,
            )
            if eval_result.is_correct:
                correct += 1

            # 计算F1
            pred_norm = self._normalize(item["predicted"])
            exp_norm = self._normalize(item["expected"])
            f1_scores.append(self._compute_f1(pred_norm, exp_norm))

        accuracy = correct / total if total > 0 else 0.0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "avg_f1": round(avg_f1, 4),
        }
