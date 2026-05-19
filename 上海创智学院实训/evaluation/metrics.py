"""
评估指标计算
支持：Exact Match、F1 Score、成功率、效率指标
"""

import re
import string
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class SingleMetrics:
    """单条数据的评测指标"""
    exact_match: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


@dataclass
class AggregateMetrics:
    """聚合评测指标"""
    total_count: int = 0
    correct_count: int = 0
    accuracy: float = 0.0  # EM准确率
    avg_f1: float = 0.0
    avg_iterations: float = 0.0
    avg_tool_calls: float = 0.0
    avg_duration: float = 0.0
    invalid_tool_call_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_count": self.total_count,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 4),
            "avg_f1": round(self.avg_f1, 4),
            "avg_iterations": round(self.avg_iterations, 2),
            "avg_tool_calls": round(self.avg_tool_calls, 2),
            "avg_duration": round(self.avg_duration, 2),
            "invalid_tool_call_rate": round(self.invalid_tool_call_rate, 4),
        }


class MetricsCalculator:
    """
    评估指标计算器

    支持的指标：
    - Exact Match (EM): 标准化后精确匹配
    - F1 Score: token级别的精确率/召回率调和平均
    - 成功率: EM为1的比例
    - 效率指标: 平均轮次、工具调用数、耗时
    """

    @staticmethod
    def compute_single(predicted: str, expected: str) -> SingleMetrics:
        """
        计算单条数据的指标

        Args:
            predicted: 预测答案
            expected: 期望答案

        Returns:
            SingleMetrics: 各项指标
        """
        pred_norm = MetricsCalculator._normalize(predicted)
        exp_norm = MetricsCalculator._normalize(expected)

        # Exact Match
        em = 1.0 if pred_norm == exp_norm else 0.0

        # F1
        pred_tokens = pred_norm.split()
        exp_tokens = exp_norm.split()

        if not pred_tokens or not exp_tokens:
            return SingleMetrics(exact_match=em, f1_score=em)

        common = set(pred_tokens) & set(exp_tokens)
        num_common = sum(min(pred_tokens.count(w), exp_tokens.count(w)) for w in common)

        if num_common == 0:
            return SingleMetrics(exact_match=em)

        precision = num_common / len(pred_tokens)
        recall = num_common / len(exp_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        return SingleMetrics(
            exact_match=em,
            f1_score=f1,
            precision=precision,
            recall=recall,
        )

    @staticmethod
    def compute_aggregate(
        results: list[dict],
    ) -> AggregateMetrics:
        """
        计算聚合指标

        Args:
            results: 评测结果列表，每项包含:
                - predicted: str
                - expected: str
                - iterations: int (optional)
                - tool_calls: int (optional)
                - duration: float (optional)
                - invalid_calls: int (optional)

        Returns:
            AggregateMetrics: 聚合指标
        """
        if not results:
            return AggregateMetrics()

        total = len(results)
        correct = 0
        f1_sum = 0.0
        iter_sum = 0
        tool_sum = 0
        duration_sum = 0.0
        invalid_sum = 0
        total_calls = 0

        for item in results:
            metrics = MetricsCalculator.compute_single(
                item.get("predicted", ""),
                item.get("expected", ""),
            )

            if metrics.exact_match > 0:
                correct += 1
            f1_sum += metrics.f1_score

            # 效率指标
            iter_sum += item.get("iterations", 0)
            calls = item.get("tool_calls", 0)
            tool_sum += calls
            total_calls += calls
            duration_sum += item.get("duration", 0.0)
            invalid_sum += item.get("invalid_calls", 0)

        return AggregateMetrics(
            total_count=total,
            correct_count=correct,
            accuracy=correct / total,
            avg_f1=f1_sum / total,
            avg_iterations=iter_sum / total,
            avg_tool_calls=tool_sum / total,
            avg_duration=duration_sum / total,
            invalid_tool_call_rate=(
                invalid_sum / total_calls if total_calls > 0 else 0.0
            ),
        )

    @staticmethod
    def compare(baseline: AggregateMetrics, evolved: AggregateMetrics) -> dict:
        """
        对比基线和进化后的指标

        Returns:
            dict: 各项指标的变化
        """
        return {
            "accuracy_delta": round(evolved.accuracy - baseline.accuracy, 4),
            "accuracy_improvement": (
                f"{(evolved.accuracy - baseline.accuracy) / max(baseline.accuracy, 0.001) * 100:.1f}%"
            ),
            "f1_delta": round(evolved.avg_f1 - baseline.avg_f1, 4),
            "iterations_saved": round(
                baseline.avg_iterations - evolved.avg_iterations, 2
            ),
            "tool_calls_saved": round(
                baseline.avg_tool_calls - evolved.avg_tool_calls, 2
            ),
            "time_saved": round(baseline.avg_duration - evolved.avg_duration, 2),
        }

    @staticmethod
    def _normalize(text: str) -> str:
        """标准化答案文本"""
        if not text:
            return ""
        # 转小写
        text = text.lower()
        # 移除标点
        text = text.translate(str.maketrans("", "", string.punctuation))
        # 移除冠词
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # 合并空白
        text = " ".join(text.split())
        return text.strip()
