"""
结果分析与对比
生成对比表格、可视化图表
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from loguru import logger

from config.settings import settings
from evaluation.metrics import MetricsCalculator, AggregateMetrics


@dataclass
class ComparisonReport:
    """对比报告"""
    baseline_metrics: AggregateMetrics
    evolved_metrics: AggregateMetrics
    improvements: dict

    def summary(self) -> str:
        """生成文本摘要"""
        lines = [
            "=" * 60,
            "         EVOLUTION COMPARISON REPORT",
            "=" * 60,
            "",
            f"{'Metric':<25} {'Baseline':<15} {'Evolved':<15} {'Delta':<15}",
            "-" * 60,
            f"{'Accuracy':<25} {self.baseline_metrics.accuracy:<15.4f} {self.evolved_metrics.accuracy:<15.4f} {self.improvements['accuracy_delta']:+.4f}",
            f"{'Avg F1':<25} {self.baseline_metrics.avg_f1:<15.4f} {self.evolved_metrics.avg_f1:<15.4f} {self.improvements['f1_delta']:+.4f}",
            f"{'Avg Iterations':<25} {self.baseline_metrics.avg_iterations:<15.2f} {self.evolved_metrics.avg_iterations:<15.2f} {self.improvements['iterations_saved']:+.2f}",
            f"{'Avg Tool Calls':<25} {self.baseline_metrics.avg_tool_calls:<15.2f} {self.evolved_metrics.avg_tool_calls:<15.2f} {self.improvements['tool_calls_saved']:+.2f}",
            f"{'Avg Duration (s)':<25} {self.baseline_metrics.avg_duration:<15.2f} {self.evolved_metrics.avg_duration:<15.2f} {self.improvements['time_saved']:+.2f}",
            "",
            "-" * 60,
            f"Accuracy Improvement: {self.improvements['accuracy_improvement']}",
            "=" * 60,
        ]
        return "\n".join(lines)


class ResultAnalyzer:
    """
    结果分析器

    功能：
    1. 加载不同模式的评测结果
    2. 生成对比报告
    3. 分析失败模式
    4. 生成可视化图表
    """

    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = Path(results_dir or settings.eval.results_dir)

    def compare(self, dataset: str, baseline_mode: str = "baseline", evolved_mode: str = "with_memory") -> ComparisonReport:
        """
        对比两种模式的结果

        Args:
            dataset: 数据集名称
            baseline_mode: 基线模式
            evolved_mode: 进化模式

        Returns:
            ComparisonReport: 对比报告
        """
        baseline_results = self._load_results(dataset, baseline_mode)
        evolved_results = self._load_results(dataset, evolved_mode)

        baseline_metrics = MetricsCalculator.compute_aggregate(baseline_results)
        evolved_metrics = MetricsCalculator.compute_aggregate(evolved_results)
        improvements = MetricsCalculator.compare(baseline_metrics, evolved_metrics)

        report = ComparisonReport(
            baseline_metrics=baseline_metrics,
            evolved_metrics=evolved_metrics,
            improvements=improvements,
        )

        # 打印报告
        print(report.summary())
        return report

    def analyze_failures(self, dataset: str, mode: str) -> dict:
        """
        分析失败案例

        Returns:
            dict: 失败原因统计
        """
        results_file = self.results_dir / f"{dataset}_{mode}_results.jsonl"
        if not results_file.exists():
            return {}

        failures = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if not data.get("is_correct", False):
                        failures.append(data)

        # 统计失败原因
        reason_stats = {}
        for fail in failures:
            reason = fail.get("finish_reason", "unknown")
            reason_stats[reason] = reason_stats.get(reason, 0) + 1

        # 统计问题类型失败率
        # (需要数据中包含question_type)

        return {
            "total_failures": len(failures),
            "failure_reasons": reason_stats,
            "sample_failures": failures[:5],  # 展示前5个失败案例
        }

    def generate_chart_data(self, dataset: str) -> dict:
        """
        生成用于可视化的数据

        Returns:
            dict: 各模式的指标数据
        """
        modes = ["baseline", "with_reflection", "with_memory"]
        chart_data = {"modes": [], "accuracy": [], "f1": [], "iterations": []}

        for mode in modes:
            results = self._load_results(dataset, mode)
            if results:
                metrics = MetricsCalculator.compute_aggregate(results)
                chart_data["modes"].append(mode)
                chart_data["accuracy"].append(metrics.accuracy)
                chart_data["f1"].append(metrics.avg_f1)
                chart_data["iterations"].append(metrics.avg_iterations)

        return chart_data

    def plot_comparison(self, dataset: str, save_path: Optional[str] = None):
        """
        绘制对比图表

        Args:
            dataset: 数据集名称
            save_path: 图表保存路径
        """
        try:
            import matplotlib.pyplot as plt

            data = self.generate_chart_data(dataset)
            if not data["modes"]:
                logger.warning("No data available for plotting")
                return

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Accuracy
            axes[0].bar(data["modes"], data["accuracy"], color=["#3498db", "#2ecc71", "#e74c3c"])
            axes[0].set_title("Accuracy")
            axes[0].set_ylim(0, 1)
            axes[0].set_ylabel("Score")

            # F1
            axes[1].bar(data["modes"], data["f1"], color=["#3498db", "#2ecc71", "#e74c3c"])
            axes[1].set_title("Average F1 Score")
            axes[1].set_ylim(0, 1)

            # Iterations
            axes[2].bar(data["modes"], data["iterations"], color=["#3498db", "#2ecc71", "#e74c3c"])
            axes[2].set_title("Average Iterations")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Chart saved to: {save_path}")
            else:
                plt.savefig(
                    self.results_dir / f"{dataset}_comparison.png",
                    dpi=150, bbox_inches="tight",
                )

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, skipping chart generation")

    def _load_results(self, dataset: str, mode: str) -> list[dict]:
        """加载评测结果"""
        results_file = self.results_dir / f"{dataset}_{mode}_results.jsonl"
        results = []

        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

        return results
