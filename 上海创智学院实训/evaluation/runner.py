"""
批量评测运行器
管理Agent在数据集上的批量执行、结果保存和断点续跑
"""

import json
import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from datetime import datetime

from loguru import logger
from tqdm import tqdm

from config.settings import settings
from core.agent import ReActAgent, AgentResult
from evaluation.datasets import DatasetLoader, QAItem
from evaluation.metrics import MetricsCalculator, AggregateMetrics
from modules.evaluator import Evaluator


@dataclass
class RunResult:
    """单条运行结果"""
    item_id: str
    question: str
    expected: str
    predicted: str
    is_correct: bool
    iterations: int
    tool_calls: int
    duration: float
    finish_reason: str
    trajectory_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunReport:
    """评测运行报告"""
    run_id: str
    dataset: str
    mode: str  # "baseline" / "with_reflection" / "with_memory"
    start_time: str
    end_time: str = ""
    total_items: int = 0
    completed_items: int = 0
    metrics: Optional[AggregateMetrics] = None
    results: list[RunResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "mode": self.mode,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
        }
        if self.metrics:
            d["metrics"] = self.metrics.to_dict()
        return d


class EvaluationRunner:
    """
    评测运行器

    功能：
    1. 批量运行Agent
    2. 保存每条结果
    3. 断点续跑
    4. 计算汇总指标
    5. 生成对比报告
    """

    def __init__(
        self,
        agent: ReActAgent,
        evaluator: Optional[Evaluator] = None,
        results_dir: Optional[str] = None,
    ):
        self.agent = agent
        self.evaluator = evaluator or Evaluator()
        self.results_dir = Path(results_dir or settings.eval.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_loader = DatasetLoader()

    async def run(
        self,
        dataset_name: str,
        mode: str = "baseline",
        sample_size: Optional[int] = None,
        memory_context_fn=None,
        reflection_fn=None,
        resume: bool = True,
    ) -> RunReport:
        """
        在指定数据集上运行评测

        Args:
            dataset_name: 数据集名称 ("simpleqa" / "2wiki")
            mode: 运行模式 ("baseline" / "with_reflection" / "with_memory")
            sample_size: 评测样本数
            memory_context_fn: 记忆检索函数 (question -> str)
            reflection_fn: 反思函数 (question, result -> ReflectionResult)
            resume: 是否断点续跑

        Returns:
            RunReport: 评测报告
        """
        # 加载数据集
        items = self._load_dataset(dataset_name, sample_size)
        if not items:
            logger.error(f"No data loaded for dataset: {dataset_name}")
            return RunReport(
                run_id=self._generate_run_id(),
                dataset=dataset_name,
                mode=mode,
                start_time=datetime.now().isoformat(),
            )

        # 初始化报告
        run_id = self._generate_run_id()
        report = RunReport(
            run_id=run_id,
            dataset=dataset_name,
            mode=mode,
            start_time=datetime.now().isoformat(),
            total_items=len(items),
        )

        # 断点续跑：加载已有结果
        completed_ids = set()
        if resume:
            existing_results = self._load_existing_results(dataset_name, mode)
            for r in existing_results:
                report.results.append(r)
                completed_ids.add(r.item_id)
            if completed_ids:
                logger.info(f"Resuming: {len(completed_ids)} items already completed")

        # 逐条运行
        pending_items = [item for item in items if item.id not in completed_ids]
        logger.info(
            f"Running evaluation: dataset={dataset_name}, mode={mode}, "
            f"pending={len(pending_items)}/{len(items)}"
        )

        for item in tqdm(pending_items, desc=f"Eval {dataset_name}/{mode}"):
            try:
                run_result = await self._run_single(
                    item=item,
                    memory_context_fn=memory_context_fn,
                    reflection_fn=reflection_fn,
                    mode=mode,
                )
                report.results.append(run_result)
                report.completed_items += 1

                # 每10条保存一次（断点续跑）
                if report.completed_items % 10 == 0:
                    self._save_results(report)

            except Exception as e:
                logger.error(f"Failed on item {item.id}: {e}")
                # 记录失败
                report.results.append(RunResult(
                    item_id=item.id,
                    question=item.question,
                    expected=item.answer,
                    predicted="",
                    is_correct=False,
                    iterations=0,
                    tool_calls=0,
                    duration=0.0,
                    finish_reason="error",
                ))

        # 计算最终指标
        report.end_time = datetime.now().isoformat()
        report.metrics = MetricsCalculator.compute_aggregate([
            {
                "predicted": r.predicted,
                "expected": r.expected,
                "iterations": r.iterations,
                "tool_calls": r.tool_calls,
                "duration": r.duration,
            }
            for r in report.results
        ])

        # 保存最终结果
        self._save_results(report)
        self._save_report(report)

        logger.info(
            f"Evaluation complete: {dataset_name}/{mode} | "
            f"Accuracy: {report.metrics.accuracy:.2%} | "
            f"F1: {report.metrics.avg_f1:.4f}"
        )

        return report

    async def _run_single(
        self,
        item: QAItem,
        memory_context_fn=None,
        reflection_fn=None,
        mode: str = "baseline",
    ) -> RunResult:
        """运行单条评测"""
        # 获取记忆上下文
        memory_context = ""
        if memory_context_fn and mode in ("with_memory", "evolved"):
            memory_context = memory_context_fn(item.question)

        # 执行Agent
        result: AgentResult = await self.agent.solve(
            question=item.question,
            memory_context=memory_context,
        )

        # 评估
        eval_result = self.evaluator.evaluate(
            question=item.question,
            predicted=result.answer,
            expected=item.answer,
            use_llm=False,  # 批量评测时不用LLM判断，节省开销
        )

        # 如果失败且有反思功能，执行反思
        if not eval_result.is_correct and reflection_fn and mode != "baseline":
            reflection_fn(
                question=item.question,
                trajectory_text=result.trajectory_text,
                expected_answer=item.answer,
                agent_answer=result.answer,
            )

        return RunResult(
            item_id=item.id,
            question=item.question,
            expected=item.answer,
            predicted=result.answer,
            is_correct=eval_result.is_correct,
            iterations=result.total_iterations,
            tool_calls=result.tool_calls_count,
            duration=result.duration,
            finish_reason=result.finish_reason,
            trajectory_summary=result.trajectory_text[:500],
        )

    def _load_dataset(self, name: str, sample_size: Optional[int]) -> list[QAItem]:
        """加载数据集"""
        size = sample_size or settings.eval.eval_sample_size

        if name == "simpleqa":
            return self.dataset_loader.load_simpleqa(sample_size=size)
        elif name == "2wiki":
            return self.dataset_loader.load_2wiki(sample_size=size)
        else:
            logger.error(f"Unknown dataset: {name}")
            return []

    def _load_existing_results(self, dataset: str, mode: str) -> list[RunResult]:
        """加载已有结果用于断点续跑"""
        results_file = self.results_dir / f"{dataset}_{mode}_results.jsonl"
        results = []

        if results_file.exists():
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            results.append(RunResult(**data))
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")

        return results

    def _save_results(self, report: RunReport):
        """保存结果（JSONL格式，支持追加）"""
        results_file = self.results_dir / f"{report.dataset}_{report.mode}_results.jsonl"
        with open(results_file, "w", encoding="utf-8") as f:
            for r in report.results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    def _save_report(self, report: RunReport):
        """保存运行报告"""
        report_file = self.results_dir / f"{report.dataset}_{report.mode}_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    def _generate_run_id(self) -> str:
        """生成运行ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_sync(self, *args, **kwargs) -> RunReport:
        """同步版本"""
        return asyncio.run(self.run(*args, **kwargs))
