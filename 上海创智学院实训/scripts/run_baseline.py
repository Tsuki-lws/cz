"""
基线测试脚本
运行纯ReAct Agent（无反思无记忆）在SimpleQA和2Wiki上的测试
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from core.llm import LLMClient
from core.agent import ReActAgent
from core.harness import Harness
from tools.registry import ToolRegistry
from tools.search import SearchTool
from tools.browser import BrowserNavigateTool
from modules.evaluator import Evaluator
from evaluation.runner import EvaluationRunner


async def run_baseline_test(dataset: str = "2wiki", sample_size: int = 50):
    """
    运行基线测试

    这是最基本的ReAct Agent，没有反思和记忆模块
    用于建立性能基线，后续对比进化效果
    """
    print("=" * 60)
    print(f"  BASELINE TEST - {dataset.upper()}")
    print(f"  Model: {settings.llm.model_name}")
    print(f"  Max Iterations: {settings.agent.max_iterations}")
    print(f"  Sample Size: {sample_size}")
    print("=" * 60)

    # 创建组件
    llm = LLMClient()
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool())
    tool_registry.register(BrowserNavigateTool())

    harness = Harness()
    agent = ReActAgent(llm=llm, tool_registry=tool_registry, harness=harness)
    evaluator = Evaluator()

    runner = EvaluationRunner(agent=agent, evaluator=evaluator)

    # 运行评测
    report = await runner.run(
        dataset_name=dataset,
        mode="baseline",
        sample_size=sample_size,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    if report.metrics:
        print(f"  Total:     {report.metrics.total_count}")
        print(f"  Correct:   {report.metrics.correct_count}")
        print(f"  Accuracy:  {report.metrics.accuracy:.2%}")
        print(f"  Avg F1:    {report.metrics.avg_f1:.4f}")
        print(f"  Avg Iter:  {report.metrics.avg_iterations:.1f}")
        print(f"  Avg Calls: {report.metrics.avg_tool_calls:.1f}")
        print(f"  Avg Time:  {report.metrics.avg_duration:.1f}s")
    print("=" * 60)

    # LLM统计
    print(f"\n  LLM Stats: {llm.stats}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--dataset", "-d", default="2wiki", choices=["simpleqa", "2wiki"])
    parser.add_argument("--sample", "-n", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(run_baseline_test(args.dataset, args.sample))
