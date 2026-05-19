"""
进化测试脚本
运行带反思+记忆的完整进化Agent
"""

import asyncio
import sys
from pathlib import Path

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
from modules.reflection import ReflectionModule
from modules.memory import MemorySystem
from evaluation.runner import EvaluationRunner


async def run_evolved_test(dataset: str = "2wiki", sample_size: int = 50):
    """
    运行完整进化测试

    包含：反思模块 + 记忆模块
    Agent在每次失败后会反思并存储经验
    后续任务会检索相关经验来指导推理
    """
    print("=" * 60)
    print(f"  EVOLVED AGENT TEST - {dataset.upper()}")
    print(f"  Model: {settings.llm.model_name}")
    print(f"  Features: Reflection + Memory")
    print(f"  Sample Size: {sample_size}")
    print("=" * 60)

    # 创建组件
    llm = LLMClient()
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool())
    tool_registry.register(BrowserNavigateTool())

    harness = Harness()
    agent = ReActAgent(llm=llm, tool_registry=tool_registry, harness=harness)
    evaluator = Evaluator(llm=llm)
    reflection = ReflectionModule(llm=llm)
    memory = MemorySystem(llm=llm)

    runner = EvaluationRunner(agent=agent, evaluator=evaluator)

    # 记忆检索函数
    def memory_retrieve(question: str) -> str:
        return memory.retrieve(question)

    # 反思+存储函数
    def reflect_and_store(question, trajectory_text, expected_answer, agent_answer):
        result = reflection.reflect_on_failure(
            question=question,
            trajectory_text=trajectory_text,
            expected_answer=expected_answer,
            agent_answer=agent_answer,
        )
        # 存储经验
        memory.store_experience(
            question=question,
            outcome="failure",
            strategy=result.correction,
            insight=result.general_rule,
            question_type="multi-hop" if "2wiki" in dataset else "factual",
        )
        # 存储规则
        memory.store_rule(
            content=result.general_rule,
            applies_to="general",
            source_task_id=result.task_id,
        )
        logger.info(f"Stored reflection: {result.general_rule[:60]}")

    # 运行评测
    report = await runner.run(
        dataset_name=dataset,
        mode="with_memory",
        sample_size=sample_size,
        memory_context_fn=memory_retrieve,
        reflection_fn=reflect_and_store,
    )

    # 记忆整合
    memory.consolidate()

    # 打印结果
    print("\n" + "=" * 60)
    print("  EVOLVED RESULTS")
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

    # 记忆统计
    print(f"\n  Memory Stats: {memory.stats}")
    print(f"  Reflection Stats: {reflection.failure_stats}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evolved agent evaluation")
    parser.add_argument("--dataset", "-d", default="2wiki", choices=["simpleqa", "2wiki"])
    parser.add_argument("--sample", "-n", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(run_evolved_test(args.dataset, args.sample))
