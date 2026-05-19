"""
自进化任务求解智能体 - 统一入口

支持的运行模式：
1. baseline - 纯ReAct Agent基线测试
2. reflection - 加入反思模块
3. memory - 加入反思+记忆模块（完整进化系统）
4. compare - 对比分析各模式结果
5. browsecomp - BrowseComp-Plus打榜
"""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from config.settings import settings
from core.llm import LLMClient
from core.agent import ReActAgent
from core.harness import Harness
from tools.registry import ToolRegistry
from tools.search import SearchTool, ImageSearchTool
from tools.browser import BrowserTool, BrowserNavigateTool
from modules.evaluator import Evaluator
from modules.reflection import ReflectionModule
from modules.memory import MemorySystem
from evaluation.runner import EvaluationRunner
from evaluation.analysis import ResultAnalyzer


def setup_logging():
    """配置日志"""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    logger.add(
        log_dir / "agent_{time}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )


def create_agent(mode: str = "baseline") -> tuple:
    """
    创建Agent及相关组件

    Args:
        mode: 运行模式

    Returns:
        tuple: (agent, memory_system, reflection_module)
    """
    # 1. LLM客户端
    llm = LLMClient()

    # 2. 注册工具
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool())
    tool_registry.register(BrowserNavigateTool())
    tool_registry.register(BrowserTool())

    # 3. Harness
    harness = Harness()

    # 4. Agent
    agent = ReActAgent(llm=llm, tool_registry=tool_registry, harness=harness)

    # 5. 反思和记忆（根据模式决定是否启用）
    memory = None
    reflection = None

    if mode in ("reflection", "memory", "evolved"):
        reflection = ReflectionModule(llm=llm)

    if mode in ("memory", "evolved"):
        memory = MemorySystem(llm=llm)

    return agent, memory, reflection


async def run_baseline(dataset: str, sample_size: int):
    """运行基线测试"""
    logger.info(f"=== BASELINE TEST: {dataset} (n={sample_size}) ===")

    agent, _, _ = create_agent("baseline")
    evaluator = Evaluator()
    runner = EvaluationRunner(agent=agent, evaluator=evaluator)

    report = await runner.run(
        dataset_name=dataset,
        mode="baseline",
        sample_size=sample_size,
    )

    print(f"\n{'='*50}")
    print(f"BASELINE RESULTS - {dataset}")
    print(f"{'='*50}")
    print(f"Accuracy: {report.metrics.accuracy:.2%}")
    print(f"Avg F1:   {report.metrics.avg_f1:.4f}")
    print(f"Avg Iter: {report.metrics.avg_iterations:.1f}")
    print(f"Avg Time: {report.metrics.avg_duration:.1f}s")

    return report


async def run_with_reflection(dataset: str, sample_size: int):
    """运行带反思的测试"""
    logger.info(f"=== REFLECTION TEST: {dataset} (n={sample_size}) ===")

    agent, _, reflection = create_agent("reflection")
    evaluator = Evaluator()
    runner = EvaluationRunner(agent=agent, evaluator=evaluator)

    report = await runner.run(
        dataset_name=dataset,
        mode="with_reflection",
        sample_size=sample_size,
        reflection_fn=reflection.reflect_on_failure if reflection else None,
    )

    print(f"\n{'='*50}")
    print(f"REFLECTION RESULTS - {dataset}")
    print(f"{'='*50}")
    print(f"Accuracy: {report.metrics.accuracy:.2%}")
    print(f"Avg F1:   {report.metrics.avg_f1:.4f}")

    return report


async def run_with_memory(dataset: str, sample_size: int):
    """运行带反思+记忆的测试（完整进化系统）"""
    logger.info(f"=== EVOLVED (MEMORY+REFLECTION) TEST: {dataset} (n={sample_size}) ===")

    agent, memory, reflection = create_agent("memory")
    evaluator = Evaluator()
    runner = EvaluationRunner(agent=agent, evaluator=evaluator)

    # 记忆检索函数
    def memory_retrieve(question: str) -> str:
        if memory:
            return memory.retrieve(question)
        return ""

    # 反思+存储函数
    def reflect_and_store(question, trajectory_text, expected_answer, agent_answer):
        if reflection:
            result = reflection.reflect_on_failure(
                question=question,
                trajectory_text=trajectory_text,
                expected_answer=expected_answer,
                agent_answer=agent_answer,
            )
            # 存储到记忆
            if memory:
                memory.store_experience(
                    question=question,
                    outcome="failure",
                    strategy=result.correction,
                    insight=result.general_rule,
                    question_type="unknown",
                )
                memory.store_rule(
                    content=result.general_rule,
                    applies_to="general",
                    source_task_id=result.task_id,
                )

    report = await runner.run(
        dataset_name=dataset,
        mode="with_memory",
        sample_size=sample_size,
        memory_context_fn=memory_retrieve,
        reflection_fn=reflect_and_store,
    )

    # 整合记忆
    if memory:
        memory.consolidate()

    print(f"\n{'='*50}")
    print(f"EVOLVED RESULTS - {dataset}")
    print(f"{'='*50}")
    print(f"Accuracy: {report.metrics.accuracy:.2%}")
    print(f"Avg F1:   {report.metrics.avg_f1:.4f}")
    if memory:
        print(f"Memory Stats: {memory.stats}")

    return report


def run_compare(dataset: str):
    """对比分析"""
    analyzer = ResultAnalyzer()
    report = analyzer.compare(dataset, "baseline", "with_memory")
    analyzer.analyze_failures(dataset, "baseline")
    analyzer.plot_comparison(dataset)
    return report


async def run_full_pipeline(dataset: str, sample_size: int):
    """运行完整流水线：基线 → 反思 → 记忆 → 对比"""
    print("\n" + "=" * 60)
    print("   SELF-EVOLVING AGENT - FULL EVALUATION PIPELINE")
    print("=" * 60)

    # Phase 1: 基线
    print("\n[Phase 1/4] Running baseline...")
    await run_baseline(dataset, sample_size)

    # Phase 2: 反思
    print("\n[Phase 2/4] Running with reflection...")
    await run_with_reflection(dataset, sample_size)

    # Phase 3: 记忆
    print("\n[Phase 3/4] Running with memory...")
    await run_with_memory(dataset, sample_size)

    # Phase 4: 对比
    print("\n[Phase 4/4] Comparing results...")
    run_compare(dataset)

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE")
    print("=" * 60)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="Self-Evolving Task-Solving Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py baseline --dataset 2wiki --sample 50
  python main.py reflection --dataset simpleqa --sample 100
  python main.py memory --dataset 2wiki --sample 100
  python main.py compare --dataset 2wiki
  python main.py full --dataset 2wiki --sample 50
        """,
    )

    parser.add_argument(
        "mode",
        choices=["baseline", "reflection", "memory", "compare", "full", "browsecomp"],
        help="Running mode",
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["simpleqa", "2wiki"],
        default="2wiki",
        help="Dataset to evaluate on (default: 2wiki)",
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    # 配置
    settings.log_level = args.log_level
    setup_logging()

    logger.info(f"Starting: mode={args.mode}, dataset={args.dataset}, sample={args.sample}")

    # 运行
    if args.mode == "baseline":
        asyncio.run(run_baseline(args.dataset, args.sample))
    elif args.mode == "reflection":
        asyncio.run(run_with_reflection(args.dataset, args.sample))
    elif args.mode == "memory":
        asyncio.run(run_with_memory(args.dataset, args.sample))
    elif args.mode == "compare":
        run_compare(args.dataset)
    elif args.mode == "full":
        asyncio.run(run_full_pipeline(args.dataset, args.sample))
    elif args.mode == "browsecomp":
        print("BrowseComp mode - see scripts/run_browsecomp.py")


if __name__ == "__main__":
    main()
