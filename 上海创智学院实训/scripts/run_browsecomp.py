"""
BrowseComp-Plus打榜脚本
在HuggingFace的BrowseComp-Plus基准上评测Agent并提交结果
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from core.llm import LLMClient
from core.agent import ReActAgent
from core.harness import Harness
from tools.registry import ToolRegistry
from tools.search import SearchTool
from tools.browser import BrowserTool, BrowserNavigateTool
from modules.reflection import ReflectionModule
from modules.memory import MemorySystem


async def run_browsecomp(team_name: str = "SII-TeamName", sample_limit: int = None):
    """
    运行BrowseComp-Plus评测

    BrowseComp-Plus测试深度网页浏览和信息检索能力
    需要Agent能够：多跳搜索、解析复杂网页、综合多源信息

    Args:
        team_name: 队伍名称（用于提交，格式：SII-队伍名称）
        sample_limit: 限制测试数量（None=全部）
    """
    print("=" * 60)
    print(f"  BROWSECOMP-PLUS EVALUATION")
    print(f"  Team: {team_name}")
    print(f"  Model: {settings.llm.model_name}")
    print("=" * 60)

    # 创建完整进化Agent（带反思+记忆）
    llm = LLMClient()
    tool_registry = ToolRegistry()
    tool_registry.register(SearchTool())
    tool_registry.register(BrowserTool())
    tool_registry.register(BrowserNavigateTool())

    # BrowseComp需要更多轮次
    from config.settings import AgentConfig
    browsecomp_config = AgentConfig(
        max_iterations=15,  # 更多轮次
        total_timeout=600,  # 更长超时
    )
    harness = Harness(config=browsecomp_config)

    agent = ReActAgent(llm=llm, tool_registry=tool_registry, harness=harness)
    reflection = ReflectionModule(llm=llm)
    memory = MemorySystem(llm=llm)

    # 加载BrowseComp数据
    # TODO: 从HuggingFace加载BrowseComp-Plus数据集
    # dataset = load_browsecomp_dataset()
    print("\nNote: Please download BrowseComp-Plus dataset first.")
    print("URL: https://huggingface.co/spaces/Tevatron/BrowseComp-Plus")
    print("\nPlaceholder - implement after dataset is available.")

    # 示例流程
    sample_questions = [
        # BrowseComp-Plus的问题通常需要深度网页浏览
        "Find the author of a specific blog post from 2008 that references DVD Talk forum comments about classic horror films.",
    ]

    results = []
    for i, question in enumerate(sample_questions):
        if sample_limit and i >= sample_limit:
            break

        print(f"\n[{i+1}/{len(sample_questions)}] Processing: {question[:60]}...")

        # 检索记忆
        memory_context = memory.retrieve(question)

        # 执行
        result = await agent.solve(
            question=question,
            memory_context=memory_context,
        )

        # 构建提交格式
        submission_entry = {
            "question_id": f"q_{i}",
            "answer": result.answer,
            "confidence": 85 if result.success else 30,
            "explanation": result.trajectory_text[:500],
            "reasoning_path": "\n".join([
                f"Step {s.iteration}: {s.action}({s.action_input})"
                for s in result.trajectory if s.action
            ]),
        }
        results.append(submission_entry)

        # 如果失败，反思并存储
        if not result.success:
            ref = reflection.reflect_on_failure(
                question=question,
                trajectory_text=result.trajectory_text,
            )
            memory.store_experience(
                question=question,
                outcome="failure",
                strategy=ref.correction,
                insight=ref.general_rule,
            )

    # 保存提交文件
    output_dir = Path(settings.eval.results_dir) / "browsecomp"
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_file = output_dir / f"submission_{team_name}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(submission_file, "w", encoding="utf-8") as f:
        json.dump({
            "team_name": team_name,
            "model": settings.llm.model_name,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Submission saved to: {submission_file}")
    print(f"  Submit to: https://huggingface.co/spaces/Tevatron/BrowseComp-Plus")
    print(f"  Team name format: SII-{team_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BrowseComp-Plus evaluation")
    parser.add_argument("--team", default="SII-TeamName", help="Team name for submission")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    asyncio.run(run_browsecomp(args.team, args.limit))
