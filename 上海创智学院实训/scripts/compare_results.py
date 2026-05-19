"""
对比分析脚本
对比基线和进化后的结果，生成报告和图表
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.analysis import ResultAnalyzer


def compare_results(dataset: str = "2wiki"):
    """对比基线和进化后的结果"""
    analyzer = ResultAnalyzer()

    print("\n" + "=" * 60)
    print(f"  COMPARISON ANALYSIS - {dataset.upper()}")
    print("=" * 60)

    # 1. 基线 vs 反思
    print("\n[1] Baseline vs Reflection:")
    try:
        analyzer.compare(dataset, "baseline", "with_reflection")
    except Exception as e:
        print(f"  (No reflection data available: {e})")

    # 2. 基线 vs 完整进化
    print("\n[2] Baseline vs Evolved (Reflection + Memory):")
    try:
        report = analyzer.compare(dataset, "baseline", "with_memory")
    except Exception as e:
        print(f"  (No evolved data available: {e})")

    # 3. 失败分析
    print("\n[3] Failure Analysis (Baseline):")
    failures = analyzer.analyze_failures(dataset, "baseline")
    if failures:
        print(f"  Total failures: {failures['total_failures']}")
        print(f"  Failure reasons: {failures['failure_reasons']}")
        if failures.get("sample_failures"):
            print("\n  Sample failed questions:")
            for f in failures["sample_failures"][:3]:
                print(f"    - Q: {f['question'][:80]}...")
                print(f"      Expected: {f['expected']}")
                print(f"      Got: {f['predicted'][:50]}")
                print()

    # 4. 生成图表
    print("\n[4] Generating comparison chart...")
    try:
        analyzer.plot_comparison(dataset)
        print("  Chart saved to data/results/")
    except Exception as e:
        print(f"  (Chart generation failed: {e})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("--dataset", "-d", default="2wiki", choices=["simpleqa", "2wiki"])
    args = parser.parse_args()

    compare_results(args.dataset)
