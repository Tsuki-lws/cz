"""Analyze an eval results jsonl with failure taxonomy and reflections."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from distill.analysis.failure_taxonomy import (
    exact_match,
    loose_match,
    normalized_match,
    reflection_dict,
)
from distill.common import read_jsonl, write_json, write_jsonl


def load_trajectories(path: str | None) -> dict[Any, dict[str, Any]]:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    return {row.get('index'): row for row in read_jsonl(resolved)}


def annotate(results: list[dict[str, Any]], trajectories: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for row in results:
        traj = trajectories.get(row.get('index'))
        reflection = reflection_dict(row, traj)
        annotated.append(
            {
                **row,
                'exact_correct': exact_match(row.get('answer'), row.get('pred')),
                'norm_correct': normalized_match(row.get('answer'), row.get('pred')),
                'loose_correct': loose_match(row.get('answer'), row.get('pred')),
                'failure_type': reflection['failure_type'],
                'task_type': reflection['task_type'],
                'reflection': reflection,
            }
        )
    return annotated


def summarize(annotated: list[dict[str, Any]], label: str) -> dict[str, Any]:
    count = len(annotated)
    by_failure = Counter(row['failure_type'] for row in annotated)
    by_task = Counter(row['task_type'] for row in annotated)
    by_skill = Counter(row['reflection']['activated_skill'] for row in annotated if row['reflection']['activated_skill'] != 'none')

    task_failure: dict[str, Counter[str]] = defaultdict(Counter)
    for row in annotated:
        task_failure[row['task_type']][row['failure_type']] += 1

    def rate(key: str) -> float:
        return mean([bool(row[key]) for row in annotated]) if annotated else 0.0

    return {
        'label': label,
        'count': count,
        'accuracy': rate('exact_correct'),
        'norm_accuracy': rate('norm_correct'),
        'loose_accuracy': rate('loose_correct'),
        'failure_counts': dict(by_failure),
        'task_counts': dict(by_task),
        'activated_skill_counts': dict(by_skill),
        'task_failure_counts': {task: dict(counter) for task, counter in task_failure.items()},
    }


def write_markdown(path: str, summary: dict[str, Any], annotated: list[dict[str, Any]], max_examples: int) -> None:
    lines = [
        f"# Eval Analysis: {summary['label']}",
        '',
        f"- count: {summary['count']}",
        f"- accuracy: {summary['accuracy']:.4f}",
        f"- norm_accuracy: {summary['norm_accuracy']:.4f}",
        f"- loose_accuracy: {summary['loose_accuracy']:.4f}",
        '',
        '## Failure Counts',
        '',
        '| failure_type | count |',
        '| --- | ---: |',
    ]
    for key, value in sorted(summary['failure_counts'].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f'| {key} | {value} |')

    lines.extend(['', '## Activated Skills', '', '| skill | count |', '| --- | ---: |'])
    for key, value in sorted(summary['activated_skill_counts'].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f'| {key} | {value} |')

    lines.extend(['', '## Examples', ''])
    failures = [row for row in annotated if row['failure_type'] != 'CORRECT']
    for row in failures[:max_examples]:
        lines.extend(
            [
                f"### index={row.get('index')} {row['failure_type']}",
                '',
                f"- question: {row.get('instruction', '')}",
                f"- gold: {row.get('answer', '')}",
                f"- pred: {row.get('pred', '')}",
                f"- skill: {row['reflection']['activated_skill']}",
                f"- better_strategy: {row['reflection']['better_strategy']}",
                '',
            ]
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze eval results with failure taxonomy.')
    parser.add_argument('--results', required=True)
    parser.add_argument('--trajectories')
    parser.add_argument('--label', default='eval')
    parser.add_argument('--output-jsonl', default='distill/outputs/analysis/annotated.jsonl')
    parser.add_argument('--summary-json', default='distill/outputs/analysis/summary.json')
    parser.add_argument('--report-md', default='distill/outputs/analysis/report.md')
    parser.add_argument('--max-examples', type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = read_jsonl(args.results)
    trajectories = load_trajectories(args.trajectories)
    annotated = annotate(results, trajectories)
    summary = summarize(annotated, args.label)

    write_jsonl(args.output_jsonl, annotated)
    write_json(args.summary_json, summary)
    write_markdown(args.report_md, summary, annotated, args.max_examples)
    print(f"analyzed {summary['count']} rows; accuracy={summary['accuracy']:.4f}; loose={summary['loose_accuracy']:.4f}")


if __name__ == '__main__':
    main()

