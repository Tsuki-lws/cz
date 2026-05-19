"""Compare annotated eval runs for A/B testing."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from distill.common import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare annotated eval runs.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output-json', default='distill/outputs/analysis/compare_runs.json')
    parser.add_argument('--report-md', default='distill/outputs/analysis/compare_runs.md')
    return parser.parse_args()


def run_stats(rows: list[dict]) -> dict:
    total = len(rows)
    return {
        'count': total,
        'accuracy': sum(bool(row.get('exact_correct')) for row in rows) / total if total else 0.0,
        'norm_accuracy': sum(bool(row.get('norm_correct')) for row in rows) / total if total else 0.0,
        'loose_accuracy': sum(bool(row.get('loose_correct')) for row in rows) / total if total else 0.0,
        'failure_counts': dict(Counter(row.get('failure_type', 'UNKNOWN') for row in rows)),
        'skill_counts': dict(Counter((row.get('reflection') or {}).get('activated_skill', 'none') for row in rows)),
    }


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise SystemExit('--inputs and --labels must have the same length')

    payload = []
    by_label = {}
    for label, path in zip(args.labels, args.inputs):
        rows = read_jsonl(path)
        stats = run_stats(rows)
        payload.append({'label': label, **stats})
        by_label[label] = {row.get('index'): row for row in rows}

    labels = args.labels
    common_indices = set.intersection(*(set(rows.keys()) for rows in by_label.values())) if by_label else set()
    oracle_exact = 0
    oracle_norm = 0
    only_counts = {label: 0 for label in labels}
    for index in common_indices:
        exact_ok = {label: bool(by_label[label][index].get('exact_correct')) for label in labels}
        norm_ok = {label: bool(by_label[label][index].get('norm_correct')) for label in labels}
        oracle_exact += int(any(exact_ok.values()))
        oracle_norm += int(any(norm_ok.values()))
        for label, ok in exact_ok.items():
            if ok and sum(exact_ok.values()) == 1:
                only_counts[label] += 1

    comparison = {
        'runs': payload,
        'common_count': len(common_indices),
        'oracle_exact_accuracy': oracle_exact / len(common_indices) if common_indices else 0.0,
        'oracle_norm_accuracy': oracle_norm / len(common_indices) if common_indices else 0.0,
        'only_correct_counts': only_counts,
    }
    write_json(args.output_json, comparison)

    lines = ['# Run Comparison', '', '| label | accuracy | norm_accuracy | loose_accuracy |', '| --- | ---: | ---: | ---: |']
    for row in payload:
        lines.append(f"| {row['label']} | {row['accuracy']:.4f} | {row['norm_accuracy']:.4f} | {row['loose_accuracy']:.4f} |")
    lines.extend(
        [
            '',
            f"- common_count: {comparison['common_count']}",
            f"- oracle_exact_accuracy: {comparison['oracle_exact_accuracy']:.4f}",
            f"- oracle_norm_accuracy: {comparison['oracle_norm_accuracy']:.4f}",
            f"- only_correct_counts: {comparison['only_correct_counts']}",
        ]
    )
    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"compared {len(payload)} runs; common={comparison['common_count']}")


if __name__ == '__main__':
    main()

