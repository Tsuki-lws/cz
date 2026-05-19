"""Run a small, reproducible eval ablation pipeline.

This script executes multiple prompt/memory configurations on the same input,
annotates failures, and writes a cross-run comparison. It is intentionally
simple so experiments can be repeated from the command line without manually
copying paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path) -> None:
    print('+ ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_run_spec(spec: str) -> dict[str, str]:
    parts = spec.split(':')
    if len(parts) not in {3, 4}:
        raise ValueError(
            'run spec must be label:prompt_mode:max_tokens[:skill_memory], '
            f'got {spec!r}'
        )
    label, prompt_mode, max_tokens = parts[:3]
    memory = parts[3] if len(parts) == 4 else ''
    return {
        'label': label,
        'prompt_mode': prompt_mode,
        'max_tokens': max_tokens,
        'skill_memory': memory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run eval + analysis ablations.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', default='distill/outputs/ablation')
    parser.add_argument('--max-steps', type=int, default=2)
    parser.add_argument('--disable-tools', action='store_true')
    parser.add_argument(
        '--run',
        action='append',
        required=True,
        help='Run spec: label:prompt_mode:max_tokens[:skill_memory]',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_paths: list[str] = []
    labels: list[str] = []
    for spec in args.run:
        item = parse_run_spec(spec)
        label = item['label']
        labels.append(label)

        results = output_dir / f'{label}_results.jsonl'
        trajectories = output_dir / f'{label}_trajectories.jsonl'
        summary = output_dir / f'{label}_summary.json'
        annotated = output_dir / f'{label}_annotated.jsonl'
        analysis_summary = output_dir / f'{label}_analysis.json'
        report = output_dir / f'{label}_report.md'

        eval_cmd = [
            sys.executable,
            '-m',
            'distill.eval.run_eval',
            '--input',
            args.input,
            '--config',
            args.config,
            '--results-output',
            str(results),
            '--trajectories-output',
            str(trajectories),
            '--summary-output',
            str(summary),
            '--max-steps',
            str(args.max_steps),
            '--max-tokens',
            item['max_tokens'],
            '--prompt-mode',
            item['prompt_mode'],
        ]
        if args.disable_tools:
            eval_cmd.append('--disable-tools')
        if item['skill_memory']:
            eval_cmd.extend(['--skill-memory', item['skill_memory']])
        run(eval_cmd, cwd=cwd)

        run(
            [
                sys.executable,
                '-m',
                'distill.analysis.analyze_eval',
                '--results',
                str(results),
                '--trajectories',
                str(trajectories),
                '--label',
                label,
                '--output-jsonl',
                str(annotated),
                '--summary-json',
                str(analysis_summary),
                '--report-md',
                str(report),
            ],
            cwd=cwd,
        )
        annotated_paths.append(str(annotated))

    compare_cmd = [
        sys.executable,
        '-m',
        'distill.analysis.compare_runs',
        '--inputs',
        *annotated_paths,
        '--labels',
        *labels,
        '--output-json',
        str(output_dir / 'compare.json'),
        '--report-md',
        str(output_dir / 'compare.md'),
    ]
    run(compare_cmd, cwd=cwd)
    print(f'wrote ablation outputs to {output_dir}')


if __name__ == '__main__':
    main()

