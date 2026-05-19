# Compare evaluation runs and generate plots.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from distill.common import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare multiple eval summary json files.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output-dir', default='distill/outputs/compare')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, path in zip(args.labels, args.inputs):
        payload = read_json(path)
        rows.append({'label': label, **payload})

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / 'compare.csv', index=False)
    write_json(output_dir / 'compare.json', rows)

    metric_names = ['accuracy', 'avg_tokens', 'avg_turns', 'avg_latency', 'avg_tool_calls']
    for metric in metric_names:
        plt.figure(figsize=(8, 4))
        plt.bar(frame['label'], frame[metric])
        plt.title(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}.png')
        plt.close()


if __name__ == '__main__':
    main()
