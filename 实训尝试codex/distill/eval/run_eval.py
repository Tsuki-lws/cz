# Run evaluation and emit task-aligned outputs.

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from statistics import mean

from distill.common import append_jsonl, load_yaml, read_jsonl, write_json
from shared_sii_adapter.types import RuntimeConfig
from track_d_ttl.agent import run_one


def build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    backend_config = load_yaml(args.config)
    max_tokens = args.max_tokens if args.max_tokens is not None else int(backend_config.get('max_tokens', 4096))
    return RuntimeConfig(
        llm_base_url=str(backend_config['base_url']),
        model_name=str(backend_config['model']),
        max_steps=args.max_steps,
        max_tokens=max_tokens,
        temperature=float(backend_config.get('temperature', 0.2)),
        enable_thinking=False,
        disable_tools=args.disable_tools,
        disable_reflection=args.disable_reflection,
        disable_memory=args.disable_memory,
        structured_final_answer=True,
        output_dir=args.output_dir,
        track_name='track_d',
        run_mode='eval',
        allow_evolution_updates=not args.disable_memory,
        enable_external_assist=False,
    )


def strip_runtime_gold(item: dict, index: int) -> dict:
    task = {
        'index': item.get('index', index),
        'instruction': item.get('instruction') or item.get('question') or item.get('query') or '',
        'image': item.get('image', ''),
        'image_url': item.get('image_url', ''),
        'image_path': item.get('image_path', ''),
        'image_b64': item.get('image_b64', ''),
    }
    return {key: value for key, value in task.items() if value not in (None, '')}


def run_eval(args: argparse.Namespace) -> None:
    dataset = read_jsonl(args.input)
    runtime = build_runtime(args)

    metrics = {'accuracy': [], 'norm_accuracy': [], 'loose_accuracy': [], 'tokens': [], 'turns': [], 'latency': [], 'tool_calls': []}
    for path in (args.results_output, args.trajectories_output, args.summary_output):
        resolved = Path(path)
        if resolved.exists() and not args.resume:
            resolved.unlink()

    for index, item in enumerate(dataset):
        task = strip_runtime_gold(item, index)
        result = run_one(task, runtime)
        gold = str(item.get('answer', ''))
        correct = int(result.pred.strip().lower() == gold.strip().lower())
        metrics['accuracy'].append(correct)
        metrics['norm_accuracy'].append(int(normalize_answer(result.pred) == normalize_answer(gold)))
        metrics['loose_accuracy'].append(int(loose_match(gold, result.pred)))
        metrics['tokens'].append(result.metrics.get('tokens', 0))
        metrics['turns'].append(result.metrics.get('turns', 0))
        metrics['latency'].append(result.metrics.get('latency', 0))
        metrics['tool_calls'].append(result.metrics.get('tool_calls', 0))

        append_jsonl(args.results_output, {
            'index': item.get('index', index),
            'instruction': item.get('instruction') or item.get('question') or item.get('query') or '',
            'image': item.get('image', ''),
            'image_url': item.get('image_url', ''),
            'answer': item.get('answer', ''),
            'pred': result.pred,
        })
        append_jsonl(args.trajectories_output, {'index': item.get('index', index), 'records': result.trajectory, 'debug': result.debug})

    summary = {
        'accuracy': mean(metrics['accuracy']) if metrics['accuracy'] else 0.0,
        'norm_accuracy': mean(metrics['norm_accuracy']) if metrics['norm_accuracy'] else 0.0,
        'loose_accuracy': mean(metrics['loose_accuracy']) if metrics['loose_accuracy'] else 0.0,
        'avg_tokens': mean(metrics['tokens']) if metrics['tokens'] else 0.0,
        'avg_turns': mean(metrics['turns']) if metrics['turns'] else 0.0,
        'avg_latency': mean(metrics['latency']) if metrics['latency'] else 0.0,
        'avg_tool_calls': mean(metrics['tool_calls']) if metrics['tool_calls'] else 0.0,
    }
    write_json(args.summary_output, summary)


def normalize_answer(text: object) -> str:
    value = unicodedata.normalize('NFKC', str(text or '').strip().lower())
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'\([^)]*\)', ' ', value)
    value = re.sub(r'（[^）]*）', ' ', value)
    value = re.sub(r'[`*_"“”‘’.,!?;:，。！？；：、\[\]{}]', ' ', value)
    value = re.sub(r'\b(the|a|an)\b', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def loose_match(gold: object, prediction: object) -> bool:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(prediction)
    if not gold_norm or not pred_norm:
        return False
    if gold_norm == pred_norm:
        return True
    if len(pred_norm) <= max(len(gold_norm) * 4, len(gold_norm) + 30) and gold_norm in pred_norm:
        return True
    return len(gold_norm) <= max(len(pred_norm) * 4, len(pred_norm) + 30) and pred_norm in gold_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run agent evaluation on a jsonl dataset.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-output', default='distill/outputs/results.jsonl')
    parser.add_argument('--trajectories-output', default='distill/outputs/trajectories.jsonl')
    parser.add_argument('--summary-output', default='distill/outputs/eval_summary.json')
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--prompt-mode', choices=['direct_vqa', 'react'], default='direct_vqa', help='Kept for CLI compatibility; track_d_ttl prompt is used.')
    parser.add_argument('--skill-memory', help='Kept for CLI compatibility; track_d_ttl online memory is used.')
    parser.add_argument('--max-memory-skills', type=int, default=3, help='Kept for CLI compatibility.')
    parser.add_argument('--output-dir', default='distill/outputs/ttl_harness')
    parser.add_argument('--disable-reflection', action='store_true')
    parser.add_argument('--disable-memory', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--disable-tools', action='store_true')
    return parser.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == '__main__':
    main()
