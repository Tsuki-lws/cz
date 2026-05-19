# Run evaluation and emit task-aligned outputs.

from __future__ import annotations

import argparse
import asyncio
from statistics import mean

from distill.common import append_jsonl, read_jsonl, write_json
from distill.harness.agent import AgentConfig, ReActAgent
from distill.harness.llm_client import AsyncLLMClient, load_backend_config
from distill.harness.tools import ToolRegistry


async def run_eval(args: argparse.Namespace) -> None:
    dataset = read_jsonl(args.input)
    client = AsyncLLMClient(load_backend_config(args.config))
    agent = ReActAgent(client, ToolRegistry.default(), AgentConfig(max_steps=args.max_steps))

    metrics = {'accuracy': [], 'tokens': [], 'turns': [], 'latency': [], 'tool_calls': []}

    for index, item in enumerate(dataset):
        result = await agent.run(
            question_id=str(item.get('index', index)),
            question=item['instruction'],
            image=item.get('image') or None,
            metadata={'answer': item.get('answer', '')},
        )
        correct = int(result.prediction.strip().lower() == str(item.get('answer', '')).strip().lower())
        metrics['accuracy'].append(correct)
        metrics['tokens'].append(result.total_tokens)
        metrics['turns'].append(result.turns)
        metrics['latency'].append(result.elapsed_seconds)
        metrics['tool_calls'].append(result.tool_call_count)

        append_jsonl(args.results_output, {
            'index': item.get('index', index),
            'instruction': item['instruction'],
            'image': item.get('image', ''),
            'answer': item.get('answer', ''),
            'pred': result.prediction,
        })
        append_jsonl(args.trajectories_output, {'index': item.get('index', index), 'records': result.records})

    summary = {
        'accuracy': mean(metrics['accuracy']) if metrics['accuracy'] else 0.0,
        'avg_tokens': mean(metrics['tokens']) if metrics['tokens'] else 0.0,
        'avg_turns': mean(metrics['turns']) if metrics['turns'] else 0.0,
        'avg_latency': mean(metrics['latency']) if metrics['latency'] else 0.0,
        'avg_tool_calls': mean(metrics['tool_calls']) if metrics['tool_calls'] else 0.0,
    }
    write_json(args.summary_output, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run agent evaluation on a jsonl dataset.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-output', default='distill/outputs/results.jsonl')
    parser.add_argument('--trajectories-output', default='distill/outputs/trajectories.jsonl')
    parser.add_argument('--summary-output', default='distill/outputs/eval_summary.json')
    parser.add_argument('--max-steps', type=int, default=8)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_eval(parse_args()))


if __name__ == '__main__':
    main()
