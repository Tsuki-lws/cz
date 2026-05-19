# Convert trajectories to LLaMA-Factory style ShareGPT JSON.

from __future__ import annotations

import argparse

from distill.common import read_jsonl, write_json


def episode_to_sharegpt(episode: dict) -> dict:
    conversations = [{'from': 'human', 'value': episode['question']}]
    conversations.append({'from': 'gpt', 'value': episode.get('prediction', '')})
    return {
        'id': episode['id'],
        'conversations': conversations,
        'system': 'Use tools when needed and provide the final answer in <answer> tags.',
        'tools': ['web_search', 'wiki_search', 'browser_get', 'image_search'],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Format filtered trajectories to ShareGPT JSON.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = read_jsonl(args.input)
    dataset = [episode_to_sharegpt(episode) for episode in episodes]
    write_json(args.output, dataset)
    print(f'formatted samples: {len(dataset)}')


if __name__ == '__main__':
    main()
