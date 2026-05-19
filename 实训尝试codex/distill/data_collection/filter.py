# Filter teacher trajectories for training.

from __future__ import annotations

import argparse

from distill.common import fuzzy_ratio, normalize_text, read_jsonl, write_json, write_jsonl


def keep_episode(episode: dict, test_questions: list[str], max_turns: int, max_tokens: int, threshold: float) -> bool:
    if episode.get('turns', 0) > max_turns:
        return False
    if episode.get('total_tokens', 0) > max_tokens:
        return False
    normalized = normalize_text(episode.get('question', ''))
    for test_q in test_questions:
        if fuzzy_ratio(normalized, test_q) >= threshold:
            return False
    return bool(episode.get('prediction'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Filter collected trajectories.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test-file', action='append', default=[])
    parser.add_argument('--max-turns', type=int, default=10)
    parser.add_argument('--max-tokens', type=int, default=8000)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--stats-output', default='distill/logs/filter_stats.json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = read_jsonl(args.input)
    test_questions = []
    for path in args.test_file:
        for row in read_jsonl(path):
            test_questions.append(normalize_text(row.get('instruction', row.get('question', ''))))

    filtered = [
        episode
        for episode in episodes
        if keep_episode(episode, test_questions, args.max_turns, args.max_tokens, args.threshold)
    ]
    write_jsonl(args.output, filtered)
    write_json(args.stats_output, {'input': len(episodes), 'kept': len(filtered)})
    print(f'filtered episodes: {len(filtered)} / {len(episodes)}')


if __name__ == '__main__':
    main()
