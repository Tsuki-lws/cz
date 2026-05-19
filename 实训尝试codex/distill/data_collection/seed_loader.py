# Seed dataset loading and test-set dedup.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset

from distill.common import as_text, extract_question_field, load_yaml, normalize_text, read_json, read_jsonl, write_json, write_jsonl


def load_test_questions(paths: list[str], *, lower: bool, strip_punctuation: bool) -> list[str]:
    questions: list[str] = []
    for path in paths:
        resolved = Path(path)
        if not resolved.exists():
            continue
        if resolved.suffix == '.jsonl':
            records = read_jsonl(resolved)
        else:
            payload = read_json(resolved)
            records = payload if isinstance(payload, list) else payload.get('data', [])
        for record in records:
            question = extract_question_field(record)
            if question:
                questions.append(normalize_text(question, lower=lower, strip_punctuation=strip_punctuation))
    return questions


def record_to_seed(dataset_name: str, idx: int, row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    image_value = row.get(spec.get('image_field', ''), None) if spec.get('image_field') else None
    return {
        'id': f'{dataset_name}-{idx}',
        'source_dataset': dataset_name,
        'question': as_text(row.get(spec['question_field'])),
        'answer': as_text(row.get(spec['answer_field'])),
        'image': as_text(image_value),
        'metadata': {'raw_index': idx},
    }


def load_seed_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dedup_cfg = config.get('dedup', {})
    lower = dedup_cfg.get('normalize_lower', True)
    strip_punctuation = dedup_cfg.get('strip_punctuation', True)
    test_questions = load_test_questions(config.get('test_files', []), lower=lower, strip_punctuation=strip_punctuation)

    kept: list[dict[str, Any]] = []
    stats: dict[str, Any] = {'total_loaded': 0, 'kept': 0, 'dropped_vs_tests': 0, 'datasets': {}}
    seen_questions: set[str] = set()

    for spec in config.get('seed_datasets', []):
        dataset = load_dataset(spec['path'], spec.get('subset'), split=spec.get('split', 'train'))
        limit = int(spec.get('limit', len(dataset)))
        dataset_name = spec['name']
        dataset_kept = 0
        dataset_dropped = 0

        for idx, row in enumerate(dataset):
            if idx >= limit:
                break
            stats['total_loaded'] += 1
            seed = record_to_seed(dataset_name, idx, row, spec)
            normalized_question = normalize_text(seed['question'], lower=lower, strip_punctuation=strip_punctuation)
            if not normalized_question or normalized_question in seen_questions:
                continue
            if normalized_question in test_questions:
                dataset_dropped += 1
                stats['dropped_vs_tests'] += 1
                continue
            seen_questions.add(normalized_question)
            kept.append(seed)
            dataset_kept += 1

        stats['datasets'][dataset_name] = {'kept': dataset_kept, 'dropped_vs_tests': dataset_dropped}

    stats['kept'] = len(kept)
    return kept, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Load seed datasets and deduplicate against test sets.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stats-output', default='distill/logs/seed_loader_stats.json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    records, stats = load_seed_records(config)
    write_jsonl(args.output, records)
    write_json(args.stats_output, stats)
    print(f'seed records kept: {len(records)}')


if __name__ == '__main__':
    main()
