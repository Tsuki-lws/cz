# Seed dataset loading and test-set dedup.

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image

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
    seed = {
        'id': f'{dataset_name}-{idx}',
        'source_dataset': dataset_name,
        'question': as_text(row.get(spec['question_field'])),
        'answer': as_text(row.get(spec['answer_field'])),
        'metadata': {'raw_index': idx},
    }
    if image_value is not None:
        image_payload = normalize_image_value(image_value, dataset_name=dataset_name, idx=idx)
        seed.update(image_payload)
    else:
        seed['image'] = ''
    return seed


def normalize_image_value(image_value: Any, *, dataset_name: str, idx: int) -> dict[str, str]:
    if isinstance(image_value, str):
        text = image_value.strip()
        if text.startswith(('http://', 'https://')):
            return {'image': text, 'image_url': text, 'image_path': '', 'image_b64': ''}
        return {'image': text, 'image_url': '', 'image_path': text if Path(text).exists() else '', 'image_b64': ''}

    save_path = Path('distill/data/cache/images') / dataset_name / f'{idx}.jpg'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_value, dict):
        for key in ('path', 'file_path', 'filename'):
            value = image_value.get(key)
            if value and Path(str(value)).exists():
                text = str(value)
                return {'image': text, 'image_url': '', 'image_path': text, 'image_b64': ''}
        for key in ('bytes', 'data'):
            value = image_value.get(key)
            if isinstance(value, (bytes, bytearray)):
                save_path.write_bytes(bytes(value))
                return {'image': str(save_path), 'image_url': '', 'image_path': str(save_path), 'image_b64': ''}
        return {'image': as_text(image_value), 'image_url': '', 'image_path': '', 'image_b64': ''}

    if isinstance(image_value, Image.Image):
        image_value.save(save_path, format='JPEG')
        return {'image': str(save_path), 'image_url': '', 'image_path': str(save_path), 'image_b64': ''}

    if hasattr(image_value, 'save'):
        try:
            image_value.save(save_path)
            return {'image': str(save_path), 'image_url': '', 'image_path': str(save_path), 'image_b64': ''}
        except Exception:  # noqa: BLE001
            pass

    if isinstance(image_value, (bytes, bytearray)):
        save_path.write_bytes(bytes(image_value))
        return {'image': str(save_path), 'image_url': '', 'image_path': str(save_path), 'image_b64': ''}

    text = as_text(image_value)
    try:
        raw = base64.b64decode(text, validate=False)
        if raw:
            return {'image': text, 'image_url': '', 'image_path': '', 'image_b64': text}
    except Exception:  # noqa: BLE001
        pass
    return {'image': text, 'image_url': '', 'image_path': '', 'image_b64': ''}


def load_seed_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dedup_cfg = config.get('dedup', {})
    lower = dedup_cfg.get('normalize_lower', True)
    strip_punctuation = dedup_cfg.get('strip_punctuation', True)
    test_questions = load_test_questions(config.get('test_files', []), lower=lower, strip_punctuation=strip_punctuation)

    kept: list[dict[str, Any]] = []
    stats: dict[str, Any] = {'total_loaded': 0, 'kept': 0, 'dropped_vs_tests': 0, 'datasets': {}}
    seen_questions: set[str] = set()

    for spec in config.get('seed_datasets', []):
        dataset_name = spec['name']
        dataset_kept = 0
        dataset_dropped = 0
        try:
            dataset = load_dataset(spec['path'], spec.get('subset'), split=spec.get('split', 'train'))
        except Exception as exc:  # noqa: BLE001
            stats['datasets'][dataset_name] = {
                'kept': 0,
                'dropped_vs_tests': 0,
                'error': str(exc),
            }
            continue
        limit = int(spec.get('limit', len(dataset)))

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
