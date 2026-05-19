from __future__ import annotations

import argparse

from distill.common import read_jsonl, write_json
from distill.data_collection.format_sft import build_human_prompt, resolve_image_path


def to_dpo_item(row: dict) -> dict:
    item = {
        'instruction': build_human_prompt(row),
        'input': '',
        'chosen': str(row.get('teacher_rewritten', '')).strip(),
        'rejected': str((row.get('dpo_pair') or {}).get('rejected', '')).strip(),
    }
    image_path = resolve_image_path(row)
    if image_path:
        item['images'] = [image_path]
    return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Format on-policy rewrites into DPO training json.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    dataset = [to_dpo_item(row) for row in rows if str(row.get('teacher_rewritten', '')).strip() and str((row.get('dpo_pair') or {}).get('rejected', '')).strip()]
    write_json(args.output, dataset)
    print(f'formatted dpo samples: {len(dataset)}')


if __name__ == '__main__':
    main()
