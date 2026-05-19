# Convert trajectories to LLaMA-Factory style ShareGPT JSON.

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Any

from distill.common import read_jsonl, write_json


def summarize_records(records: list[dict[str, Any]], *, max_items: int = 8) -> str:
    lines: list[str] = []
    for record in records[:max_items]:
        role = str(record.get('role', ''))
        content = str(record.get('content', '')).strip()
        fn_name = str(record.get('fn_name', '')).strip()
        if role == 'tool' and fn_name:
            lines.append(f'[{role}:{fn_name}] {content[:600]}')
        else:
            lines.append(f'[{role}] {content[:600]}')
    return '\n'.join(line for line in lines if line)


def build_human_prompt(episode: dict[str, Any]) -> str:
    prompt = str(episode['question'])
    if resolve_image_path(episode):
        prompt = "<image>\n" + prompt
    elif episode.get('image_url'):
        prompt += f"\nimage_url: {episode['image_url']}"
    return prompt


def build_teacher_target(episode: dict[str, Any]) -> str:
    summary = summarize_records(episode.get('records', []))
    prediction = str(episode.get('prediction', '')).strip()
    parts: list[str] = []
    if summary:
        parts.append("Teacher trajectory summary:\n" + summary)
    if prediction:
        parts.append("Final answer:\n" + prediction)
    return '\n\n'.join(parts).strip() or prediction


def resolve_image_path(episode: dict[str, Any]) -> str:
    image_path = str(episode.get('image_path', '')).strip()
    if image_path and Path(image_path).exists():
        return image_path
    image = str(episode.get('image', '')).strip()
    if image and Path(image).exists():
        return image
    image_b64 = str(episode.get('image_b64', '')).strip()
    if image_b64:
        out = Path('distill/data/cache/formatted_images') / f"{episode['id']}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = image_b64
        if payload.startswith('data:image/'):
            payload = payload.split(',', 1)[-1]
        out.write_bytes(base64.b64decode(payload, validate=False))
        return str(out)
    return ''


def episode_to_alpaca(episode: dict) -> dict:
    image_path = resolve_image_path(episode)
    item = {
        'instruction': build_human_prompt(episode),
        'input': '',
        'output': build_teacher_target(episode),
    }
    if image_path:
        item['images'] = [image_path]
    return item


def episode_to_sharegpt(episode: dict) -> dict:
    return {
        'id': episode['id'],
        **episode_to_alpaca(episode),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Format filtered trajectories to ShareGPT JSON.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = read_jsonl(args.input)
    dataset = [episode_to_alpaca(episode) for episode in episodes]
    write_json(args.output, dataset)
    print(f'formatted samples: {len(dataset)}')


if __name__ == '__main__':
    main()
