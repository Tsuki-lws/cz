"""Compress structured reflections into a reusable skill memory file."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from distill.common import read_jsonl, write_json


SKILL_TEMPLATES = {
    'image_first_then_verify': {
        'instruction': 'For landmarks, objects, posters, and visual entities, identify the visual target first; if uncertain, prefer a vision teacher or top-k candidates before finalizing.',
        'avoid': 'Do not guess a famous-looking landmark from a weak visual match.',
    },
    'entity_identification_then_verify': {
        'instruction': 'For person/book/author questions, first identify the entity, then answer the requested attribute such as appointer, disease, date, or role.',
        'avoid': 'Do not answer the attribute before the entity is reliable.',
    },
    'verify_date_year': {
        'instruction': 'For date/year questions, verify the entity-year pair and output only the year/date span.',
        'avoid': 'Do not infer a year from visual style alone.',
    },
    'normalize_country_region': {
        'instruction': 'For country or region questions, verify origin carefully and normalize common aliases such as UK/United Kingdom and USA/United States.',
        'avoid': 'Do not confuse filming location, nationality, origin, and current location.',
    },
    'concept_disambiguation': {
        'instruction': 'For category, relation, style, family, or disease questions, choose the most specific concept that answers the exact wording.',
        'avoid': 'Do not return a broad superclass when the question expects a specific label.',
    },
    'answer_normalization': {
        'instruction': 'Output the minimal answer span with no explanation, parentheses, citations, or extra qualifiers.',
        'avoid': 'Do not include surrounding sentences when a short answer is enough.',
    },
    'concise_answer': {
        'instruction': 'Use direct answer mode and keep the final answer under one short line unless the task asks otherwise.',
        'avoid': 'Do not write bullet points or analysis in benchmark answers.',
    },
    'loop_breaker': {
        'instruction': 'If reasoning repeats, stop exploring and provide the best supported final answer immediately.',
        'avoid': 'Do not repeat the same hypothesis or search wording.',
    },
    'tool_protocol_guard': {
        'instruction': 'When tools are enabled, use registered tool calls only; when tools are disabled, answer from image and knowledge without pseudo code.',
        'avoid': 'Do not print search commands or tool code as natural language.',
    },
}


DEFAULT_APPLIES_TO = {
    'image_first_then_verify': ['location_landmark', 'other'],
    'entity_identification_then_verify': ['entity_name', 'other'],
    'verify_date_year': ['date_year'],
    'normalize_country_region': ['country_region'],
    'concept_disambiguation': ['type_concept', 'visual_attribute'],
    'answer_normalization': ['all'],
    'concise_answer': ['all'],
    'loop_breaker': ['all'],
    'tool_protocol_guard': ['all'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build reusable skill memory from annotated eval jsonl files.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-json', default='distill/data/memory/skill_memory.json')
    parser.add_argument('--report-md', default='distill/data/memory/skill_memory.md')
    parser.add_argument('--min-count', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skill_counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)

    for path in args.inputs:
        for row in read_jsonl(path):
            reflection = row.get('reflection') or {}
            skill = reflection.get('activated_skill')
            if not skill or skill == 'none':
                continue
            skill_counts[skill] += 1
            task_counts[skill][reflection.get('task_type', 'other')] += 1
            if len(examples[skill]) < 5:
                examples[skill].append(
                    {
                        'index': row.get('index'),
                        'question': row.get('instruction', ''),
                        'gold': row.get('answer', ''),
                        'pred': row.get('pred', ''),
                        'failure_type': row.get('failure_type'),
                    }
                )

    skills = []
    for skill_name, count in skill_counts.most_common():
        if count < args.min_count:
            continue
        template = SKILL_TEMPLATES.get(skill_name, {})
        applies_to = sorted(task_counts[skill_name], key=lambda key: (-task_counts[skill_name][key], key))
        if not applies_to:
            applies_to = DEFAULT_APPLIES_TO.get(skill_name, ['all'])
        skills.append(
            {
                'name': skill_name,
                'priority': count,
                'count': count,
                'applies_to': applies_to,
                'instruction': template.get('instruction', ''),
                'avoid': template.get('avoid', ''),
                'examples': examples[skill_name],
            }
        )

    payload = {'skills': skills}
    write_json(args.output_json, payload)

    lines = ['# Skill Memory', '', '| skill | count | applies_to |', '| --- | ---: | --- |']
    for skill in skills:
        lines.append(f"| {skill['name']} | {skill['count']} | {', '.join(skill['applies_to'])} |")
    lines.append('')
    for skill in skills:
        lines.extend(
            [
                f"## {skill['name']}",
                '',
                f"- instruction: {skill['instruction']}",
                f"- avoid: {skill['avoid']}",
                '',
            ]
        )
    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {len(skills)} skills to {args.output_json}')


if __name__ == '__main__':
    main()

