# Filter teacher trajectories for training.

from __future__ import annotations

import argparse
import ast
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from distill.common import fuzzy_ratio, normalize_text, read_jsonl, write_json, write_jsonl


REJECTION_PHRASES = (
    'cannot answer',
    "can't answer",
    'do not know',
    "don't know",
    'not enough information',
    'unable to determine',
    '无法确定',
    '不知道',
    '不能确定',
)


def normalize_answer(text: Any) -> str:
    value = unicodedata.normalize('NFKC', str(text or '').strip().lower())
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'\([^)]*\)', ' ', value)
    value = re.sub(r'（[^）]*）', ' ', value)
    value = re.sub(r'[`*_"“”‘’.,!?;:，。！？；：、\[\]{}]', ' ', value)
    value = re.sub(r'\b(the|a|an)\b', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def answer_candidates(answer: Any) -> list[Any]:
    if isinstance(answer, (list, tuple, set)):
        return list(answer)
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        if '|' in stripped:
            return [part.strip() for part in stripped.split('|') if part.strip()]
    return [answer]


def answer_matches(gold: Any, prediction: Any) -> bool:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    for candidate in answer_candidates(gold):
        gold_norm = normalize_answer(candidate)
        if not gold_norm:
            continue
        if gold_norm == pred_norm:
            return True

        # Most collected tasks have short factual answers. Keep predictions that
        # include the normalized gold answer, but avoid accepting long rambles.
        if len(pred_norm) <= max(len(gold_norm) * 4, len(gold_norm) + 30) and gold_norm in pred_norm:
            return True
        if len(gold_norm) <= max(len(pred_norm) * 4, len(pred_norm) + 30) and pred_norm in gold_norm:
            return True
    return False


def has_gold_answer(episode: dict) -> bool:
    return any(normalize_answer(candidate) for candidate in answer_candidates(episode.get('answer')))


def has_repetition(text: str) -> bool:
    normalized = normalize_answer(text)
    if not normalized:
        return False
    tokens = normalized.split()
    if len(tokens) >= 24:
        trigrams = [' '.join(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
        counts = Counter(trigrams)
        if counts and counts.most_common(1)[0][1] >= 4:
            return True
    return bool(re.search(r'(.{20,}?)\1{2,}', normalized))


def trajectory_quality_reasons(
    episode: dict,
    *,
    max_turns: int,
    max_tokens: int,
    max_tool_error_rate: float,
) -> list[str]:
    reasons: list[str] = []
    prediction = str(episode.get('prediction') or '').strip()
    records = episode.get('records') or []

    if not prediction:
        reasons.append('empty_prediction')
    if any(phrase in prediction.lower() for phrase in REJECTION_PHRASES):
        reasons.append('refusal_or_unknown')
    if has_repetition(prediction):
        reasons.append('repeated_prediction')
    if episode.get('turns', 0) > max_turns:
        reasons.append('too_many_turns')
    if episode.get('total_tokens', 0) > max_tokens:
        reasons.append('too_many_tokens')
    if not isinstance(records, list) or len(records) < 3:
        reasons.append('missing_or_short_trajectory')
        return reasons

    roles = [record.get('role') for record in records if isinstance(record, dict)]
    if 'user' not in roles or 'assistant' not in roles:
        reasons.append('missing_user_or_assistant_record')

    assistant_contents = [
        str(record.get('content') or '').strip()
        for record in records
        if isinstance(record, dict) and record.get('role') == 'assistant'
    ]
    if not assistant_contents:
        reasons.append('missing_assistant_records')
    if any(content and has_repetition(content) for content in assistant_contents):
        reasons.append('repeated_assistant_record')
    if len([content for content in assistant_contents if content]) != len(set(content for content in assistant_contents if content)):
        reasons.append('duplicate_assistant_record')

    toolish_records = [
        str(record.get('content') or '')
        for record in records
        if isinstance(record, dict)
        and ('<tool_call>' in str(record.get('content') or '') or record.get('fn_name'))
    ]
    error_records = [
        str(record.get('content') or '')
        for record in records
        if isinstance(record, dict)
        and record.get('role') != 'system'
        and re.search(r'\b(error|exception|traceback|timeout|failed|失败|错误)\b', str(record.get('content') or ''), re.I)
    ]
    if toolish_records and len(error_records) / max(len(toolish_records), 1) > max_tool_error_rate:
        reasons.append('high_tool_error_rate')

    return reasons


def rejection_reasons(
    episode: dict,
    test_questions: list[str],
    *,
    max_turns: int,
    max_tokens: int,
    threshold: float,
    require_correct: bool,
    max_tool_error_rate: float,
) -> list[str]:
    reasons = trajectory_quality_reasons(
        episode,
        max_turns=max_turns,
        max_tokens=max_tokens,
        max_tool_error_rate=max_tool_error_rate,
    )
    normalized = normalize_text(episode.get('question', ''))
    for test_q in test_questions:
        if fuzzy_ratio(normalized, test_q) >= threshold:
            reasons.append('test_leakage')
            break
    if require_correct and has_gold_answer(episode) and not answer_matches(episode.get('answer'), episode.get('prediction')):
        reasons.append('answer_mismatch')
    return reasons


def keep_episode(
    episode: dict,
    test_questions: list[str],
    max_turns: int,
    max_tokens: int,
    threshold: float,
    require_correct: bool = True,
    max_tool_error_rate: float = 0.5,
) -> bool:
    return not rejection_reasons(
        episode,
        test_questions,
        max_turns=max_turns,
        max_tokens=max_tokens,
        threshold=threshold,
        require_correct=require_correct,
        max_tool_error_rate=max_tool_error_rate,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Filter collected trajectories.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test-file', action='append', default=[])
    parser.add_argument('--max-turns', type=int, default=10)
    parser.add_argument('--max-tokens', type=int, default=8000)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--allow-answer-mismatch', action='store_true')
    parser.add_argument('--max-tool-error-rate', type=float, default=0.5)
    parser.add_argument('--stats-output', default='distill/logs/filter_stats.json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = read_jsonl(args.input)
    test_questions = []
    for path in args.test_file:
        for row in read_jsonl(path):
            test_questions.append(normalize_text(row.get('instruction', row.get('question', ''))))

    filtered = []
    reason_counts: Counter[str] = Counter()
    reason_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)

    for episode in episodes:
        reasons = rejection_reasons(
            episode,
            test_questions,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            threshold=args.threshold,
            require_correct=not args.allow_answer_mismatch,
            max_tool_error_rate=args.max_tool_error_rate,
        )
        dataset = str(episode.get('id', '')).split('-')[0] or 'unknown'
        by_dataset[dataset]['input'] += 1
        if not reasons:
            filtered.append(episode)
            by_dataset[dataset]['kept'] += 1
            continue
        by_dataset[dataset]['rejected'] += 1
        for reason in reasons:
            reason_counts[reason] += 1
            if len(reason_examples[reason]) < 5:
                reason_examples[reason].append({
                    'id': episode.get('id'),
                    'question': episode.get('question'),
                    'answer': episode.get('answer'),
                    'prediction': episode.get('prediction'),
                })

    write_jsonl(args.output, filtered)
    write_json(args.stats_output, {
        'input': len(episodes),
        'kept': len(filtered),
        'rejected': len(episodes) - len(filtered),
        'reason_counts': dict(reason_counts.most_common()),
        'by_dataset': {dataset: dict(counter) for dataset, counter in sorted(by_dataset.items())},
        'reason_examples': dict(reason_examples),
        'settings': {
            'max_turns': args.max_turns,
            'max_tokens': args.max_tokens,
            'threshold': args.threshold,
            'require_correct': not args.allow_answer_mismatch,
            'max_tool_error_rate': args.max_tool_error_rate,
        },
    })
    print(f'filtered episodes: {len(filtered)} / {len(episodes)}')
    print(f'rejected episodes: {len(episodes) - len(filtered)}')
    print('top rejection reasons:', dict(reason_counts.most_common(10)))


if __name__ == '__main__':
    main()
