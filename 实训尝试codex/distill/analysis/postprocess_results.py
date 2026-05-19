from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distill.common import read_jsonl, write_json, write_jsonl
from shared_sii_adapter.react_runner import extract_final_answer


def normalize_answer(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "").strip().lower())
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"（[^）]*）", " ", value)
    value = re.sub(r"[`*_\"“”‘’.,!?;:，。！？；：、\[\]{}]", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


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


def summarize(rows: list[dict]) -> dict:
    exact = []
    norm = []
    loose = []
    changed = 0
    for row in rows:
        gold = str(row.get("answer", ""))
        pred = str(row.get("pred", ""))
        changed += int(pred != str(row.get("raw_pred", pred)))
        exact.append(pred.strip().lower() == gold.strip().lower())
        norm.append(normalize_answer(pred) == normalize_answer(gold))
        loose.append(loose_match(gold, pred))
    return {
        "count": len(rows),
        "changed_count": changed,
        "accuracy": mean(exact) if exact else 0.0,
        "norm_accuracy": mean(norm) if norm else 0.0,
        "loose_accuracy": mean(loose) if loose else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess distill result predictions with the TTL answer extractor.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    out = []
    for row in rows:
        new_row = dict(row)
        raw_pred = str(row.get("pred", ""))
        pred = extract_final_answer(raw_pred)
        new_row["raw_pred"] = raw_pred
        new_row["pred"] = pred
        out.append(new_row)
    write_jsonl(Path(args.output), out)
    write_json(Path(args.summary_output), summarize(out))


if __name__ == "__main__":
    main()
