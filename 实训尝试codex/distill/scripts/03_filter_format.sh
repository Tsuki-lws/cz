#!/usr/bin/env bash
# 用途: 过滤教师数据并格式化为训练集。
# 用法: bash distill/scripts/03_filter_format.sh
# 参数: 可通过环境变量设置 INPUT、FILTERED、FINAL。

set -euo pipefail

INPUT=${INPUT:-distill/data/raw/teacher_local/episodes.jsonl}
FILTERED=${FILTERED:-distill/data/filtered/teacher_local.filtered.jsonl}
FINAL=${FINAL:-distill/data/final/distill_sft_v1.json}
ON_POLICY=${ON_POLICY:-distill/data/raw/on_policy_pairs.jsonl}
DPO_FINAL=${DPO_FINAL:-distill/data/final/distill_dpo_v1.json}

python -m distill.data_collection.filter --input $INPUT --output $FILTERED
python -m distill.data_collection.format_sft --input $FILTERED --output $FINAL
if [ -f "$ON_POLICY" ]; then
  python -m distill.data_collection.format_dpo --input $ON_POLICY --output $DPO_FINAL
fi
