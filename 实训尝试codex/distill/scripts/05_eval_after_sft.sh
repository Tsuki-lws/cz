#!/usr/bin/env bash
# 用途: 评估 SFT 后模型。
# 用法: bash distill/scripts/05_eval_after_sft.sh
# 参数: 可通过环境变量设置 INPUT、CONFIG。

set -euo pipefail

INPUT=${INPUT:-distill/data/eval/simplevqa_2wiki.jsonl}
CONFIG=${CONFIG:-distill/configs/teacher_local.yaml}

python -m distill.eval.run_eval --input $INPUT --config $CONFIG
