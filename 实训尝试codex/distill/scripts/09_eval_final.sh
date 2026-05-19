#!/usr/bin/env bash
# 用途: 运行最终评测并输出结果。
# 用法: bash distill/scripts/09_eval_final.sh
# 参数: 可通过环境变量设置 INPUT、CONFIG。

set -euo pipefail

INPUT=${INPUT:-distill/data/eval/simplevqa_2wiki.jsonl}
CONFIG=${CONFIG:-distill/configs/student_qwen35_9b.yaml}

python -m distill.eval.run_eval --input $INPUT --config $CONFIG
