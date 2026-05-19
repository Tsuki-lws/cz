#!/usr/bin/env bash
# 用途: 运行 LLaMA-Factory DPO。
# 用法: bash distill/scripts/07_dpo_train.sh
# 参数: 可通过环境变量设置 CONFIG。

set -euo pipefail

CONFIG=${CONFIG:-distill/train/dpo_lf.yaml}

llamafactory-cli train $CONFIG
