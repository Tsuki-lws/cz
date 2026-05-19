#!/usr/bin/env bash
# 用途: 采集教师轨迹。
# 用法: bash distill/scripts/02_collect_teacher.sh
# 参数: 可通过环境变量设置 SEEDS、CONFIG、OUT_DIR。

set -euo pipefail

SEEDS=${SEEDS:-distill/data/raw/seed_pool.jsonl}
CONFIG=${CONFIG:-distill/configs/teacher_local.yaml}
OUT_DIR=${OUT_DIR:-distill/data/raw/teacher_local}

python -m distill.data_collection.teacher_collect --seeds $SEEDS --config $CONFIG --output-dir $OUT_DIR
