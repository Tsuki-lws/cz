#!/usr/bin/env bash
# 用途: 采集 on-policy 失败轨迹和教师重写。
# 用法: bash distill/scripts/06_on_policy_collect.sh
# 参数: 通过环境变量设置 SEEDS、STUDENT、TEACHER、OUT。

set -euo pipefail

SEEDS=${SEEDS:-distill/data/raw/seed_pool.jsonl}
STUDENT=${STUDENT:-distill/configs/teacher_local.yaml}
TEACHER=${TEACHER:-distill/configs/teacher_api.yaml}
OUT=${OUT:-distill/data/raw/on_policy_pairs.jsonl}

python -m distill.data_collection.on_policy_collect --seeds $SEEDS --student-config $STUDENT --teacher-config $TEACHER --output $OUT
