#!/usr/bin/env bash
# 用途: 采集 on-policy 失败轨迹和教师重写。
# 用法: bash distill/scripts/06_on_policy_collect.sh
# 参数: 通过环境变量设置 SEEDS、STUDENT、TEACHER、OUT。

set -euo pipefail

SEEDS=${SEEDS:-distill/data/raw/seed_pool.jsonl}
STUDENT=${STUDENT:-distill/configs/student_qwen35_9b.yaml}
TEACHER=${TEACHER:-distill/configs/teacher_api.yaml}
VISION_TEACHER=${VISION_TEACHER:-distill/configs/teacher_vl_api.yaml}
JUDGE=${JUDGE:-distill/configs/teacher_api.yaml}
OUT=${OUT:-distill/data/raw/on_policy_pairs.jsonl}

python -m distill.data_collection.on_policy_collect --seeds $SEEDS --student-config $STUDENT --teacher-config $TEACHER --vision-teacher-config $VISION_TEACHER --judge-config $JUDGE --output $OUT
