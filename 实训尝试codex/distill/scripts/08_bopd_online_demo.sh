#!/usr/bin/env bash
# 用途: 运行小规模 online BOPD demo。
# 用法: bash distill/scripts/08_bopd_online_demo.sh
# 参数: 通过环境变量设置 SEEDS、TEACHER、STUDENT。

set -euo pipefail

SEEDS=${SEEDS:-distill/data/raw/seed_pool.jsonl}
TEACHER=${TEACHER:-distill/configs/teacher_local.yaml}
STUDENT=${STUDENT:-distill/outputs/sft_v1}

python -m distill.train.bopd_online_demo --seed-file $SEEDS --teacher-config $TEACHER --student-model $STUDENT
