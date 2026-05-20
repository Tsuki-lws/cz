#!/usr/bin/env bash
set -euo pipefail

cd /inspire/qb-ilm2/project/26summer-camp-01/26210300/实训尝试codex

export HF_HOME="${HF_HOME:-/inspire/qb-ilm2/project/26summer-camp-01/26210300/.global/hf_home}"
HF_BIN="${HF_BIN:-/opt/conda/envs/py310/bin/hf}"
REPO_ID="${REPO_ID:-Tsukilws/cz}"
LOCAL_PATH="${LOCAL_PATH:-distill/outputs/qwen35_9b_trackg_sft_full}"
NUM_WORKERS="${NUM_WORKERS:-16}"

"$HF_BIN" upload-large-folder "$REPO_ID" "$LOCAL_PATH" \
  --repo-type model \
  --num-workers "$NUM_WORKERS"
