#!/usr/bin/env bash
set -euo pipefail

cd /inspire/qb-ilm2/project/26summer-camp-01/26210300/实训尝试codex

BASE_MODEL="/inspire/qb-ilm2/project/26summer-camp-01/public/Qwen3.5-9B"
OUT_ROOT="distill/outputs/qwen35_9b_trackg_sft_full"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pegp/bin/python}"
ZERO_SCRIPT="${ZERO_SCRIPT:-tools/zero_to_fp32_nodeepspeed.py}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"

export HF_HOME="${HF_HOME:-/inspire/qb-ilm2/project/26summer-camp-01/26210300/.global/hf_home}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

convert_one() {
  local step="$1"
  local ckpt="${OUT_ROOT}/checkpoint-${step}"
  local out="${OUT_ROOT}_sglang_hf/checkpoint-${step}"
  local log="distill/logs/convert_checkpoint_${step}.log"

  mkdir -p "$(dirname "$log")" "${OUT_ROOT}_sglang_hf"
  echo "[$(date '+%F %T')] converting checkpoint-${step}" | tee "$log"
  echo "checkpoint: $ckpt" | tee -a "$log"
  echo "output:     $out" | tee -a "$log"

  rm -rf "$out"
  "$PYTHON_BIN" "$ZERO_SCRIPT" \
    --safe_serialization \
    --max_shard_size "$MAX_SHARD_SIZE" \
    "$ckpt" "$out" 2>&1 | tee -a "$log"

  "$PYTHON_BIN" tools/prepare_sglang_hf_dir.py \
    --base-model "$BASE_MODEL" \
    --output-dir "$out" 2>&1 | tee -a "$log"

  echo "[$(date '+%F %T')] done checkpoint-${step}" | tee -a "$log"
}

if [[ "${1:-}" == "693" || "${1:-}" == "1260" ]]; then
  convert_one "$1"
else
  convert_one 693 &
  pid_693=$!
  convert_one 1260 &
  pid_1260=$!
  wait "$pid_693"
  wait "$pid_1260"
fi

echo "Converted outputs:"
du -sh "${OUT_ROOT}_sglang_hf"/checkpoint-* 2>/dev/null || true
