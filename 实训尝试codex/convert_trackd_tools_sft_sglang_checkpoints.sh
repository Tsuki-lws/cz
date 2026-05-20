#!/usr/bin/env bash
set -euo pipefail

cd /inspire/qb-ilm2/project/26summer-camp-01/26210300/实训尝试codex

BASE_MODEL="${BASE_MODEL:-/inspire/qb-ilm2/project/26summer-camp-01/public/Qwen3.5-9B}"
OUT_ROOT="${OUT_ROOT:-distill/outputs/qwen35_9b_trackd_tools_sft_500_clean}"
SGLANG_ROOT="${SGLANG_ROOT:-${OUT_ROOT}_sglang_hf}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/pegp/bin/python}"
ZERO_SCRIPT="${ZERO_SCRIPT:-tools/zero_to_fp32_nodeepspeed.py}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"

export HF_HOME="${HF_HOME:-/inspire/qb-ilm2/project/26summer-camp-01/26210300/.global/hf_home}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

copy_metadata() {
  local ckpt="$1"
  local out="$2"

  # Keep checkpoint-side metadata first because the fine-tuning run may have
  # written a task-specific chat template or processor config.
  for name in \
    config.json \
    generation_config.json \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    vocab.json \
    merges.txt \
    chat_template.json \
    chat_template.jinja \
    preprocessor_config.json \
    processor_config.json \
    video_preprocessor_config.json; do
    if [[ -f "${ckpt}/${name}" ]]; then
      cp -f "${ckpt}/${name}" "${out}/${name}"
    elif [[ -f "${BASE_MODEL}/${name}" ]]; then
      cp -f "${BASE_MODEL}/${name}" "${out}/${name}"
    fi
  done
}

convert_one() {
  local step="$1"
  local ckpt="${OUT_ROOT}/checkpoint-${step}"
  local out="${SGLANG_ROOT}/checkpoint-${step}"
  local log="distill/logs/convert_trackd_tools_sft_checkpoint_${step}.log"

  if [[ ! -d "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$log")" "$SGLANG_ROOT"
  echo "[$(date '+%F %T')] converting checkpoint-${step}" | tee "$log"
  echo "checkpoint: $ckpt" | tee -a "$log"
  echo "output:     $out" | tee -a "$log"
  echo "python:     $PYTHON_BIN" | tee -a "$log"

  rm -rf "$out"
  "$PYTHON_BIN" "$ZERO_SCRIPT" \
    --safe_serialization \
    --max_shard_size "$MAX_SHARD_SIZE" \
    "$ckpt" "$out" 2>&1 | tee -a "$log"

  copy_metadata "$ckpt" "$out" 2>&1 | tee -a "$log"

  echo "[$(date '+%F %T')] done checkpoint-${step}" | tee -a "$log"
}

if [[ "$#" -gt 0 ]]; then
  for step in "$@"; do
    convert_one "$step"
  done
else
  for step in 180 210 240; do
    convert_one "$step"
  done
fi

echo "Converted outputs:"
du -sh "${SGLANG_ROOT}"/checkpoint-* 2>/dev/null || true
