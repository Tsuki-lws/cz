#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-pegp}"
BASE_MODEL="/inspire/qb-ilm2/project/26summer-camp-01/public/Qwen3.5-9B"
OUTPUT_DIR="${OUTPUT_DIR:-distill/outputs/qwen35_9b_trackg_sft_full}"
EVAL_INPUT="${EVAL_INPUT:-distill/data/eval/simplevqa_2_smoke_local_image.jsonl}"
REPORT_DIR="${REPORT_DIR:-distill/outputs/qwen35_9b_trackg_sft_full_eval}"

mkdir -p "$REPORT_DIR"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Missing output dir: $OUTPUT_DIR" >&2
  exit 1
fi

find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V > "$REPORT_DIR/checkpoints.txt"
echo "Checkpoints:"
cat "$REPORT_DIR/checkpoints.txt"

if [ ! -s "$REPORT_DIR/checkpoints.txt" ]; then
  echo "No checkpoints found yet." >&2
  exit 1
fi

while read -r CKPT; do
  STEP="$(basename "$CKPT" | sed 's/checkpoint-//')"
  echo "checkpoint=$CKPT step=$STEP" | tee -a "$REPORT_DIR/summary.txt"
  # Fill this with your serving/eval command once the GPU service is started.
  # The adapter checkpoint is $CKPT and the immutable base model is $BASE_MODEL.
  # Example placeholder:
  # conda run -n "$CONDA_ENV" python -m distill.eval.run_eval \
  #   --input "$EVAL_INPUT" \
  #   --config distill/configs/student_qwen35_9b.yaml \
  #   --results-output "$REPORT_DIR/results_step${STEP}.jsonl" \
  #   --summary-output "$REPORT_DIR/summary_step${STEP}.json"
done < "$REPORT_DIR/checkpoints.txt"
