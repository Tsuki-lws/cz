#!/usr/bin/env bash
set -euo pipefail

INPUT=${INPUT:-distill/data/eval/simplevqa_99_distill_local_image.jsonl}
CONFIG=${CONFIG:-distill/configs/student_qwen35_9b_remote.yaml}
MEMORY=${MEMORY:-distill/data/memory/simplevqa_skill_memory.json}
OUTPUT_DIR=${OUTPUT_DIR:-distill/outputs/ablation/simplevqa_student}

python -m distill.analysis.run_ablation \
  --input "$INPUT" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 2 \
  --disable-tools \
  --run baseline:direct_vqa:128 \
  --run memory:direct_vqa:128:"$MEMORY" \
  --run react:react:256

