#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-qwen35_sft_multimodal.yaml}"
CONDA_ENV="${CONDA_ENV:-pegp}"
if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
  CONDA_PREFIX_FOR_ENV="${CONDA_PREFIX_FOR_ENV:-$CONDA_PREFIX}"
else
  CONDA_PREFIX_FOR_ENV="${CONDA_PREFIX_FOR_ENV:-/opt/conda/envs/$CONDA_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-$CONDA_PREFIX_FOR_ENV/bin/python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-$CONDA_PREFIX_FOR_ENV/bin/llamafactory-cli}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export LLAMAFACTORY_ALLOW_TORCH29_CONV3D="${LLAMAFACTORY_ALLOW_TORCH29_CONV3D:-1}"
MODEL_DIR="/inspire/qb-ilm2/project/26summer-camp-01/public/Qwen3.5-9B"
DATA_FILE="distill/data/final/distill_sft_v1.json"
DATASET_INFO="distill/data/final/dataset_info.json"
LOG_DIR="distill/logs"
RUN_LOG="$LOG_DIR/qwen35_9b_trackg_sft_full.log"

mkdir -p "$LOG_DIR" distill/outputs/qwen35_9b_trackg_sft_full

test -d "$MODEL_DIR"
test -f "$DATA_FILE"
test -f "$DATASET_INFO"
test -f "$CONFIG"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -x "$LLAMAFACTORY_CLI" ]; then
  echo "llamafactory-cli not found or not executable: $LLAMAFACTORY_CLI" >&2
  echo "Run: bash setup_llamafactory_pegp.sh" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import json
import shutil
from pathlib import Path

data_path = Path("distill/data/final/distill_sft_v1.json")
data = json.loads(data_path.read_text(encoding="utf-8"))
image_count = sum(1 for item in data if item.get("images"))
print(f"SFT samples: {len(data)}")
print(f"Multimodal samples: {image_count}")

if not data:
    raise SystemExit("SFT dataset is empty.")
if image_count == 0:
    raise SystemExit("Expected multimodal samples, but no images column was found.")

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}") from exc

print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda devices: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Run this script in a GPU job/runtime.")
if torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected 4 visible GPUs from CUDA_VISIBLE_DEVICES=0,1,2,3, got {torch.cuda.device_count()}.")
PY

echo "Starting SFT with config: $CONFIG"
echo "Conda env: $CONDA_ENV"
echo "Python: $PYTHON_BIN"
echo "LLaMA-Factory CLI: $LLAMAFACTORY_CLI"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "FORCE_TORCHRUN=$FORCE_TORCHRUN"
echo "Logs: $RUN_LOG"
"$LLAMAFACTORY_CLI" train "$CONFIG" 2>&1 | tee "$RUN_LOG"
