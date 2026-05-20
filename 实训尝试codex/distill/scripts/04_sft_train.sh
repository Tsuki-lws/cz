#!/usr/bin/env bash
# 用途: 运行 LLaMA-Factory SFT。
# 用法: bash distill/scripts/04_sft_train.sh
# 参数: 可通过环境变量设置 CONFIG。

set -euo pipefail

CONFIG=${CONFIG:-distill/train/sft_lf.yaml}

python - <<'PY'
import shutil
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}") from exc

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment; start a GPU runtime before SFT training.")

if shutil.which("llamafactory-cli") is None:
    raise SystemExit("llamafactory-cli is not installed in this environment.")

print(f"CUDA devices: {torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(idx)
    print(f"  {idx}: {props.name}, {props.total_memory // 1024**3} GiB")
PY

llamafactory-cli train $CONFIG
