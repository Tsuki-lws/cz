#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-pegp}"
INSTALL_DIR="${INSTALL_DIR:-third_party/LLaMA-Factory}"

test -d "$INSTALL_DIR"

conda run -n "$CONDA_ENV" python -m pip install --no-deps --ignore-requires-python -e "$INSTALL_DIR"
conda run -n "$CONDA_ENV" python -m pip install --no-deps \
  "accelerate==1.11.0" \
  "peft==0.18.1" \
  "trl==0.24.0" \
  "torchdata==0.11.0" \
  "hf-transfer" \
  "sse-starlette" \
  "py-cpuinfo" \
  "hjson" \
  "msgpack" \
  "av==16.0.0" \
  "fire" \
  "tyro<0.9.0" \
  "gradio==5.50.0" \
  "termcolor" \
  "aiofiles==24.1.0" \
  "brotli" \
  "ffmpy" \
  "gradio-client==1.14.0" \
  "groovy" \
  "pydub" \
  "ruff" \
  "safehttpx" \
  "semantic-version" \
  "shtab" \
  "websockets==15.0.1"

DS_BUILD_OPS=0 conda run -n "$CONDA_ENV" python -m pip install --no-deps "deepspeed==0.16.9"
conda run -n "$CONDA_ENV" llamafactory-cli version
