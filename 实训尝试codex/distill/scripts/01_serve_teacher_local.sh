#!/usr/bin/env bash
# 用途: 启动本地教师模型服务。
# 用法: bash distill/scripts/01_serve_teacher_local.sh
# 参数: 通过环境变量覆盖模型或端口。

set -euo pipefail

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-32B}
PORT=${PORT:-30000}

python -m sglang.launch_server --model-path $MODEL_NAME --port $PORT
