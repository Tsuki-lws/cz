#!/usr/bin/env bash
# 用途: 初始化 Python 环境并安装依赖。
# 用法: bash distill/scripts/00_setup_env.sh
# 参数: 无。

set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r distill/requirements.txt
