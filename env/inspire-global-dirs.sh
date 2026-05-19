#!/usr/bin/env bash

# Persistent base directory for user-level caches in this environment.
# No personal global directory is available, so use the personal project dir.
export INSPIRE_USER_GLOBAL_DIR="/inspire/qb-ilm2/project/26summer-camp-01/26210300/.global"

mkdir -p "$INSPIRE_USER_GLOBAL_DIR"/{cache,tmp,hf,hf_home,hf_datasets,hf_hub,hf_modules,transformers,pip,conda_pkgs,conda_envs,rclone,xdg_cache,npm,torch,matplotlib}

export XDG_CACHE_HOME="$INSPIRE_USER_GLOBAL_DIR/xdg_cache"
export TMPDIR="$INSPIRE_USER_GLOBAL_DIR/tmp"

export HF_HOME="$INSPIRE_USER_GLOBAL_DIR/hf_home"
export HUGGINGFACE_HUB_CACHE="$INSPIRE_USER_GLOBAL_DIR/hf_hub"
export HF_DATASETS_CACHE="$INSPIRE_USER_GLOBAL_DIR/hf_datasets"
export TRANSFORMERS_CACHE="$INSPIRE_USER_GLOBAL_DIR/transformers"

export PIP_CACHE_DIR="$INSPIRE_USER_GLOBAL_DIR/pip"
export CONDA_PKGS_DIRS="$INSPIRE_USER_GLOBAL_DIR/conda_pkgs"
export CONDA_ENVS_PATH="$INSPIRE_USER_GLOBAL_DIR/conda_envs"
export RCLONE_CONFIG_DIR="$INSPIRE_USER_GLOBAL_DIR/rclone"

export NPM_CONFIG_CACHE="$INSPIRE_USER_GLOBAL_DIR/npm"
export TORCH_HOME="$INSPIRE_USER_GLOBAL_DIR/torch"
export MPLCONFIGDIR="$INSPIRE_USER_GLOBAL_DIR/matplotlib"

# Local service endpoints for this project.
export SANDBOX_BASE_URL="${SANDBOX_BASE_URL:-http://127.0.0.1:8080}"
export SEARCH_PROXY_URL="${SEARCH_PROXY_URL:-http://127.0.0.1:1227}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/user-b260c9e2-91ae-48ff-bfce-dcfd887a0358/vscode/a41611cb-40b3-4a11-9e03-c916019c5521/2a946687-0b65-48c4-af9a-14410787f628/proxy/8000/v1}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
