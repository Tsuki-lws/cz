#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-evo}"
ROOT="/inspire/qb-ilm2/project/26summer-camp-01/26210300"
WORKDIR="${ROOT}/实训尝试codex"
DATASET="${ROOT}/datasets/trackd_2wiki_simplevqa.jsonl"
CONCURRENCY="${CONCURRENCY:-100}"

cd "${WORKDIR}"

python scripts/build_trackd_dataset.py \
  --datasets-root "${ROOT}/datasets" \
  --output "${DATASET}"

export SEARCH_PROXY_URL="${SEARCH_PROXY_URL:-}"
export SERPER_API_KEY="${SERPER_API_KEY:-d0872cf8e3d2c6bdd9f1f15b9b0a7cd81d5bf3d5}"
export JINA_API_KEY="${JINA_API_KEY:-jina_27b632dc368a4d878d77a086367a1493HIydGZpZQJhxBWjflZBGmr89R44M}"
export IMAGE_UPLOADER="${IMAGE_UPLOADER:-uguu}"
export SANDBOX_BASE_URL="${SANDBOX_BASE_URL:-http://127.0.0.1:8080}"
export MAX_BROWSER_PARALLEL_CONCURRENCY="${MAX_BROWSER_PARALLEL_CONCURRENCY:-6}"
export LLM_HTTP_TIMEOUT="${LLM_HTTP_TIMEOUT:-180}"

LLM_BASE_URL="${LLM_BASE_URL:-https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/user-b1acf6ce-25a4-4cb6-b428-f427f4a59686/vscode/b2aa27b1-e0f7-425d-b208-acbd7f40ef68/68f1224c-8cc9-4e87-8701-523c6e59db1f/proxy/8000/}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/user-b260c9e2-91ae-48ff-bfce-dcfd887a0358/vscode/aace7e69-939d-426f-944d-8d2e148bdb2a/926b48b6-abf2-4e2b-b2ff-4dea116721c0/proxy/30000/}"
JUDGE_MODEL="${JUDGE_MODEL:-/inspire/qb-ilm2/project/26summer-camp-01/26210300/Qwen3-32B}"

TRACK_NAME="${TRACK_NAME:-track_d_slot_evo}"

COMMON_ARGS=(
  --input "${DATASET}"
  --track "${TRACK_NAME}"
  --group-id 8
  --llm-base-url "${LLM_BASE_URL}"
  --model "${MODEL_NAME}"
  --judge-base-url "${JUDGE_BASE_URL}"
  --judge-model "${JUDGE_MODEL}"
  --run-mode eval
  --max-steps 20
  --max-tokens 16000
  --score-mode llm
  --judge-concurrency 16
  --write-submission
)

case "${MODE}" in
  baseline|no-evo)
    python -m shared_sii_adapter.run_dataset \
      "${COMMON_ARGS[@]}" \
      --output-dir "${ROOT}/runs/slot_evo_ablation_no_memory" \
      --disable-reflection \
      --disable-memory \
      --disable-evolution-updates \
      --concurrency "${CONCURRENCY}"
    ;;
  evo)
    python -m shared_sii_adapter.run_dataset \
      "${COMMON_ARGS[@]}" \
      --output-dir "${ROOT}/runs/slot_evo_enabled" \
      --concurrency "${CONCURRENCY}"
    ;;
  report)
    python scripts/evolution_report.py \
      --baseline-run "${ROOT}/runs/slot_evo_ablation_no_memory" \
      --evo-run "${ROOT}/runs/slot_evo_enabled" \
      --track-baseline "${TRACK_NAME}" \
      --track-evo "${TRACK_NAME}" \
      --output "${ROOT}/runs/slot_evo_comparison_report.json"
    ;;
  *)
    echo "Usage: bash run_trackd_evo_2datasets.sh [baseline|evo|report]" >&2
    exit 2
    ;;
esac
