#!/usr/bin/env bash
# Convenience wrapper for one 8-GPU node: launch servers, run shards, score.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_KEY="${MODEL_KEY:-${1:-}}"
DATASET="${DATASET:-${2:-}}"
if [[ -z "${MODEL_KEY}" || -z "${DATASET}" ]]; then
  echo "Usage: MODEL_KEY=step3vl-10b DATASET=mmlongbench-doc $0" >&2
  exit 2
fi

export MODEL_KEY DATASET
export NUM_NODES="${NUM_NODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export TOTAL_SHARDS="${TOTAL_SHARDS:-8}"

"${SCRIPT_DIR}/launch_vllm_servers.sh"
"${SCRIPT_DIR}/run_eval_shards.sh"

if [[ "${NODE_RANK}" == "0" ]]; then
  source "${SCRIPT_DIR}/model_presets.sh"
  resolve_model_preset "${MODEL_KEY}"
  RUN_NAME="${RUN_NAME:-${SERVED_MODEL_NAME}-${DATASET}-${SPLIT:-val}-img${MAX_IMAGES:-8}-tok${MAX_TOKENS:-256}}"
  RUN_DIR="${OUTPUT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)/outputs/distributed}/${RUN_NAME}"
  DATASET="${DATASET}" RUN_DIR="${RUN_DIR}" "${SCRIPT_DIR}/score_run.sh"
fi

