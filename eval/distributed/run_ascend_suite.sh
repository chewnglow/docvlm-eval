#!/usr/bin/env bash
# Run the three benchmark datasets sequentially for one model/NPU node group.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/model_presets.sh"

MODEL_KEY="${MODEL_KEY:-${1:-}}"
if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: MODEL_KEY=qwen3.5-9b $0" >&2
  exit 2
fi
export MODEL_KEY

BENCHMARKS="${BENCHMARKS:-mmlongbench-doc:val longdocurl:val slidevqa:val}"
if [[ "${NODE_RANK:-0}" == "0" ]]; then
  resolve_model_preset "${MODEL_KEY}"
  "${PYTHON:-${ROOT_DIR}/.venv/bin/python}" "${SCRIPT_DIR}/suite_plan.py" \
    --output-root "${OUTPUT_ROOT:-${ROOT_DIR}/outputs/distributed}" \
    --dataset-root "${DATASET_ROOT:-${ROOT_DIR}/dataset}" \
    --model-key "${MODEL_KEY}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --benchmarks "${BENCHMARKS}" \
    --max-images "${MAX_IMAGES:-all}" \
    --max-tokens "${MAX_TOKENS:-256}" \
    --device-count "${MODEL_GROUP_DEVICE_COUNT:-0}"
fi
for SPEC in ${BENCHMARKS}; do
  export DATASET="${SPEC%%:*}"
  export SPLIT="${SPEC#*:}"
  unset RUN_NAME QUEUE_DIR
  echo "Starting ${MODEL_KEY} on ${DATASET}:${SPLIT}"
  "${SCRIPT_DIR}/run_ascend_queue_eval.sh"
done
