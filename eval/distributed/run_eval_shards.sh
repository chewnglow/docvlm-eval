#!/usr/bin/env bash
# Run one benchmark shard per local vLLM server/GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/model_presets.sh"

MODEL_KEY="${MODEL_KEY:-${1:-}}"
DATASET="${DATASET:-${2:-}}"
if [[ -z "${MODEL_KEY}" || -z "${DATASET}" ]]; then
  echo "Usage: MODEL_KEY=step3vl-10b DATASET=mmlongbench-doc $0" >&2
  exit 2
fi

NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TOTAL_SHARDS="${TOTAL_SHARDS:-$((NUM_NODES * GPUS_PER_NODE))}"
BASE_PORT="${BASE_PORT:-8000}"
SPLIT="${SPLIT:-val}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/distributed}"
MAX_IMAGES="${MAX_IMAGES:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
LIMIT_PER_SHARD="${LIMIT_PER_SHARD:-}"
IMAGE_CACHE="${IMAGE_CACHE:-${OUTPUT_ROOT}/image_cache}"
API_KEY="${OPENAI_API_KEY:-dummy}"

resolve_model_preset "${MODEL_KEY}"
RUN_NAME="${RUN_NAME:-${SERVED_MODEL_NAME}-${DATASET}-${SPLIT}-img${MAX_IMAGES}-tok${MAX_TOKENS}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SHARD_DIR="${RUN_DIR}/shards"
LOG_DIR="${RUN_DIR}/logs/node-${NODE_RANK}"
mkdir -p "${SHARD_DIR}" "${LOG_DIR}"

TOTAL_COUNT="$("${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/count_records.py" \
  --dataset "${DATASET}" \
  --dataset-root "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --image-cache "${IMAGE_CACHE}")"

echo "Run: ${RUN_NAME}"
echo "Dataset records: ${TOTAL_COUNT}; total shards: ${TOTAL_SHARDS}; node rank: ${NODE_RANK}"

PIDS=()
for LOCAL_RANK in $(seq 0 $((GPUS_PER_NODE - 1))); do
  SHARD_ID=$((NODE_RANK * GPUS_PER_NODE + LOCAL_RANK))
  if [[ "${SHARD_ID}" -ge "${TOTAL_SHARDS}" ]]; then
    continue
  fi
  read -r OFFSET LIMIT < <("${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/shard_plan.py" \
    --total "${TOTAL_COUNT}" \
    --num-shards "${TOTAL_SHARDS}" \
    --shard-id "${SHARD_ID}")
  if [[ -n "${LIMIT_PER_SHARD}" && "${LIMIT}" -gt "${LIMIT_PER_SHARD}" ]]; then
    LIMIT="${LIMIT_PER_SHARD}"
  fi
  PORT=$((BASE_PORT + LOCAL_RANK))
  OUT="${SHARD_DIR}/shard-$(printf "%05d" "${SHARD_ID}")-of-$(printf "%05d" "${TOTAL_SHARDS}").jsonl"
  LOG="${LOG_DIR}/shard-$(printf "%05d" "${SHARD_ID}").log"
  echo "GPU ${LOCAL_RANK}: shard ${SHARD_ID}/${TOTAL_SHARDS}, offset ${OFFSET}, limit ${LIMIT}, port ${PORT}"
  (
    cd "${ROOT_DIR}"
    OPENAI_API_KEY="${API_KEY}" OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
    "${ROOT_DIR}/.venv/bin/python" eval/run_benchmarks.py \
      --dataset "${DATASET}" \
      --dataset-root "${DATASET_ROOT}" \
      --split "${SPLIT}" \
      --model "${SERVED_MODEL_NAME}" \
      --output "${OUT}" \
      --offset "${OFFSET}" \
      --limit "${LIMIT}" \
      --max-images "${MAX_IMAGES}" \
      --max-tokens "${MAX_TOKENS}" \
      --image-cache "${IMAGE_CACHE}" \
      --resume
  ) >"${LOG}" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    STATUS=1
  fi
done

if [[ "${STATUS}" -ne 0 ]]; then
  echo "At least one shard failed. Check ${LOG_DIR}" >&2
  exit "${STATUS}"
fi
echo "All local shards finished for ${RUN_NAME}."

