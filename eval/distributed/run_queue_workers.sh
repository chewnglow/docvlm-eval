#!/usr/bin/env bash
# Run one pull-based queue worker per local vLLM endpoint.

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
NODE_ID="${NODE_ID:-$(hostname -s 2>/dev/null || echo node-${NODE_RANK})}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-${GPUS_PER_NODE:-8}}"
BASE_PORT="${BASE_PORT:-8000}"
SPLIT="${SPLIT:-val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/distributed}"
MAX_IMAGES="${MAX_IMAGES:-all}"
MAX_TOKENS="${MAX_TOKENS:-256}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
MAX_RETRIES="${MAX_RETRIES:-3}"
IMAGE_DATA_CACHE_MB="${IMAGE_DATA_CACHE_MB:-256}"
LEASE_SECONDS="${LEASE_SECONDS:-3600}"
MAX_TASK_ATTEMPTS="${MAX_TASK_ATTEMPTS:-3}"
API_KEY="${OPENAI_API_KEY:-dummy}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

resolve_model_preset "${MODEL_KEY}"
RUN_NAME="${RUN_NAME:-${SERVED_MODEL_NAME}-${DATASET}-${SPLIT}-img${MAX_IMAGES}-tok${MAX_TOKENS}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
QUEUE_DIR="${QUEUE_DIR:-${RUN_DIR}/queue}"
SHARD_DIR="${RUN_DIR}/shards"
LOG_DIR="${RUN_DIR}/logs/${NODE_ID}"
mkdir -p "${SHARD_DIR}" "${LOG_DIR}"

MAX_IMAGE_ARGS=()
if [[ "${MAX_IMAGES}" != "all" && -n "${MAX_IMAGES}" ]]; then
  MAX_IMAGE_ARGS=(--max-images "${MAX_IMAGES}")
fi

echo "Run: ${RUN_NAME}; node: ${NODE_ID}; queue: ${QUEUE_DIR}"
PIDS=()
for LOCAL_RANK in $(seq 0 $((DEVICES_PER_NODE - 1))); do
  PORT=$((BASE_PORT + LOCAL_RANK))
  WORKER_ID="${NODE_ID}-device-${LOCAL_RANK}"
  LOG="${LOG_DIR}/${WORKER_ID}.log"
  (
    cd "${ROOT_DIR}"
    "${PYTHON}" eval/distributed/queue_worker.py \
      --queue-dir "${QUEUE_DIR}" \
      --output-dir "${SHARD_DIR}" \
      --worker-id "${WORKER_ID}" \
      --model "${SERVED_MODEL_NAME}" \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --api-key "${API_KEY}" \
      --max-tokens "${MAX_TOKENS}" \
      --request-concurrency "${REQUEST_CONCURRENCY}" \
      --request-timeout "${REQUEST_TIMEOUT}" \
      --max-retries "${MAX_RETRIES}" \
      --image-data-cache-mb "${IMAGE_DATA_CACHE_MB}" \
      --lease-seconds "${LEASE_SECONDS}" \
      --max-task-attempts "${MAX_TASK_ATTEMPTS}" \
      "${MAX_IMAGE_ARGS[@]}"
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
  echo "At least one queue worker failed. Check ${LOG_DIR}" >&2
  exit "${STATUS}"
fi
echo "All local queue workers finished for ${RUN_NAME}."
