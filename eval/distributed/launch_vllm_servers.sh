#!/usr/bin/env bash
# Launch one vLLM OpenAI-compatible server per local GPU/NPU on this node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/model_presets.sh"

MODEL_KEY="${MODEL_KEY:-${1:-}}"
if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: MODEL_KEY=step3vl-10b $0" >&2
  exit 2
fi

NODE_RANK="${NODE_RANK:-0}"
NODE_ID="${NODE_ID:-node-${NODE_RANK}}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-${GPUS_PER_NODE:-8}}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 256}}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-2}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/distributed/logs/${MODEL_KEY}/${NODE_ID}}"
PID_DIR="${PID_DIR:-${ROOT_DIR}/outputs/distributed/pids/${MODEL_KEY}/${NODE_ID}}"

case "${DEVICE_TYPE}" in
  cuda) VISIBILITY_ENV="CUDA_VISIBLE_DEVICES" ;;
  ascend|npu) VISIBILITY_ENV="ASCEND_RT_VISIBLE_DEVICES" ;;
  *) echo "DEVICE_TYPE must be 'cuda' or 'ascend', got '${DEVICE_TYPE}'" >&2; exit 2 ;;
esac

resolve_model_preset "${MODEL_KEY}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

echo "Launching ${DEVICES_PER_NODE} ${DEVICE_TYPE} vLLM servers for ${SERVED_MODEL_NAME} on ${NODE_ID}"
for LOCAL_RANK in $(seq 0 $((DEVICES_PER_NODE - 1))); do
  PORT=$((BASE_PORT + LOCAL_RANK))
  LOG_FILE="${LOG_DIR}/device-${LOCAL_RANK}.log"
  PID_FILE="${PID_DIR}/device-${LOCAL_RANK}.pid"
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "Device ${LOCAL_RANK}: already running pid $(cat "${PID_FILE}")"
    continue
  fi

  echo "Device ${LOCAL_RANK}: port ${PORT}, log ${LOG_FILE}"
  (
    cd "${ROOT_DIR}"
    export "${VISIBILITY_ENV}=${LOCAL_RANK}"
    SERVER_EXTRA_ARGS=(--mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB}")
    if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
      SERVER_EXTRA_ARGS+=(--enable-prefix-caching)
    fi
    if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
      SERVER_EXTRA_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
    fi
    vllm serve "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
      "${SERVER_EXTRA_ARGS[@]}" \
      "${MODEL_EXTRA_ARGS[@]}" \
      ${VLLM_EXTRA_ARGS:-}
  ) >"${LOG_FILE}" 2>&1 &
  echo "$!" >"${PID_FILE}"
done

echo "Waiting for servers..."
for LOCAL_RANK in $(seq 0 $((DEVICES_PER_NODE - 1))); do
  PORT=$((BASE_PORT + LOCAL_RANK))
  "${PYTHON}" "${SCRIPT_DIR}/wait_openai_server.py" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --timeout "${SERVER_WAIT_TIMEOUT:-1200}"
done
echo "All local vLLM servers are ready."
