#!/usr/bin/env bash
# Launch one vLLM OpenAI-compatible server per local GPU on this node.

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
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 64}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/outputs/distributed/logs/${MODEL_KEY}/node-${NODE_RANK}}"
PID_DIR="${PID_DIR:-${ROOT_DIR}/outputs/distributed/pids/${MODEL_KEY}/node-${NODE_RANK}}"

resolve_model_preset "${MODEL_KEY}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

echo "Launching ${GPUS_PER_NODE} vLLM servers for ${SERVED_MODEL_NAME} on node ${NODE_RANK}"
for LOCAL_RANK in $(seq 0 $((GPUS_PER_NODE - 1))); do
  PORT=$((BASE_PORT + LOCAL_RANK))
  LOG_FILE="${LOG_DIR}/gpu-${LOCAL_RANK}.log"
  PID_FILE="${PID_DIR}/gpu-${LOCAL_RANK}.pid"
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "GPU ${LOCAL_RANK}: already running pid $(cat "${PID_FILE}")"
    continue
  fi

  echo "GPU ${LOCAL_RANK}: port ${PORT}, log ${LOG_FILE}"
  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${LOCAL_RANK}" \
    vllm serve "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
      "${MODEL_EXTRA_ARGS[@]}" \
      ${VLLM_EXTRA_ARGS:-}
  ) >"${LOG_FILE}" 2>&1 &
  echo "$!" >"${PID_FILE}"
done

echo "Waiting for servers..."
for LOCAL_RANK in $(seq 0 $((GPUS_PER_NODE - 1))); do
  PORT=$((BASE_PORT + LOCAL_RANK))
  "${ROOT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/wait_openai_server.py" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --timeout "${SERVER_WAIT_TIMEOUT:-1200}"
done
echo "All local vLLM servers are ready."
