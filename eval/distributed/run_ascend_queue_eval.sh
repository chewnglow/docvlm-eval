#!/usr/bin/env bash
# End-to-end multi-node Ascend run using independent TP=1 replicas and a shared queue.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/model_presets.sh"

MODEL_KEY="${MODEL_KEY:-${1:-}}"
DATASET="${DATASET:-${2:-}}"
if [[ -z "${MODEL_KEY}" || -z "${DATASET}" ]]; then
  echo "Usage: MODEL_KEY=qwen3.5-9b DATASET=mmlongbench-doc $0" >&2
  exit 2
fi

export DEVICE_TYPE=ascend
export TP_SIZE="${TP_SIZE:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NODE_ID="${NODE_ID:-$(hostname -s 2>/dev/null || echo node-${NODE_RANK})}"
export SPLIT="${SPLIT:-val}"
export MAX_IMAGES="${MAX_IMAGES:-all}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/distributed}"
export PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

if [[ -z "${TOTAL_DEVICES:-}" ]]; then
  echo "Set TOTAL_DEVICES to the total fleet size (16..128)." >&2
  exit 2
fi
NODE_DEVICE_CAPACITY="${NODE_DEVICE_CAPACITY:-${DEVICES_PER_NODE:-8}}"
PLAN_ARGS=(
  --total-devices "${TOTAL_DEVICES}"
  --model-key "${MODEL_KEY}"
  --node-rank "${NODE_RANK}"
  --node-capacity "${NODE_DEVICE_CAPACITY}"
)
if [[ -n "${QWEN_DEVICES:-}" ]]; then
  PLAN_ARGS+=(--qwen-devices "${QWEN_DEVICES}")
fi
if [[ -n "${STEP_DEVICES:-}" ]]; then
  PLAN_ARGS+=(--step-devices "${STEP_DEVICES}")
fi
read -r MODEL_GROUP_DEVICES MODEL_GROUP_NODES LOCAL_DEVICES QWEN_DEVICE_COUNT STEP_DEVICE_COUNT < <(
  "${PYTHON}" "${SCRIPT_DIR}/device_plan.py" "${PLAN_ARGS[@]}"
)
if [[ "${LOCAL_DEVICES}" -eq 0 ]]; then
  echo "${NODE_ID}: NODE_RANK=${NODE_RANK} is outside this model's ${MODEL_GROUP_NODES}-node group; nothing to run."
  exit 0
fi
export DEVICES_PER_NODE="${LOCAL_DEVICES}"
echo "Fleet: ${TOTAL_DEVICES} devices; Qwen: ${QWEN_DEVICE_COUNT}; Step: ${STEP_DEVICE_COUNT}."
echo "${MODEL_KEY}: ${MODEL_GROUP_DEVICES} devices across ${MODEL_GROUP_NODES} node(s); ${NODE_ID} uses ${LOCAL_DEVICES}."

resolve_model_preset "${MODEL_KEY}"
RUN_NAME="${RUN_NAME:-${SERVED_MODEL_NAME}-${DATASET}-${SPLIT}-img${MAX_IMAGES}-tok${MAX_TOKENS}}"
export RUN_NAME
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
export QUEUE_DIR="${QUEUE_DIR:-${RUN_DIR}/queue}"

"${SCRIPT_DIR}/launch_vllm_servers.sh"

if [[ "${NODE_RANK}" == "0" ]]; then
  MAX_IMAGE_ARGS=()
  if [[ "${MAX_IMAGES}" != "all" && -n "${MAX_IMAGES}" ]]; then
    MAX_IMAGE_ARGS=(--max-images "${MAX_IMAGES}")
  fi
  "${PYTHON}" "${SCRIPT_DIR}/build_work_queue.py" \
    --dataset "${DATASET}" \
    --dataset-root "${DATASET_ROOT:-${ROOT_DIR}/dataset}" \
    --split "${SPLIT}" \
    --image-cache "${IMAGE_CACHE:-${OUTPUT_ROOT}/image_cache}" \
    --queue-dir "${QUEUE_DIR}" \
    --max-records-per-task "${MAX_RECORDS_PER_TASK:-4}" \
    "${MAX_IMAGE_ARGS[@]}"
else
  echo "Waiting for queue ${QUEUE_DIR}..."
  for _ in $(seq 1 "${QUEUE_WAIT_POLLS:-720}"); do
    [[ -f "${QUEUE_DIR}/READY" ]] && break
    sleep "${QUEUE_WAIT_INTERVAL:-5}"
  done
  if [[ ! -f "${QUEUE_DIR}/READY" ]]; then
    echo "Timed out waiting for ${QUEUE_DIR}/READY" >&2
    exit 1
  fi
fi

"${SCRIPT_DIR}/run_queue_workers.sh"

if [[ "${NODE_RANK}" == "0" ]]; then
  DATASET="${DATASET}" RUN_DIR="${RUN_DIR}" "${SCRIPT_DIR}/score_run.sh"
fi
