#!/usr/bin/env bash
# Stop vLLM servers launched by launch_vllm_servers.sh on this node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_KEY="${MODEL_KEY:-${1:-}}"
NODE_RANK="${NODE_RANK:-0}"
NODE_ID="${NODE_ID:-node-${NODE_RANK}}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-${GPUS_PER_NODE:-8}}"
PID_DIR="${PID_DIR:-${ROOT_DIR}/outputs/distributed/pids/${MODEL_KEY}/${NODE_ID}}"

if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: MODEL_KEY=step3vl-10b $0" >&2
  exit 2
fi

for LOCAL_RANK in $(seq 0 $((DEVICES_PER_NODE - 1))); do
  PID_FILE="${PID_DIR}/device-${LOCAL_RANK}.pid"
  if [[ ! -f "${PID_FILE}" ]]; then
    continue
  fi
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Stopping device ${LOCAL_RANK} pid ${PID}"
    kill "${PID}" || true
  fi
  rm -f "${PID_FILE}"
done
