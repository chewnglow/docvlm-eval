#!/usr/bin/env bash
# Run the three benchmark datasets sequentially for one model/NPU node group.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_KEY="${MODEL_KEY:-${1:-}}"
if [[ -z "${MODEL_KEY}" ]]; then
  echo "Usage: MODEL_KEY=qwen3.5-9b $0" >&2
  exit 2
fi
export MODEL_KEY

BENCHMARKS="${BENCHMARKS:-mmlongbench-doc:val longdocurl:val slidevqa:val}"
for SPEC in ${BENCHMARKS}; do
  export DATASET="${SPEC%%:*}"
  export SPLIT="${SPEC#*:}"
  unset RUN_NAME QUEUE_DIR
  echo "Starting ${MODEL_KEY} on ${DATASET}:${SPLIT}"
  "${SCRIPT_DIR}/run_ascend_queue_eval.sh"
done
