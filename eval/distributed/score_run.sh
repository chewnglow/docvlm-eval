#!/usr/bin/env bash
# Merge shard outputs and run the benchmark scorer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

DATASET="${DATASET:-${1:-}}"
RUN_DIR="${RUN_DIR:-${2:-}}"
if [[ -z "${DATASET}" || -z "${RUN_DIR}" ]]; then
  echo "Usage: DATASET=mmlongbench-doc RUN_DIR=outputs/distributed/<run-name> $0" >&2
  exit 2
fi

PRED="${RUN_DIR}/predictions.jsonl"
SCORED="${RUN_DIR}/scored.jsonl"
SUMMARY="${RUN_DIR}/metrics.json"

"${PYTHON}" "${SCRIPT_DIR}/merge_predictions.py" \
  --shard-dir "${RUN_DIR}/shards" \
  --output "${PRED}"

"${PYTHON}" "${ROOT_DIR}/eval/score_benchmarks.py" \
  --dataset "${DATASET}" \
  --predictions "${PRED}" \
  --output "${SCORED}" \
  --summary "${SUMMARY}" \
  ${SCORE_EXTRA_ARGS:-}

echo "Wrote:"
echo "  ${PRED}"
echo "  ${SCORED}"
echo "  ${SUMMARY}"
