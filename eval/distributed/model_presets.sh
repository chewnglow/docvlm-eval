#!/usr/bin/env bash
# Model presets shared by distributed vLLM/eval scripts.

set -euo pipefail

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

resolve_model_preset() {
  local model_key="${1:?model key is required}"
  local root
  root="$(repo_root)"

  MODEL_EXTRA_ARGS=()
  case "${model_key}" in
    qwen3.5-9b|Qwen3.5-9B)
      MODEL_PATH="${MODEL_PATH:-${root}/models/Qwen3.5-9B}"
      SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-9B}"
      MODEL_EXTRA_ARGS=(
        --trust-remote-code
        --reasoning-parser qwen3
      )
      ;;
    qwen3.5-9b-base|Qwen3.5-9B-Base)
      MODEL_PATH="${MODEL_PATH:-${root}/models/Qwen3.5-9B-Base}"
      SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-9B-Base}"
      MODEL_EXTRA_ARGS=(
        --trust-remote-code
        --reasoning-parser qwen3
      )
      ;;
    step3vl-10b|Step3VL10B)
      MODEL_PATH="${MODEL_PATH:-${root}/models/Step3VL10B}"
      SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Step3VL10B}"
      MODEL_EXTRA_ARGS=(
        --trust-remote-code
        --reasoning-parser deepseek_r1
        --enable-auto-tool-choice
        --tool-call-parser hermes
      )
      ;;
    step3vl-10b-base|Step3VL10B-Base)
      MODEL_PATH="${MODEL_PATH:-${root}/models/Step3VL10B-base}"
      SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Step3VL10B-Base}"
      MODEL_EXTRA_ARGS=(
        --trust-remote-code
        --reasoning-parser deepseek_r1
      )
      ;;
    *)
      echo "Unknown MODEL_KEY='${model_key}'" >&2
      echo "Known: qwen3.5-9b, qwen3.5-9b-base, step3vl-10b, step3vl-10b-base" >&2
      return 2
      ;;
  esac

  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    echo "Override with MODEL_PATH=/abs/path if the directory name differs." >&2
    return 2
  fi
}

