# Distributed 8-GPU Evaluation

This folder wraps the existing benchmark code in `eval/run_benchmarks.py` and
`eval/score_benchmarks.py` for multi-GPU and multi-node runs.

The design is intentionally simple:

- launch one vLLM OpenAI-compatible server per GPU;
- run one benchmark shard per local server;
- write shard JSONL files into a shared run directory;
- merge and score once after all shards finish.

## Prerequisites

All nodes should have the same project path, Python dependencies, dataset files,
and model files:

```bash
uv pip install --python .venv/bin/python -r eval/requirements-benchmark.txt
```

Install a vLLM version that supports the target models. Qwen3.5 and Step3VL both
require recent vLLM builds; follow each model README if the installed vLLM is too
old.

If you use multiple nodes, set `OUTPUT_ROOT` to a shared filesystem path visible
from all nodes. If you do not have shared storage, copy each node's `shards/`
folder to node 0 before scoring.

## Dataset Preparation

Prepare datasets before launching vLLM servers or eval shards:

```bash
.venv/bin/python eval/fetch_datasets.py \
  --root dataset \
  --max-workers 4 \
  --render-workers 4
```

This downloads and processes:

- MMLongBench-Doc into `dataset/mmlongbench-doc`
- LongDocURL into `dataset/longdocurl`
- SlideVQA into `dataset/slidevqa`

For document datasets, page images are placed under:

```text
dataset/<dataset-name>/document_images/<document-filename-without-suffix>/
```

For multi-node runs, either put `dataset/` on shared storage or run the same
fetch command on every node. If the raw files are already present and you only
need extraction/rendering/layout normalization, run:

```bash
.venv/bin/python eval/fetch_datasets.py --root dataset --process-only
```

The fetch/process command is resumable. It skips completed archive extraction
and rendered PDFs on reruns.

## Model Keys

Use one of:

- `qwen3.5-9b`
- `qwen3.5-9b-base`
- `step3vl-10b`
- `step3vl-10b-base`

If a model directory differs, override it:

```bash
export MODEL_PATH=/abs/path/to/model
export SERVED_MODEL_NAME=CustomName
```

## One Node, 8 GPUs

Example: Step3VL-10B on LongDocURL, capped to 8 images/sample and 256 output
tokens:

```bash
export MODEL_KEY=step3vl-10b
export DATASET=longdocurl
export SPLIT=val
export GPUS_PER_NODE=8
export TOTAL_SHARDS=8
export BASE_PORT=8000
export MAX_IMAGES=8
export MAX_TOKENS=256
export MAX_MODEL_LEN=16384
export LIMIT_MM_PER_PROMPT='{"image": 64}'
export OUTPUT_ROOT="$PWD/outputs/distributed"
export OPENAI_API_KEY=dummy

bash eval/distributed/run_local_8gpu_eval.sh
```

This creates:

```text
outputs/distributed/<model>-<dataset>-<split>-img8-tok256/
  shards/
  logs/
  predictions.jsonl
  scored.jsonl
  metrics.json
```

## Multi-Node, 8 GPUs Total

Example: two nodes, four GPUs per node.

Run on node 0:

```bash
export MODEL_KEY=step3vl-10b
export DATASET=mmlongbench-doc
export NUM_NODES=2
export NODE_RANK=0
export GPUS_PER_NODE=4
export TOTAL_SHARDS=8
export BASE_PORT=8000
export MAX_IMAGES=8
export MAX_TOKENS=256
export OUTPUT_ROOT=/shared/docvlm_eval
export OPENAI_API_KEY=dummy

bash eval/distributed/launch_vllm_servers.sh
bash eval/distributed/run_eval_shards.sh
```

Run on node 1 with the same settings except:

```bash
export NODE_RANK=1
bash eval/distributed/launch_vllm_servers.sh
bash eval/distributed/run_eval_shards.sh
```

After both nodes finish, score on node 0:

```bash
export RUN_DIR=/shared/docvlm_eval/Step3VL10B-mmlongbench-doc-val-img8-tok256
export DATASET=mmlongbench-doc
bash eval/distributed/score_run.sh
```

## Dataset Commands

MMLongBench-Doc:

```bash
export DATASET=mmlongbench-doc
export SPLIT=val
```

LongDocURL:

```bash
export DATASET=longdocurl
export SPLIT=val
```

SlideVQA validation:

```bash
export DATASET=slidevqa
export SPLIT=val
```

SlideVQA test:

```bash
export DATASET=slidevqa
export SPLIT=test
```

`SPLIT` is only meaningful for SlideVQA. MMLongBench-Doc uses its only local
split, and LongDocURL uses the public JSONL file.

## Fast Pilot Runs

For a quick sanity check, cap each shard:

```bash
export LIMIT_PER_SHARD=5
export MAX_IMAGES=4
export MAX_TOKENS=128
bash eval/distributed/run_local_8gpu_eval.sh
```

This runs at most `8 * LIMIT_PER_SHARD` examples on an 8-shard run.

## Stopping Servers

On each node:

```bash
export MODEL_KEY=step3vl-10b
export NODE_RANK=0
export GPUS_PER_NODE=8
bash eval/distributed/stop_vllm_servers.sh
```

## Important Knobs

- `MAX_IMAGES`: biggest runtime lever. Start with `4` or `8`.
- `MAX_TOKENS`: use `128` or `256` for short-answer benchmarking.
- `MAX_MODEL_LEN`: vLLM max input/context length. Default is `16384`; raise only when needed.
- `LIMIT_MM_PER_PROMPT`: should be >= `MAX_IMAGES`.
- `TOTAL_SHARDS`: normally total GPUs across all nodes.
- `BASE_PORT`: local port for GPU 0; GPU k uses `BASE_PORT + k`.
- `VLLM_EXTRA_ARGS`: append extra vLLM flags without editing scripts.
- `SCORE_EXTRA_ARGS`: append scoring flags, e.g. answer extraction via an API.

## Recommended Workflow

1. Pilot one model/dataset:

   ```bash
   export LIMIT_PER_SHARD=5 MAX_IMAGES=4 MAX_TOKENS=128
   bash eval/distributed/run_local_8gpu_eval.sh
   ```

2. Throughput run:

   ```bash
   export LIMIT_PER_SHARD=25 MAX_IMAGES=8 MAX_TOKENS=256
   bash eval/distributed/run_local_8gpu_eval.sh
   ```

3. Full capped-image run:

   ```bash
   unset LIMIT_PER_SHARD
   export MAX_IMAGES=8 MAX_TOKENS=256
   bash eval/distributed/run_local_8gpu_eval.sh
   ```

4. Only for the best model, consider larger `MAX_IMAGES` or full-image settings.
