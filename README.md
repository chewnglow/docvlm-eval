# Document VQA Evaluation

This repository evaluates Qwen3.5-9B and Step3VL-10B on:

- MMLongBench-Doc
- LongDocURL
- SlideVQA

Documents are supplied to the models as ordered raw page images. Generation is
performed through a vLLM-compatible OpenAI endpoint, and predictions are saved
before scoring so generation and metric calculation remain separable.

## Choose an Execution Mode

| Scenario | Entry point | Scheduling |
|---|---|---|
| Existing model endpoint | `eval/run_benchmarks.py` | Concurrent requests to one endpoint |
| One small GPU node | `eval/distributed/run_local_8gpu_eval.sh` | Fixed contiguous shards |
| Ray cluster/web job | `eval/distributed/ray_orchestrator.py` | One submission, automatic node assignment |
| 16–128 Ascend NPUs | `eval/distributed/run_ascend_suite.sh` | Dynamic shared pull queue |

For the Ascend fleet, each NPU hosts one complete TP=1 model replica. Workers
claim document-aware tasks dynamically, process image-heavy work first, retry
transient failures, recover abandoned tasks, and write resumable worker outputs.

## 1. Install Benchmark Dependencies

From the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r eval/requirements-benchmark.txt
```

The Ascend servers additionally need a compatible vLLM/vLLM-Ascend installation
and model-specific dependencies. If you use another environment, set:

```bash
export PYTHON=/absolute/path/to/python
```

## 2. Prepare the Datasets

Download, extract, and render the three benchmarks:

```bash
.venv/bin/python eval/fetch_datasets.py \
  --root dataset \
  --max-workers 4 \
  --render-workers 4
```

This creates:

```text
dataset/
  mmlongbench-doc/
  longdocurl/
  slidevqa/
```

Rendered documents use:

```text
dataset/<dataset-name>/document_images/<document-id>/
```

Dataset preparation is resumable. Useful alternatives:

```bash
# Process files that are already downloaded.
.venv/bin/python eval/fetch_datasets.py --root dataset --process-only

# Prepare only one benchmark.
.venv/bin/python eval/fetch_datasets.py \
  --root dataset \
  --datasets longdocurl \
  --process-only
```

Some PDFs contain icon fonts with malformed glyph names such as
`angle_up`, `angle_down`, or `google_plus`. Poppler can still render these
documents but emits one warning for every affected glyph. The fetch script
suppresses only that known non-fatal warning and prints one count per PDF; all
other Poppler diagnostics remain visible. A `.render_complete` marker is written
only after at least one page PNG exists.

To retry MMLongBench rendering after updating the script:

```bash
.venv/bin/python eval/fetch_datasets.py \
  --root dataset \
  --datasets mmlongbench-doc \
  --process-only
```

For a multi-node run, stage the dataset locally on every node at the same
absolute path. Keep generated queue/results under a shared `OUTPUT_ROOT`.

## 3. Configure Model Locations

The distributed launchers recognize:

- `qwen3.5-9b`
- `qwen3.5-9b-base`
- `step3vl-10b`
- `step3vl-10b-base`

Their default paths are under `models/`. Override them when needed:

```bash
export MODEL_PATH=/absolute/path/to/model
export SERVED_MODEL_NAME=CustomModelName
```

All nodes in one model group must see the same model path.

## 4. Run on 16–128 Ascend NPUs

### One-command Ray orchestration

Use Ray when the cluster or rental platform exposes a Ray dashboard/job endpoint.
The orchestrator is submitted once and performs the per-node launch automatically.
It discovers live compute nodes, divides them between Qwen and Step, assigns
node-local ranks, reserves the advertised NPU capacity, and pins one suite
launcher to every selected node.

Ray must already be running on the allocated nodes. Advertise the Ascend devices
as a custom `NPU` resource when starting each compute node. A CPU-only head node
does not need this resource:

```bash
# Head node
ray start --head --port=6379 --dashboard-host=0.0.0.0

# Every 8-NPU compute node
ray start \
  --address=<head-node-ip>:6379 \
  --resources='{"NPU": 8}'
```

If the head node is also an 8-NPU compute node, add
`--resources='{"NPU": 8}'` to its `ray start` command.

Submit the complete evaluation from a workstation, the Ray dashboard, or the
rental platform's web-job form:

```bash
ray job submit \
  --address=http://<head-node-ip>:8265 \
  --working-dir . \
  -- \
  python eval/distributed/ray_orchestrator.py \
    --total-devices 128 \
    --qwen-devices 64 \
    --step-devices 64 \
    --node-capacity 8 \
    --resource-key NPU \
    --output-root /shared/docvlm_eval \
    --dataset-root /data/docvlm/dataset \
    --python /opt/docvlm/bin/python \
    --qwen-model-path /models/Qwen3.5-9B \
    --step-model-path /models/Step3VL10B
```

In a web form that already supplies the Ray cluster and source working
directory, use only this entrypoint:

```bash
python eval/distributed/ray_orchestrator.py \
  --total-devices 128 \
  --output-root /shared/docvlm_eval \
  --dataset-root /data/docvlm/dataset \
  --python /opt/docvlm/bin/python \
  --qwen-model-path /models/Qwen3.5-9B \
  --step-model-path /models/Step3VL10B
```

Use `--dry-run` to verify the discovered nodes and model allocation without
starting vLLM. Qwen and Step run concurrently on separate node groups; each
group runs its configured `BENCHMARKS` in sequence while all nodes assigned to
that group pull evaluation work concurrently.

The source tree is uploaded by Ray Jobs. `.rayignore` excludes `dataset/`,
`models/`, `outputs/`, and `.venv/` so multi-gigabyte artifacts are not included.
Those directories must instead be mounted or staged on every compute node. In
particular:

- `--output-root` must be one shared filesystem path visible to every node;
- `--dataset-root` and both model paths must exist at the same paths on the
  nodes that run them;
- the Python/vLLM-Ascend environment must already exist on every node;
- every compute node must advertise at least `--node-capacity` units of the
  selected Ray resource.

The orchestrator stops the vLLM servers when the suite finishes. Pass
`--keep-servers` to retain them. If a node launcher fails, successful responses
remain checkpointed in the shared queue; resubmitting the same command resumes
the run. Per-node server logs and PID files are retained under
`OUTPUT_ROOT/_ray_nodes/`.

The manual per-node procedure below remains available when the platform does
not provide Ray.

### Device allocation

`TOTAL_DEVICES` is the size of the complete two-model fleet and must be between
16 and 128. The default split is as even as possible:

| `TOTAL_DEVICES` | Qwen replicas | Step replicas |
|---:|---:|---:|
| 16 | 8 | 8 |
| 48 | 24 | 24 |
| 96 | 48 | 48 |
| 128 | 64 | 64 |

Override the split if one model is slower:

```bash
export TOTAL_DEVICES=96
export QWEN_DEVICES=32
export STEP_DEVICES=64
```

`QWEN_DEVICES + STEP_DEVICES` must equal `TOTAL_DEVICES`. Device groups do not
need to be multiples of the node capacity: a 20-device group on 8-NPU nodes is
automatically launched as `8 + 8 + 4`.

You can inspect an allocation before launching:

```bash
.venv/bin/python eval/distributed/device_plan.py \
  --total-devices 96 \
  --model-key qwen3.5-9b \
  --node-rank 0 \
  --node-capacity 8 \
  --qwen-devices 32 \
  --step-devices 64
```

The output fields are:

```text
model_group_devices model_group_nodes local_devices qwen_devices step_devices
```

### Launch each model group

Assign nodes to the Qwen and Step groups. `NODE_RANK` starts at zero separately
inside each group. For example, a 24-device group on 8-NPU nodes uses ranks
`0, 1, 2`.

Set these values on every participating node:

```bash
export TOTAL_DEVICES=48
export NODE_DEVICE_CAPACITY=8
export NODE_RANK=0                 # unique rank within this model group
export NODE_ID="$(hostname -s)"   # unique across the shared run

export OUTPUT_ROOT=/shared/docvlm_eval
export DATASET_ROOT=/same/absolute/path/on/every/node/dataset
export PYTHON=/absolute/path/to/python
export OPENAI_API_KEY=dummy

export MAX_IMAGES=all
export MAX_TOKENS=256
export MAX_MODEL_LEN=131072         # use only if supported by the model/protocol
export LIMIT_MM_PER_PROMPT='{"image": 256}'

export REQUEST_CONCURRENCY=2
export MAX_NUM_SEQS=2
export MM_PROCESSOR_CACHE_GB=2
export ENABLE_PREFIX_CACHING=1
```

On the Qwen nodes:

```bash
export MODEL_KEY=qwen3.5-9b
bash eval/distributed/run_ascend_suite.sh
```

On the Step nodes:

```bash
export MODEL_KEY=step3vl-10b
bash eval/distributed/run_ascend_suite.sh
```

The default suite runs:

```text
mmlongbench-doc:val
longdocurl:val
slidevqa:val
```

Override it when necessary:

```bash
export BENCHMARKS='mmlongbench-doc:val longdocurl:val slidevqa:test'
```

Do not set one shared `RUN_NAME` for the full suite: each model/dataset pair
requires its own queue and output directory.

### What the queue does

For each model/dataset run:

1. group questions by source document;
2. order tasks by estimated image-prefill cost, largest first;
3. let every replica atomically claim the next task;
4. warm the shared document prefix before concurrent sibling questions;
5. checkpoint every successful response immediately;
6. retry failed requests and recover tasks whose worker lease expired;
7. merge worker JSONL files, remove duplicate IDs, and score the run.

The request is structured as `[shared instruction, document images, question]`
so repeated questions can reuse vLLM prefix and multimodal preprocessing caches.

### Monitor a live queue

```bash
RUN_DIR=/shared/docvlm_eval/Qwen3.5-9B-mmlongbench-doc-val-imgall-tok256

$PYTHON eval/distributed/queue_status.py \
  --queue-dir "$RUN_DIR/queue" \
  --shard-dir "$RUN_DIR/shards"
```

The status reports pending, active, completed, failed, and stale tasks plus the
number of result rows written so far.

### Rebalance after one model finishes

New workers may join an existing queue without rebuilding shards. Stop the
finished model's servers, assign the released nodes new, non-overlapping ranks
inside the expanded model group, and point them at the active dataset queue.

For example, to give all 96 devices to Qwen:

```bash
export TOTAL_DEVICES=96
export QWEN_DEVICES=96
export STEP_DEVICES=0

export MODEL_KEY=qwen3.5-9b
export DATASET=mmlongbench-doc
export SPLIT=val
export NODE_RANK=6                 # first new rank after the original ranks 0..5
export RUN_NAME=Qwen3.5-9B-mmlongbench-doc-val-imgall-tok256
export QUEUE_DIR="$OUTPUT_ROOT/$RUN_NAME/queue"

bash eval/distributed/run_ascend_queue_eval.sh
```

Use the next rank on each additional node. Before switching models, stop the old
servers with `eval/distributed/stop_vllm_servers.sh` while its old `MODEL_KEY`
and local device count are still set.

## 5. Run on One GPU Node

The legacy fixed-shard path remains useful for a small pilot:

```bash
export MODEL_KEY=step3vl-10b
export DATASET=longdocurl
export SPLIT=val
export GPUS_PER_NODE=8
export TOTAL_SHARDS=8
export MAX_IMAGES=8
export MAX_TOKENS=256
export REQUEST_CONCURRENCY=4
export MAX_NUM_SEQS=4
export PYTHON="$PWD/.venv/bin/python"
export OPENAI_API_KEY=dummy

bash eval/distributed/run_local_8gpu_eval.sh
```

This path launches one server per GPU, runs one contiguous shard per server,
merges predictions, and scores the run.

## 6. Run Against an Existing Endpoint

```bash
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:8000/v1 \
.venv/bin/python eval/run_benchmarks.py \
  --dataset mmlongbench-doc \
  --dataset-root dataset \
  --model Step3VL10B \
  --max-images 8 \
  --max-tokens 256 \
  --request-concurrency 4 \
  --request-timeout 1800 \
  --max-retries 3 \
  --resume \
  --output outputs/benchmarks/mmlongbench-doc.pred.jsonl
```

Use `--limit 20` for a smoke test. For SlideVQA, select `--split val` or
`--split test`.

## 7. Score Saved Predictions Manually

The distributed wrappers score automatically. To score a standalone prediction
file:

```bash
.venv/bin/python eval/score_benchmarks.py \
  --dataset mmlongbench-doc \
  --predictions outputs/benchmarks/mmlongbench-doc.pred.jsonl \
  --output outputs/benchmarks/mmlongbench-doc.scored.jsonl \
  --summary outputs/benchmarks/mmlongbench-doc.metrics.json
```

Each distributed run writes:

```text
outputs/distributed/<run-name>/
  queue/                 # queue mode only
  shards/                # resumable per-worker predictions
  logs/
  predictions.jsonl      # merged predictions
  scored.jsonl
  metrics.json
```

## Important Tuning Knobs

- `MAX_IMAGES=all`: use every document image; use a small integer only for pilots.
- `MAX_TOKENS`: output-token cap. Short-answer evaluation usually needs 128–256.
- `MAX_MODEL_LEN`: maximum model context; increase only when the model supports it.
- `REQUEST_CONCURRENCY`: requests submitted concurrently to each local endpoint.
- `MAX_NUM_SEQS`: vLLM server-side sequence capacity; keep it at least as large as request concurrency when memory allows.
- `MAX_NUM_BATCHED_TOKENS`: optional vLLM scheduler token budget.
- `MAX_RECORDS_PER_TASK`: document affinity versus load-balancing granularity; default `4`.
- `IMAGE_DATA_CACHE_MB`: per-worker encoded-image cache; default `256` MB.
- `MM_PROCESSOR_CACHE_GB`: vLLM multimodal processor cache per instance.
- `LEASE_SECONDS`: time before an un-heartbeated task is considered abandoned; default one hour.
- `VLLM_EXTRA_ARGS`: extra arguments appended to `vllm serve`.
- `SCORE_EXTRA_ARGS`: extra arguments passed to the scorer.

Start full-document pilots with `REQUEST_CONCURRENCY=1, 2, 4` and matching
`MAX_NUM_SEQS`, then choose the setting with the highest completed
samples/hour/NPU rather than the lowest single-request latency.

More detailed operational notes are in
[`eval/distributed/README.md`](eval/distributed/README.md), and benchmark/scoring
details are in [`eval/README_benchmarks.md`](eval/README_benchmarks.md).
