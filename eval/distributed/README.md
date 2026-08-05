# Distributed GPU/NPU Evaluation

This folder wraps the existing benchmark code in `eval/run_benchmarks.py` and
`eval/score_benchmarks.py` for multi-GPU and multi-node runs.

Two execution modes are available:

- fixed contiguous shards, retained for small single-node GPU runs;
- a shared pull queue for large multi-node Ascend runs.

The queue mode is the production path for 16–128 devices. It launches one
complete TP=1 model replica per NPU, keeps every worker pulling work until the
global queue is empty, orders image-heavy work first, bundles questions from the
same document, retries transient failures, and recovers abandoned tasks by
lease.

For a single web or CLI submission to an existing Ray cluster, use
`ray_orchestrator.py`. Ray handles node discovery, resource reservation, and
remote process launch; the filesystem queue continues to handle fine-grained
evaluation scheduling.

## Prerequisites

All nodes should have the same project path, Python dependencies, dataset files,
and model files:

```bash
uv pip install --python .venv/bin/python -r eval/requirements-benchmark.txt
```

The shell wrappers default to `${PWD}/.venv/bin/python`. If this checkout uses
another environment, export it before running the distributed scripts:

```bash
export PYTHON=/abs/path/to/python
```

Install a vLLM version that supports the target models. Qwen3.5 and Step3VL both
require recent vLLM builds; follow each model README if the installed vLLM is too
old.

If you use multiple nodes, set `OUTPUT_ROOT` to a shared filesystem path visible
from all nodes. If you do not have shared storage, copy each node's `shards/`
folder to node 0 before scoring.

For queue mode, `OUTPUT_ROOT` must be shared and mounted at the same path. The
dataset may be staged on node-local storage for throughput, but its absolute path
must also be identical on every node because the queue manifest stores image
paths.

## Flexible 16–128 NPU Ascend Run

### Submit once with Ray

Start every Ascend compute node with a custom resource matching its device
capacity, for example `--resources='{"NPU": 8}'`. Then submit:

```bash
ray job submit \
  --address=http://<head-node-ip>:8265 \
  --working-dir . \
  -- \
  python eval/distributed/ray_orchestrator.py \
    --total-devices 128 \
    --node-capacity 8 \
    --output-root /shared/docvlm_eval \
    --dataset-root /data/docvlm/dataset \
    --python /opt/docvlm/bin/python \
    --qwen-model-path /models/Qwen3.5-9B \
    --qwen-base-model-path /models/Qwen3.5-9B-Base \
    --step-model-path /models/Step3VL10B \
    --step-base-model-path /models/Step3VL10B-base
```

The head node may be CPU-only. Only live nodes advertising the `NPU` resource
are selected. By default, the orchestrator divides the devices evenly among
Qwen3.5-9B, Qwen3.5-9B-Base, Step3VL-10B, and Step3VL-10B-Base. It assigns
non-overlapping device ranges to each model group, launches all four groups
concurrently, and cleans up their vLLM servers at the end. Groups smaller than
one node may safely share a physical node because their device visibility and
ports are offset. Use `--dry-run` to print the assignment without launching.

For a custom split, set all four values and make their sum equal
`--total-devices`:

```bash
--qwen-devices 24 \
--qwen-base-devices 24 \
--step-devices 40 \
--step-base-devices 40
```

The Ray working-directory upload excludes datasets, models, outputs, and virtual
environments through the repository's `.rayignore`. Data and models must be
mounted at consistent paths, and `OUTPUT_ROOT` must be shared by all nodes.

The remaining commands in this section are the original manual two-model
alternative.

Set `TOTAL_DEVICES` to any integer from 16 through 128. By default the planner
splits the devices evenly between the two models, giving the extra device to
Qwen when the total is odd:

```text
TOTAL_DEVICES=16   -> Qwen 8,  Step 8
TOTAL_DEVICES=48   -> Qwen 24, Step 24
TOTAL_DEVICES=96   -> Qwen 48, Step 48
TOTAL_DEVICES=128  -> Qwen 64, Step 64
```

You can choose a non-even allocation as long as it sums to `TOTAL_DEVICES`:

```bash
export TOTAL_DEVICES=96
export QWEN_DEVICES=32
export STEP_DEVICES=64
```

`NODE_RANK` is the rank within that model's node group, starting at zero.
`NODE_DEVICE_CAPACITY` is the maximum number of local NPUs the launcher may use.
The planner calculates the required number of nodes and automatically uses only
the remainder on the final node. For example, 20 devices allocated to one model
with 8-device nodes uses local counts `8, 8, 4`.

On every Qwen group node, run:

```bash
export MODEL_KEY=qwen3.5-9b
export TOTAL_DEVICES=48            # any value from 16 through 128
export NODE_RANK=0                 # 0..ceil(QWEN_DEVICES / capacity)-1
export NODE_ID="$(hostname -s)"   # must be unique across the shared run
export NODE_DEVICE_CAPACITY=8
export OUTPUT_ROOT=/shared/docvlm_eval
export DATASET_ROOT=/local-identical-path/dataset
export PYTHON=/path/to/python
export MAX_IMAGES=all              # do not silently truncate the benchmark
export MAX_TOKENS=256
export MAX_MODEL_LEN=131072         # only if this matches the chosen protocol/model
export LIMIT_MM_PER_PROMPT='{"image": 256}'
export REQUEST_CONCURRENCY=2
export MAX_NUM_SEQS=2
export MM_PROCESSOR_CACHE_GB=2
export ENABLE_PREFIX_CACHING=1
export OPENAI_API_KEY=dummy

bash eval/distributed/run_ascend_suite.sh
```

On the Step group nodes use the same command, but set:

```bash
export MODEL_KEY=step3vl-10b
export NODE_RANK=0                 # rank within the Step group
bash eval/distributed/run_ascend_suite.sh
```

Each group processes MMLongBench-Doc, LongDocURL, and SlideVQA validation in
sequence. Override the suite if needed, for example:

```bash
export BENCHMARKS='mmlongbench-doc:val longdocurl:val slidevqa:test'
```

`NODE_RANK=0` builds each queue and scores it. Other nodes wait for the queue's
`READY` marker, then every allocated worker pulls document bundles dynamically.
Do not set a common `RUN_NAME` for the entire suite; each dataset needs its own
queue.

Start conservatively at concurrency 2 for full, long image documents. Sweep
`REQUEST_CONCURRENCY=1,2,4` together with `MAX_NUM_SEQS` on a representative
pilot, then use the setting with the highest completed samples/hour/NPU. For
shorter capped-image pilots, 4 or 8 can be better.

Monitor the complete suite from any machine that can read the shared output root:

```bash
$PYTHON eval/distributed/progress_monitor.py \
  --output-root /shared/docvlm_eval \
  --watch 10
```

This is the primary monitoring interface. It shows overall progress and ETA, one
row for every model/benchmark sub-experiment (including benchmarks not started
yet), task and record progress, excluded failures, throughput, and a row for every
active NPU/GPU with HBM/VRAM occupation. Model groups are treated as concurrent;
benchmarks within one model group are treated as sequential when calculating the
overall ETA. Waiting-experiment ETAs use that model's measured rate and carry a
`~` prefix. ETA remains `--` while a model has not completed enough work to
measure a rate.

An experiment remaining without metrics for more than ten minutes after its
queue drains is marked `UNSCORED`, and the overall suite state changes to
`ATTENTION`.

The experiment table's `TASK S/A/P` column means settled, active, and pending
queue tasks; `EXCL` is the number of terminally failed records excluded from the
reported score.

Options:

- `--watch 10`: refresh in place every ten seconds;
- `--json`: emit one machine-readable snapshot;
- `--hide-devices`: show experiment and overall progress without the full device
  table;
- `--eta-window-minutes 30`: tune recent-throughput smoothing;
- `--memory-max-age 120`: control when a device sample is considered stale.

For a compact check of one experiment only:

```bash
RUN_DIR=/shared/docvlm_eval/Qwen3.5-9B-mmlongbench-doc-val-imgall-tok256
$PYTHON eval/distributed/queue_status.py \
  --queue-dir "$RUN_DIR/queue" \
  --shard-dir "$RUN_DIR/shards"
```

The second output line reports live device-memory occupation aggregated from all
nodes currently working on that run:

```text
device_memory=742400/1048576MB (70.8%) per_device=67.3-74.9% devices=16 nodes=2 backend=ascend
```

The launcher samples Ascend HBM with `npu-smi` or NVIDIA VRAM with `nvidia-smi`
every 30 seconds. Set `DEVICE_MEMORY_INTERVAL=10` to change the interval or
`DEVICE_MEMORY_MONITOR=0` to disable it. Ray forwards both settings. Per-node
current snapshots and JSONL history are stored under
`RUN_DIR/device_memory/{current,history}/`, and readable monitor logs are under
`RUN_DIR/logs/<node>/device-memory.log`. Monitoring failures never stop an
evaluation; `queue_status.py` reports unavailable data when vendor tooling is not
visible inside the container.

If one model group finishes first, stop its servers, point those nodes at the
unfinished model's current `RUN_NAME`/`QUEUE_DIR`, launch that model, and run
`run_queue_workers.sh`. New workers can join a live queue; no resharing step is
required.

To give all devices to Qwen after Step finishes, for example:

```bash
export TOTAL_DEVICES=96 QWEN_DEVICES=96 STEP_DEVICES=0
```

The prompt runner sends `[shared instruction, document images, question]` so
questions from one document share the longest possible prefix. The local client
also uses a byte-bounded encoded-image cache. Server-side prefix caching and the
multimodal processor cache are enabled by the launcher and remain tunable via
`ENABLE_PREFIX_CACHING` and `MM_PROCESSOR_CACHE_GB`.

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
export REQUEST_CONCURRENCY=4
export MAX_MODEL_LEN=16384
export MAX_NUM_SEQS=4
export PYTHON="$PWD/.venv/bin/python"
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
  failed_records.jsonl
```

Requests and tasks are retried according to `MAX_RETRIES` and `MAX_TASK_ATTEMPTS`.
After those retries are exhausted, terminally failed records remain under
`queue/failed`, are copied to `failed_records.jsonl` with their final error, and are
excluded from scoring. They do not stop the suite from advancing to the next benchmark.
Check the `coverage` object in `metrics.json`: `status: partial` means the reported
metric covers only the successfully evaluated records. Scoring still fails for an empty
queue or for missing records that have no corresponding terminal failure entry.

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
export REQUEST_CONCURRENCY=4
export MAX_NUM_SEQS=4
export PYTHON=/abs/path/to/python
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
- `MAX_IMAGES=all`: production queue mode with every raw document image.
- `MAX_TOKENS`: use `128` or `256` for short-answer benchmarking.
- `MAX_MODEL_LEN`: vLLM max input/context length. Default is `16384`; raise only when needed.
- `REQUEST_CONCURRENCY`: concurrent requests sent by each shard process to its local vLLM server. Default is `4`.
- `MAX_NUM_SEQS`: vLLM server-side sequence batch capacity. Default is `4`; keep
  it at least as large as `REQUEST_CONCURRENCY` when memory allows.
- `DEVICE_MEMORY_MONITOR`: set to `1` (default) to sample live HBM/VRAM usage, or `0` to disable it.
- `DEVICE_MEMORY_INTERVAL`: seconds between device-memory samples. Default is `30`.
- `PYTHON`: Python executable for helper scripts. Default is `${repo}/.venv/bin/python`.
- `LIMIT_MM_PER_PROMPT`: should be >= `MAX_IMAGES`.
- `TOTAL_SHARDS`: normally total GPUs across all nodes.
- `BASE_PORT`: local port for GPU 0; GPU k uses `BASE_PORT + k`.
- `VLLM_EXTRA_ARGS`: append extra vLLM flags without editing scripts.
- `SCORE_EXTRA_ARGS`: append scoring flags, e.g. answer extraction via an API.
- `DEVICE_TYPE`: `cuda` (default) or `ascend`; Ascend uses `ASCEND_RT_VISIBLE_DEVICES`.
- `TOTAL_DEVICES`: total active fleet size, validated in `[16, 128]`.
- Ray defaults to four equal model allocations. Override them with
  `--qwen-devices`, `--qwen-base-devices`, `--step-devices`, and
  `--step-base-devices`; their sum must equal `TOTAL_DEVICES`.
- Manual mode retains `QWEN_DEVICES` and `STEP_DEVICES` as its two model-family
  allocations.
- `NODE_DEVICE_CAPACITY`: maximum devices available on one node, normally `8`.
- `DEVICES_PER_NODE`: computed active local count in queue mode; still accepted directly by lower-level launchers.
- `MAX_RECORDS_PER_TASK`: document-affinity/straggler tradeoff in queue mode; default `4`.
- `LEASE_SECONDS`: time before an un-heartbeated claimed task is recoverable; default one hour.
- `IMAGE_DATA_CACHE_MB`: per-worker encoded-image cache; default `256` MB.

## Recommended Workflow

1. Pilot one model/dataset:

   ```bash
   export LIMIT_PER_SHARD=5 MAX_IMAGES=4 MAX_TOKENS=128 REQUEST_CONCURRENCY=2 MAX_NUM_SEQS=2
   bash eval/distributed/run_local_8gpu_eval.sh
   ```

2. Throughput run:

   ```bash
   export LIMIT_PER_SHARD=25 MAX_IMAGES=8 MAX_TOKENS=256 REQUEST_CONCURRENCY=4 MAX_NUM_SEQS=4
   bash eval/distributed/run_local_8gpu_eval.sh
   ```

3. Full capped-image run:

   ```bash
   unset LIMIT_PER_SHARD
   export MAX_IMAGES=8 MAX_TOKENS=256 REQUEST_CONCURRENCY=4 MAX_NUM_SEQS=4
   bash eval/distributed/run_local_8gpu_eval.sh
   ```

4. Only for the best model, consider larger `MAX_IMAGES` or full-image settings.
