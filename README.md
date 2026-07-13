## Document VQA Benchmark Workflow

This repository includes local benchmark scripts for MMLongBench-Doc,
LongDocURL, and SlideVQA under `eval/`. The expected end-to-end order is:

1. install benchmark dependencies;
2. download and process datasets into `dataset/`;
3. serve one model through a vLLM OpenAI-compatible endpoint;
4. run generation;
5. score saved predictions.

### 1. Install Benchmark Dependencies

Run from the project root:

```bash
uv pip install --python .venv/bin/python -r eval/requirements-benchmark.txt
```

### 2. Download And Process Datasets

The fetch script downloads the raw datasets, extracts archives, renders PDF
documents to page images, and normalizes document images into the layout used by
the benchmark runner:

```bash
.venv/bin/python eval/fetch_datasets.py \
  --root dataset \
  --max-workers 4 \
  --render-workers 4
```

The script prepares:

- `dataset/mmlongbench-doc` from `yubo2333/MMLongBench-Doc`
- `dataset/longdocurl` from `dengchao/LongDocURL`
- `dataset/slidevqa` from `NTT-hil-insight/SlideVQA`

For datasets containing document files, rendered pages are stored as:

```text
dataset/<dataset-name>/document_images/<document-filename-without-suffix>/
```

The command is resumable. Archive extraction and PDF rendering use completion
markers, so rerunning it skips finished work. If the raw files are already
downloaded under `dataset/`, process them without refreshing Hugging Face
snapshots:

```bash
.venv/bin/python eval/fetch_datasets.py --root dataset --process-only
```

To process just one dataset:

```bash
.venv/bin/python eval/fetch_datasets.py --root dataset --datasets longdocurl --process-only
```

### 3. Serve A Model Locally

Use the distributed wrappers for vLLM serving. They default to
`MAX_MODEL_LEN=16384`.

Single node with 8 GPUs:

```bash
export MODEL_KEY=step3vl-10b
export DATASET=longdocurl
export SPLIT=val
export GPUS_PER_NODE=8
export TOTAL_SHARDS=8
export MAX_IMAGES=8
export MAX_TOKENS=256
export OPENAI_API_KEY=dummy

bash eval/distributed/run_local_8gpu_eval.sh
```

For multi-node 8-GPU runs, or if model paths differ from the defaults, see
`eval/distributed/README.md`.

### 4. Run A Single-Endpoint Evaluation

If you already have a model server running:

```bash
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:8000/v1 \
.venv/bin/python eval/run_benchmarks.py \
  --dataset mmlongbench-doc \
  --dataset-root dataset \
  --model Step3VL10B \
  --max-images 8 \
  --max-tokens 256 \
  --output outputs/benchmarks/mmlongbench-doc.pred.jsonl
```

For quick pilots, add `--limit 20`. For SlideVQA, pass `--split val` or
`--split test`.

### 5. Score Predictions

```bash
.venv/bin/python eval/score_benchmarks.py \
  --dataset mmlongbench-doc \
  --predictions outputs/benchmarks/mmlongbench-doc.pred.jsonl \
  --output outputs/benchmarks/mmlongbench-doc.scored.jsonl \
  --summary outputs/benchmarks/mmlongbench-doc.metrics.json
```

More details are in `eval/README_benchmarks.md`.
