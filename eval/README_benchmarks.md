# Benchmark Scripts

These scripts fetch/process datasets, run generation, and score saved
predictions.

Install runtime dependencies in the project venv before use:

```bash
uv pip install --python .venv/bin/python -r eval/requirements-benchmark.txt
```

## Fetch And Process Datasets

Run from the project root:

```bash
.venv/bin/python eval/fetch_datasets.py \
  --root dataset \
  --max-workers 4 \
  --render-workers 4
```

This downloads:

- `yubo2333/MMLongBench-Doc` into `dataset/mmlongbench-doc`
- `dengchao/LongDocURL` into `dataset/longdocurl`
- `NTT-hil-insight/SlideVQA` into `dataset/slidevqa`

For datasets with document files, PDFs are rendered into:

```text
dataset/<dataset-name>/document_images/<document-filename-without-suffix>/
```

LongDocURL already ships page images, so the script organizes/link-copies those
images into the same `document_images/<document-stem>/` layout. MMLongBench-Doc
has two upstream-broken PDF entries; the script skips those broken objects and
fetches fallback rendered images for them.

The script is resumable. Completed archive extraction and PDF rendering are
tracked with marker files, so rerunning the same command skips finished work.

Useful variants:

```bash
# Process only one dataset.
.venv/bin/python eval/fetch_datasets.py --root dataset --datasets longdocurl

# Process already-downloaded raw files under dataset/ without refreshing the
# primary Hugging Face snapshots.
.venv/bin/python eval/fetch_datasets.py --root dataset --process-only

# Download/unpack only; skip PDF rendering/image organization.
.venv/bin/python eval/fetch_datasets.py --root dataset --skip-render
```

## Run Generation

Run generation through an OpenAI-compatible multimodal endpoint:

```bash
OPENAI_API_KEY=... OPENAI_BASE_URL=http://localhost:8000/v1 \
.venv/bin/python eval/run_benchmarks.py \
  --dataset mmlongbench-doc \
  --dataset-root dataset \
  --model Step3VL10B \
  --request-concurrency 4 \
  --output outputs/benchmarks/mmlongbench-doc.pred.jsonl
```

`run_benchmarks.py` uses concurrent OpenAI-compatible requests rather than
direct `vllm.LLM.generate(prompts, ...)`. With vLLM serving, those concurrent
requests are batched by the vLLM server when its `--max-num-seqs` is greater
than 1. The distributed launcher defaults to `MAX_NUM_SEQS=4` and
`REQUEST_CONCURRENCY=4`.

For SlideVQA, choose `--split val` or `--split test`:

```bash
.venv/bin/python eval/run_benchmarks.py \
  --dataset slidevqa \
  --split val \
  --dataset-root dataset \
  --model Step3VL10B \
  --request-concurrency 4 \
  --output outputs/benchmarks/slidevqa-val.pred.jsonl
```

## Score Predictions

Score saved predictions:

```bash
.venv/bin/python eval/score_benchmarks.py \
  --dataset mmlongbench-doc \
  --predictions outputs/benchmarks/mmlongbench-doc.pred.jsonl \
  --output outputs/benchmarks/mmlongbench-doc.scored.jsonl \
  --summary outputs/benchmarks/mmlongbench-doc.metrics.json
```

If model responses are long free-form answers, add `--extract-model gpt-4o` to
use the official MMLongBench/LongDocURL-style answer-extraction stage before
rule-based scoring.
