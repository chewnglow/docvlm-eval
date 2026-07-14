#!/usr/bin/env python3
"""Run document VQA benchmark generation with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from benchmark_common import append_jsonl, image_to_data_url, iter_dataset_records


_THREAD_LOCAL = threading.local()


class ImageDataCache:
    """Small process-local, byte-bounded cache for repeated document images."""

    def __init__(self, capacity_mb: int) -> None:
        self.capacity = max(0, capacity_mb) * 1024 * 1024
        self.size = 0
        self.values: OrderedDict[str, str] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, path: Path) -> str:
        key = str(path.resolve())
        if not self.capacity:
            return image_to_data_url(path)
        with self.lock:
            value = self.values.pop(key, None)
            if value is not None:
                self.values[key] = value
                return value
            value = image_to_data_url(path)
            value_size = len(value)
            if value_size <= self.capacity:
                while self.values and self.size + value_size > self.capacity:
                    _, evicted = self.values.popitem(last=False)
                    self.size -= len(evicted)
                self.values[key] = value
                self.size += value_size
            return value


_IMAGE_CACHES: dict[int, ImageDataCache] = {}
_IMAGE_CACHES_LOCK = threading.Lock()


def get_image_cache(capacity_mb: int) -> ImageDataCache:
    with _IMAGE_CACHES_LOCK:
        cache = _IMAGE_CACHES.get(capacity_mb)
        if cache is None:
            cache = ImageDataCache(capacity_mb)
            _IMAGE_CACHES[capacity_mb] = cache
        return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmlongbench-doc", "longdocurl", "slidevqa"], required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", default="val", help="Only used by SlideVQA; MMLongBench uses train.")
    parser.add_argument("--output", required=True, help="Prediction JSONL path.")
    parser.add_argument("--model", required=True, help="OpenAI-compatible model name.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=int(os.environ.get("BENCHMARK_REQUEST_CONCURRENCY", "4")),
        help="Concurrent OpenAI requests. vLLM batches these server-side when --max-num-seqs allows it.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.environ.get("BENCHMARK_REQUEST_TIMEOUT", "1800")),
        help="Timeout in seconds for one long multimodal request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("BENCHMARK_MAX_RETRIES", "3")),
        help="Retries for transient request/server failures.",
    )
    parser.add_argument(
        "--image-data-cache-mb",
        type=int,
        default=int(os.environ.get("BENCHMARK_IMAGE_DATA_CACHE_MB", "256")),
        help="Process-local byte budget for base64-encoded repeated images; 0 disables it.",
    )
    parser.add_argument("--image-cache", default="outputs/benchmark_image_cache")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_instruction(record: dict[str, Any]) -> str:
    dataset = record["dataset"]
    if dataset == "slidevqa":
        return (
            "You are answering a question about a slide deck. Images are ordered by page, "
            "starting from page 1. Answer from the slides only."
        )
    if dataset == "longdocurl":
        return (
            "You are an expert in visual document question answering. Answer the question "
            "using only the provided document page images. Give a concise final answer."
        )
    return (
        "You are an expert in visual long-document question answering. Answer the question "
        "using only the provided document page images. Give a concise final answer."
    )


def make_question_suffix(record: dict[str, Any]) -> str:
    suffix = f"Question: {record['question']}\n\n"
    if record["dataset"] == "slidevqa":
        return (
            suffix
            + "End your response with exactly these two lines:\n"
            + "Final answer: <short answer>\n"
            + "Evidence pages: [page numbers used]"
        )
    return suffix + "End your response with one line: Final answer: <short answer>"


def make_prompt(record: dict[str, Any]) -> str:
    """Text-only representation kept for callers that display the prompt."""
    return f"{make_instruction(record)}\n\n{make_question_suffix(record)}"


def get_openai_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'openai'. Install eval/requirements-benchmark.txt first."
        ) from exc
    client = getattr(_THREAD_LOCAL, "openai_client", None)
    if client is None:
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.request_timeout,
            max_retries=0,
        )
        _THREAD_LOCAL.openai_client = client
    return client


def call_openai(args: argparse.Namespace, record: dict[str, Any]) -> str:
    client = get_openai_client(args)
    image_paths = [Path(p) for p in record.get("image_paths", []) if Path(p).exists()]
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    # Put the varying question after the document. Requests for the same document
    # then share an identical long prefix, which is required for vLLM APC reuse.
    content: list[dict[str, Any]] = [{"type": "text", "text": make_instruction(record)}]
    image_cache = get_image_cache(args.image_data_cache_mb)
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_cache.get(path)}})
    content.append({"type": "text", "text": make_question_suffix(record)})
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return response.choices[0].message.content or ""


def run_record(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            response = call_openai(args, record)
            break
        except Exception:
            _THREAD_LOCAL.openai_client = None
            if attempts > args.max_retries:
                raise
            time.sleep(min(30.0, 2.0 ** (attempts - 1)))
    row = {k: v for k, v in record.items() if k != "image_paths"}
    image_count = len(record.get("image_paths", []))
    if args.max_images is not None:
        image_count = min(image_count, args.max_images)
    row["num_images"] = image_count
    row["response"] = response
    row["request_seconds"] = round(time.monotonic() - started, 3)
    row["request_attempts"] = attempts
    return row


def load_done_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    done: set[str] = set()
    with output.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A process may have died during its final append. The queue can
                # safely regenerate that record and merge will ignore this fragment.
                continue
            done.add(str(row["id"]))
    return done


def run_records(
    args: argparse.Namespace,
    records: Any,
    output: Path,
    done: set[str] | None = None,
) -> set[str]:
    """Evaluate records with bounded concurrency and append each success immediately."""
    done = done if done is not None else set()
    submitted = 0
    completed = 0
    started = time.monotonic()
    in_flight: dict[Future[dict[str, Any]], dict[str, Any]] = {}

    def finish(future: Future[dict[str, Any]]) -> None:
        nonlocal completed
        record = in_flight.pop(future)
        row = future.result()
        append_jsonl(output, row)
        record_id = str(record["id"])
        done.add(record_id)
        completed += 1
        hours = max((time.monotonic() - started) / 3600.0, 1e-9)
        print(
            f"[{completed}/{submitted}] {record['dataset']} {record_id} "
            f"({completed / hours:.2f} samples/hour)",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=args.request_concurrency) as executor:
        for record in records:
            if str(record["id"]) in done:
                continue
            future = executor.submit(run_record, args, record)
            in_flight[future] = record
            submitted += 1
            if len(in_flight) >= args.request_concurrency:
                done_futures, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    finish(future)

        while in_flight:
            done_futures, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done_futures:
                finish(future)
    return done


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")
    if args.request_concurrency < 1:
        raise SystemExit("--request-concurrency must be >= 1.")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0.")
    if args.image_data_cache_mb < 0:
        raise SystemExit("--image-data-cache-mb must be >= 0.")
    output = Path(args.output)
    done = load_done_ids(output) if args.resume else set()
    records = iter_dataset_records(
        args.dataset,
        Path(args.dataset_root),
        split=args.split,
        image_cache=Path(args.image_cache),
    )
    selected = 0
    seen = 0
    selected_records = []
    for record in records:
        if seen < args.offset:
            seen += 1
            continue
        seen += 1
        if str(record["id"]) in done:
            continue
        if args.limit is not None and selected >= args.limit:
            break
        selected_records.append(record)
        selected += 1
    run_records(args, selected_records, output, done)


if __name__ == "__main__":
    main()
