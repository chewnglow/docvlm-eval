#!/usr/bin/env python3
"""Run document VQA benchmark generation with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from benchmark_common import append_jsonl, image_to_data_url, iter_dataset_records


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
    parser.add_argument("--image-cache", default="outputs/benchmark_image_cache")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_prompt(record: dict[str, Any]) -> str:
    dataset = record["dataset"]
    if dataset == "slidevqa":
        return (
            "You are answering a question about a slide deck. Images are ordered by page, "
            "starting from page 1. Answer from the slides only.\n\n"
            f"Question: {record['question']}\n\n"
            "End your response with exactly these two lines:\n"
            "Final answer: <short answer>\n"
            "Evidence pages: [page numbers used]"
        )
    if dataset == "longdocurl":
        return (
            "You are an expert in visual document question answering. Answer the question "
            "using only the provided document page images. Give a concise final answer.\n\n"
            f"Question: {record['question']}\n\n"
            "End your response with one line: Final answer: <short answer>"
        )
    return (
        "You are an expert in visual long-document question answering. Answer the question "
        "using only the provided document page images. Give a concise final answer.\n\n"
        f"Question: {record['question']}\n\n"
        "End your response with one line: Final answer: <short answer>"
    )


def call_openai(args: argparse.Namespace, record: dict[str, Any]) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'openai'. Install eval/requirements-benchmark.txt first."
        ) from exc
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    image_paths = [Path(p) for p in record.get("image_paths", []) if Path(p).exists()]
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    content: list[dict[str, Any]] = [{"type": "text", "text": make_prompt(record)}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return response.choices[0].message.content or ""


def load_done_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    done: set[str] = set()
    with output.open() as f:
        for line in f:
            if not line.strip():
                continue
            import json

            row = json.loads(line)
            done.add(str(row["id"]))
    return done


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")
    output = Path(args.output)
    done = load_done_ids(output) if args.resume else set()
    records = iter_dataset_records(
        args.dataset,
        Path(args.dataset_root),
        split=args.split,
        image_cache=Path(args.image_cache),
    )
    emitted = 0
    seen = 0
    for record in records:
        if seen < args.offset:
            seen += 1
            continue
        seen += 1
        if str(record["id"]) in done:
            continue
        if args.limit is not None and emitted >= args.limit:
            break
        response = call_openai(args, record)
        row = {k: v for k, v in record.items() if k != "image_paths"}
        row["num_images"] = len(record.get("image_paths", []))
        row["response"] = response
        append_jsonl(output, row)
        emitted += 1
        print(f"[{emitted}] {record['dataset']} {record['id']}", flush=True)


if __name__ == "__main__":
    main()
