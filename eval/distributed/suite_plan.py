#!/usr/bin/env python3
"""Declare a model's sequential benchmark experiments for suite monitoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from count_records import count_dataset_records


DEFAULT_BENCHMARKS = "mmlongbench-doc:val longdocurl:val slidevqa:val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--max-images", default="all")
    parser.add_argument("--max-tokens", default="256")
    parser.add_argument("--device-count", type=int, default=0)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "model"


def parse_benchmarks(value: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in value.split():
        dataset, separator, split = item.partition(":")
        if not separator or not dataset or not split:
            raise ValueError(f"Invalid benchmark specification {item!r}; expected DATASET:SPLIT")
        if dataset not in {"mmlongbench-doc", "longdocurl", "slidevqa"}:
            raise ValueError(f"Unknown benchmark dataset: {dataset}")
        specs.append((dataset, split))
    if not specs:
        raise ValueError("At least one benchmark specification is required")
    return specs


def build_plan(
    *,
    output_root: Path,
    dataset_root: Path,
    model_key: str,
    served_model_name: str,
    benchmarks: str,
    max_images: str,
    max_tokens: str,
    device_count: int,
) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    for sequence, (dataset, split) in enumerate(parse_benchmarks(benchmarks)):
        experiment: dict[str, Any] = {
            "sequence": sequence,
            "dataset": dataset,
            "split": split,
            "run_name": f"{served_model_name}-{dataset}-{split}-img{max_images}-tok{max_tokens}",
        }
        try:
            count = count_dataset_records(dataset, dataset_root, split)
            experiment["expected_count"] = count
            if count == 0:
                experiment["count_error"] = "No dataset records found"
        except Exception as exc:
            experiment["expected_count"] = None
            experiment["count_error"] = f"{type(exc).__name__}: {exc}"
        experiments.append(experiment)
    return {
        "version": 1,
        "created_at": time.time(),
        "output_root": str(output_root.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "model_key": model_key,
        "served_model_name": served_model_name,
        "device_count": device_count,
        "sequential": True,
        "experiments": experiments,
    }


def write_plan(output_root: Path, plan: dict[str, Any]) -> Path:
    plan_dir = output_root / "_suite_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    path = plan_dir / f"{safe_name(str(plan['model_key']))}.json"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    return path


def main() -> None:
    args = parse_args()
    try:
        plan = build_plan(
            output_root=Path(args.output_root),
            dataset_root=Path(args.dataset_root),
            model_key=args.model_key,
            served_model_name=args.served_model_name,
            benchmarks=args.benchmarks,
            max_images=args.max_images,
            max_tokens=args.max_tokens,
            device_count=args.device_count,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    path = write_plan(Path(args.output_root), plan)
    print(f"Suite plan: {path} ({len(plan['experiments'])} experiments)", flush=True)


if __name__ == "__main__":
    main()
