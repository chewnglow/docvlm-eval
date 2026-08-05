#!/usr/bin/env python3
"""Count records without materializing image payloads."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmlongbench-doc", "longdocurl", "slidevqa"], required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-cache", default="outputs/benchmark_image_cache")
    return parser.parse_args()


def count_dataset_records(dataset: str, dataset_root: Path, split: str = "val") -> int:
    if dataset == "longdocurl":
        return sum(
            1
            for line in (dataset_root / "longdocurl" / "LongDocURL_public_with_subtask_category.jsonl").open()
            if line.strip()
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'pyarrow'. Install eval/requirements-benchmark.txt first."
        ) from exc
    if dataset == "mmlongbench-doc":
        files = sorted((dataset_root / "mmlongbench-doc" / "data").glob("train-*.parquet"))
    elif dataset == "slidevqa":
        files = sorted((dataset_root / "slidevqa" / "data").glob(f"{split}-*.parquet"))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def main() -> None:
    args = parse_args()
    try:
        count = count_dataset_records(args.dataset, Path(args.dataset_root), args.split)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(count)


if __name__ == "__main__":
    main()
