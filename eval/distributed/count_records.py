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


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if args.dataset == "longdocurl":
        count = sum(
            1
            for line in (dataset_root / "longdocurl" / "LongDocURL_public_with_subtask_category.jsonl").open()
            if line.strip()
        )
    else:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency 'pyarrow'. Install eval/requirements-benchmark.txt first."
            ) from exc
        if args.dataset == "mmlongbench-doc":
            files = sorted((dataset_root / "mmlongbench-doc" / "data").glob("train-*.parquet"))
        else:
            files = sorted((dataset_root / "slidevqa" / "data").glob(f"{args.split}-*.parquet"))
        count = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    print(count)


if __name__ == "__main__":
    main()
