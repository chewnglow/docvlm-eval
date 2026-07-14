#!/usr/bin/env python3
"""Materialize a document-aware, longest-first work queue for many replicas."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from benchmark_common import iter_dataset_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmlongbench-doc", "longdocurl", "slidevqa"], required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-cache", default="outputs/benchmark_image_cache")
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--max-records-per-task",
        type=int,
        default=4,
        help="Questions from one document kept together in each queue task.",
    )
    parser.add_argument("--reset", action="store_true", help="Replace an existing queue.")
    return parser.parse_args()


def document_key(record: dict[str, Any]) -> str:
    if record["dataset"] == "longdocurl":
        return f"longdocurl:{record.get('doc_no', record['id'])}"
    if record["dataset"] == "slidevqa":
        deck = record.get("deck_name") or record.get("deck_url") or record["id"]
        return f"slidevqa:{deck}"
    return f"mmlongbench-doc:{record.get('doc_id', record['id'])}"


def capped_image_count(record: dict[str, Any], max_images: int | None) -> int:
    count = len(record.get("image_paths", []))
    return min(count, max_images) if max_images is not None else count


def expected_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": 1,
        "dataset": args.dataset,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": args.split,
        "image_cache": str(Path(args.image_cache).resolve()),
        "max_images": args.max_images,
        "max_records_per_task": args.max_records_per_task,
    }


def validate_existing(queue_dir: Path, expected: dict[str, Any]) -> bool:
    manifest_path = queue_dir / "manifest.json"
    ready = queue_dir / "READY"
    if not (manifest_path.exists() and ready.exists()):
        return False
    actual = json.loads(manifest_path.read_text())
    for key, value in expected.items():
        if actual.get(key) != value:
            raise SystemExit(
                f"Queue {queue_dir} already exists with different {key}: "
                f"{actual.get(key)!r} != {value!r}. Use a new RUN_NAME or --reset."
            )
    print(f"Queue is already ready: {queue_dir} ({actual['record_count']} records)")
    return True


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_records_per_task < 1:
        raise SystemExit("--max-records-per-task must be >= 1")
    if args.max_images is not None and args.max_images < 1:
        raise SystemExit("--max-images must be >= 1 when provided")

    queue_dir = Path(args.queue_dir)
    expected = expected_manifest(args)
    if queue_dir.exists():
        if args.reset:
            shutil.rmtree(queue_dir)
        elif validate_existing(queue_dir, expected):
            return
        else:
            raise SystemExit(f"Incomplete queue already exists: {queue_dir}. Use --reset after checking it.")

    queue_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = queue_dir.parent / f".{queue_dir.name}.building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    for name in ("pending", "claimed", "completed", "failed"):
        (staging / name).mkdir()

    records = list(
        iter_dataset_records(
            args.dataset,
            Path(args.dataset_root),
            split=args.split,
            image_cache=Path(args.image_cache),
        )
    )
    seen: set[str] = set()
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (staging / "records.jsonl").open("w") as out:
        for record in records:
            record_id = str(record["id"])
            if record_id in seen:
                raise SystemExit(f"Duplicate record id in {args.dataset}: {record_id}")
            seen.add(record_id)
            record["id"] = record_id
            by_document[document_key(record)].append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    tasks: list[dict[str, Any]] = []
    for doc_key, doc_records in by_document.items():
        for start in range(0, len(doc_records), args.max_records_per_task):
            chunk = doc_records[start : start + args.max_records_per_task]
            image_counts = [capped_image_count(record, args.max_images) for record in chunk]
            tasks.append(
                {
                    "document_key": doc_key,
                    "record_ids": [str(record["id"]) for record in chunk],
                    "estimated_cost": sum(image_counts),
                    "max_images_in_record": max(image_counts, default=0),
                    "attempts": 0,
                }
            )

    tasks.sort(key=lambda task: (-task["estimated_cost"], -task["max_images_in_record"], task["document_key"]))
    for index, task in enumerate(tasks):
        task["task_id"] = f"task-{index:06d}"
        write_json(staging / "pending" / f"{task['task_id']}.json", task)

    manifest = {
        **expected,
        "record_count": len(records),
        "document_count": len(by_document),
        "task_count": len(tasks),
    }
    write_json(staging / "manifest.json", manifest)
    (staging / "READY").write_text("ready\n")
    os.rename(staging, queue_dir)
    print(
        f"Built {queue_dir}: {len(records)} records, {len(by_document)} documents, "
        f"{len(tasks)} longest-first tasks"
    )


if __name__ == "__main__":
    main()
