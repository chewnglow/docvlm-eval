#!/usr/bin/env python3
"""Print a compact status summary for a shared evaluation queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--lease-seconds", type=float, default=3600)
    return parser.parse_args()


def count_rows(folder: Path | None) -> int:
    if folder is None or not folder.exists():
        return 0
    count = 0
    for path in folder.glob("*.jsonl"):
        with path.open() as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def main() -> None:
    args = parse_args()
    queue_dir = Path(args.queue_dir)
    manifest = json.loads((queue_dir / "manifest.json").read_text())
    counts = {
        state: len(list((queue_dir / state).glob("task-*.json")))
        for state in ("pending", "claimed", "completed", "failed")
    }
    stale = sum(
        1
        for path in (queue_dir / "claimed").glob("task-*.json")
        if time.time() - path.stat().st_mtime > args.lease_seconds
    )
    result_rows = count_rows(Path(args.shard_dir) if args.shard_dir else None)
    print(
        f"{manifest['dataset']} records={manifest['record_count']} tasks={manifest['task_count']} "
        f"pending={counts['pending']} active={counts['claimed']} complete={counts['completed']} "
        f"failed={counts['failed']} stale={stale} result_rows={result_rows}"
    )


if __name__ == "__main__":
    main()
