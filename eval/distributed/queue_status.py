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
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Device-memory directory; defaults to <queue parent>/device_memory.",
    )
    parser.add_argument(
        "--memory-max-age",
        type=float,
        default=120.0,
        help="Ignore memory snapshots older than this many seconds.",
    )
    return parser.parse_args()


def count_rows(folder: Path | None) -> int:
    if folder is None or not folder.exists():
        return 0
    count = 0
    for path in folder.glob("*.jsonl"):
        with path.open() as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def memory_status(memory_dir: Path, max_age: float, now: float | None = None) -> str:
    now = time.time() if now is None else now
    snapshots: list[dict] = []
    current_dir = memory_dir / "current"
    for path in sorted(current_dir.glob("*.json")):
        try:
            snapshot = json.loads(path.read_text())
            age = now - float(snapshot["timestamp"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if snapshot.get("active") and age <= max_age:
            snapshot["age_seconds"] = age
            snapshots.append(snapshot)
    if not snapshots:
        return "device_memory=unavailable (no fresh active snapshots)"

    devices = [device for snapshot in snapshots for device in snapshot.get("devices", [])]
    if not devices:
        error_count = sum(bool(snapshot.get("error")) for snapshot in snapshots)
        return f"device_memory=unavailable nodes={len(snapshots)} monitor_errors={error_count}"

    used_mb = sum(float(device["used_mb"]) for device in devices)
    total_mb = sum(float(device["total_mb"]) for device in devices)
    percentages = [
        100.0 * float(device["used_mb"]) / float(device["total_mb"])
        for device in devices
        if float(device["total_mb"]) > 0
    ]
    utilization = 100.0 * used_mb / total_mb if total_mb else 0.0
    backend_names = ",".join(sorted({str(snapshot.get("backend", "unknown")) for snapshot in snapshots}))
    range_text = (
        f"{min(percentages):.1f}-{max(percentages):.1f}%" if percentages else "unavailable"
    )
    return (
        f"device_memory={used_mb:.0f}/{total_mb:.0f}MB ({utilization:.1f}%) "
        f"per_device={range_text} devices={len(devices)} nodes={len(snapshots)} backend={backend_names}"
    )


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
    memory_dir = Path(args.memory_dir) if args.memory_dir else queue_dir.parent / "device_memory"
    print(memory_status(memory_dir, args.memory_max_age))


if __name__ == "__main__":
    main()
