#!/usr/bin/env python3
"""Pull tasks atomically from a shared filesystem queue and evaluate them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from run_benchmarks import load_done_ids, run_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--image-data-cache-mb", type=int, default=256)
    parser.add_argument("--lease-seconds", type=float, default=3600)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--max-task-attempts", type=int, default=3)
    return parser.parse_args()


def safe_worker_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "worker"


def load_records(queue_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with (queue_dir / "records.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["id"])] = record
    return records


def recover_stale_claims(queue_dir: Path, lease_seconds: float) -> int:
    recovered = 0
    now = time.time()
    for claimed in (queue_dir / "claimed").glob("task-*.json"):
        try:
            if now - claimed.stat().st_mtime <= lease_seconds:
                continue
            if time.time() - claimed.stat().st_mtime <= lease_seconds:
                continue
            os.rename(claimed, queue_dir / "pending" / claimed.name)
            recovered += 1
        except FileNotFoundError:
            continue
    return recovered


def claim_task(queue_dir: Path) -> Path | None:
    for pending in sorted((queue_dir / "pending").glob("task-*.json")):
        claimed = queue_dir / "claimed" / pending.name
        try:
            os.rename(pending, claimed)
            return claimed
        except FileNotFoundError:
            continue
    return None


class Heartbeat:
    def __init__(self, path: Path, interval: float) -> None:
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.path.touch(exist_ok=True)
            except FileNotFoundError:
                return

    def __enter__(self) -> "Heartbeat":
        self.path.touch(exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval))


def write_task(path: Path, task: dict[str, Any]) -> None:
    temp = path.with_suffix(f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    if args.request_concurrency < 1 or args.max_task_attempts < 1:
        raise SystemExit("request concurrency and task attempts must be >= 1")
    queue_dir = Path(args.queue_dir)
    if not (queue_dir / "READY").exists():
        raise SystemExit(f"Queue is not ready: {queue_dir}")
    records_by_id = load_records(queue_dir)
    worker_id = safe_worker_id(args.worker_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"worker-{worker_id}.jsonl"
    done = load_done_ids(output)
    runner_args = SimpleNamespace(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_images=args.max_images,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_concurrency=args.request_concurrency,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        image_data_cache_mb=args.image_data_cache_mb,
    )
    warmup_args = SimpleNamespace(**vars(runner_args))
    warmup_args.request_concurrency = 1

    heartbeat_interval = max(5.0, min(60.0, args.lease_seconds / 4.0))
    print(f"Worker {worker_id}: {args.base_url} -> {output}", flush=True)
    while True:
        recovered = recover_stale_claims(queue_dir, args.lease_seconds)
        if recovered:
            print(f"Worker {worker_id}: recovered {recovered} stale task(s)", flush=True)
        claimed = claim_task(queue_dir)
        if claimed is None:
            if any((queue_dir / "claimed").glob("task-*.json")):
                time.sleep(args.poll_seconds)
                continue
            break

        task = json.loads(claimed.read_text())
        task_started = time.time()
        task.setdefault("first_started_at", task_started)
        task["last_started_at"] = task_started
        write_task(claimed, task)
        task_records = [records_by_id[record_id] for record_id in task["record_ids"] if record_id not in done]
        try:
            with Heartbeat(claimed, heartbeat_interval):
                if task_records:
                    done = run_records(warmup_args, task_records[:1], output, done)
                    done = run_records(runner_args, task_records[1:], output, done)
            task["completed_at"] = time.time()
            write_task(claimed, task)
            os.rename(claimed, queue_dir / "completed" / claimed.name)
        except Exception as exc:
            done = load_done_ids(output)
            task["attempts"] = int(task.get("attempts", 0)) + 1
            task["last_error"] = f"{type(exc).__name__}: {exc}"
            task["last_attempt_at"] = time.time()
            if task["attempts"] >= args.max_task_attempts:
                task["failed_at"] = task["last_attempt_at"]
            write_task(claimed, task)
            destination = "failed" if task["attempts"] >= args.max_task_attempts else "pending"
            os.rename(claimed, queue_dir / destination / claimed.name)
            print(
                f"Worker {worker_id}: task {task['task_id']} failed attempt "
                f"{task['attempts']}/{args.max_task_attempts}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    failed = list((queue_dir / "failed").glob("task-*.json"))
    if failed:
        print(
            f"Worker {worker_id}: queue complete with {len(failed)} terminally failed task(s); "
            "their unresolved records will be excluded from scoring and written to failed_records.jsonl",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"Worker {worker_id}: queue complete", flush=True)


if __name__ == "__main__":
    main()
