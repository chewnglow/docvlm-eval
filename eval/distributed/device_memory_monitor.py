#!/usr/bin/env python3
"""Sample node-local GPU VRAM or Ascend HBM usage during an evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--device-indices", required=True, help="Comma-separated physical device IDs.")
    parser.add_argument(
        "--backend",
        choices=("auto", "ascend", "npu", "nvidia", "cuda"),
        default="auto",
    )
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="Write one sample and exit.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "node"


def parse_device_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("device indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices) or len(indices) != len(set(indices)):
        raise ValueError("device indices must be a non-empty list of unique non-negative integers")
    return indices


def parse_nvidia_csv(output: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            index, used_mb, total_mb = int(fields[0]), float(fields[1]), float(fields[2])
        except ValueError:
            continue
        devices.append({"index": index, "used_mb": used_mb, "total_mb": total_mb})
    return devices


def parse_ascend_info(output: str) -> list[dict[str, Any]]:
    """Parse the device rows containing HBM-Usage(MB) from ``npu-smi info``."""
    devices: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        pairs = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)", line)
        if not pairs:
            continue
        columns = [column.strip() for column in line.split("|") if column.strip()]
        if not columns:
            continue
        identity = re.match(r"^(\d+)\s+(\d+)(?:\s|$)", columns[0])
        if identity is None:
            continue
        index = int(identity.group(1))
        used_mb, total_mb = (float(value) for value in pairs[-1])
        if total_mb <= 0:
            continue
        devices[index] = {"index": index, "used_mb": used_mb, "total_mb": total_mb}
    return [devices[index] for index in sorted(devices)]


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    return result.stdout


def resolve_backend(requested: str) -> str:
    aliases = {"npu": "ascend", "cuda": "nvidia"}
    if requested != "auto":
        return aliases.get(requested, requested)
    device_type = os.environ.get("DEVICE_TYPE", "").lower()
    if device_type in {"ascend", "npu"}:
        return "ascend"
    if device_type in {"cuda", "gpu", "nvidia"}:
        return "nvidia"
    if shutil.which("npu-smi"):
        return "ascend"
    if shutil.which("nvidia-smi"):
        return "nvidia"
    raise RuntimeError("neither npu-smi nor nvidia-smi is available")


def query_devices(backend: str, requested_indices: list[int]) -> list[dict[str, Any]]:
    if backend == "nvidia":
        output = run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        devices = parse_nvidia_csv(output)
    else:
        devices = parse_ascend_info(run_command(["npu-smi", "info"]))

    by_index = {int(device["index"]): device for device in devices}
    missing = [index for index in requested_indices if index not in by_index]
    if missing:
        raise RuntimeError(f"{backend} memory output omitted requested device IDs {missing}")
    return [by_index[index] for index in requested_indices]


def make_sample(node_id: str, backend: str, interval: float, devices: list[dict[str, Any]]) -> dict[str, Any]:
    used_mb = sum(float(device["used_mb"]) for device in devices)
    total_mb = sum(float(device["total_mb"]) for device in devices)
    enriched = []
    for device in devices:
        item = dict(device)
        item["utilization_percent"] = (
            100.0 * float(item["used_mb"]) / float(item["total_mb"])
            if float(item["total_mb"])
            else 0.0
        )
        enriched.append(item)
    return {
        "timestamp": time.time(),
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "active": True,
        "node_id": node_id,
        "backend": backend,
        "interval_seconds": interval,
        "used_mb": used_mb,
        "total_mb": total_mb,
        "utilization_percent": 100.0 * used_mb / total_mb if total_mb else 0.0,
        "devices": enriched,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def append_history(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def sample_error(node_id: str, backend: str, interval: float, exc: Exception) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "active": True,
        "node_id": node_id,
        "backend": backend,
        "interval_seconds": interval,
        "error": f"{type(exc).__name__}: {exc}",
        "devices": [],
    }


def format_sample(sample: dict[str, Any]) -> str:
    if sample.get("error"):
        return f"device-memory node={sample['node_id']} backend={sample['backend']} error={sample['error']}"
    per_device = ",".join(
        f"{device['index']}:{device['used_mb']:.0f}/{device['total_mb']:.0f}MB"
        for device in sample["devices"]
    )
    return (
        f"device-memory node={sample['node_id']} backend={sample['backend']} "
        f"used={sample['used_mb']:.0f}/{sample['total_mb']:.0f}MB "
        f"({sample['utilization_percent']:.1f}%) devices={per_device}"
    )


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be > 0")
    try:
        indices = parse_device_indices(args.device_indices)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        backend = resolve_backend(args.backend)
    except Exception as exc:
        backend = args.backend
        print(f"device-memory monitor unavailable: {exc}", flush=True)

    output_dir = Path(args.output_dir)
    name = safe_name(args.node_id)
    current_path = output_dir / "current" / f"{name}.json"
    history_path = output_dir / "history" / f"{name}.jsonl"
    stop_event = threading.Event()

    def stop(*_: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    last_sample: dict[str, Any] | None = None
    try:
        while not stop_event.is_set():
            try:
                devices = query_devices(backend, indices)
                last_sample = make_sample(args.node_id, backend, args.interval, devices)
            except Exception as exc:
                last_sample = sample_error(args.node_id, backend, args.interval, exc)
            write_json_atomic(current_path, last_sample)
            append_history(history_path, last_sample)
            print(format_sample(last_sample), flush=True)
            if args.once or stop_event.wait(args.interval):
                break
    finally:
        if last_sample is not None and not args.once:
            stopped = dict(last_sample)
            stopped["active"] = False
            stopped["stopped_at"] = time.time()
            write_json_atomic(current_path, stopped)


if __name__ == "__main__":
    main()
