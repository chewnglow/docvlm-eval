#!/usr/bin/env python3
"""Render a suite-wide evaluation dashboard from shared queue state."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


TERMINAL_STATES = {"COMPLETE", "PARTIAL", "INVALID"}


@dataclass
class Experiment:
    run_name: str
    model: str
    dataset: str
    split: str
    sequence: int = 0
    allocated_devices: int = 0
    expected: int | None = None
    successful: int = 0
    excluded_failed: int = 0
    pending_tasks: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_tasks: int = 0
    state: str = "WAITING"
    rate_per_hour: float | None = None
    eta_seconds: float | None = None
    elapsed_seconds: float | None = None
    count_error: str | None = None
    run_dir: str = ""

    @property
    def settled(self) -> int:
        return self.successful + self.excluded_failed

    @property
    def remaining(self) -> int | None:
        return None if self.expected is None else max(0, self.expected - self.settled)


@dataclass
class DeviceRow:
    experiment: str
    model: str
    benchmark: str
    node: str
    index: int
    backend: str
    used_mb: float
    total_mb: float
    utilization_percent: float
    age_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show suite-wide experiment progress, ETA, failures, and per-device memory."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Refresh continuously at this interval; default prints once.",
    )
    parser.add_argument("--memory-max-age", type=float, default=120.0)
    parser.add_argument("--eta-window-minutes", type=float, default=30.0)
    parser.add_argument("--hide-devices", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable snapshot.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def path_mtime(path: Path, default: float) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return default


def result_events(run_dir: Path) -> dict[str, float | None]:
    events: dict[str, float | None] = {}
    shard_dir = run_dir / "shards"
    paths = sorted(shard_dir.glob("*.jsonl"))
    if not paths and (run_dir / "predictions.jsonl").is_file():
        paths = [run_dir / "predictions.jsonl"]
    for path in paths:
        try:
            handle = path.open()
        except OSError:
            continue
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    record_id = str(row["id"])
                    completed_at = row.get("completed_at")
                    timestamp = float(completed_at) if completed_at is not None else None
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                previous = events.get(record_id)
                if previous is None or (timestamp is not None and timestamp > previous):
                    events[record_id] = timestamp
    return events


def jsonl_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    try:
        handle = path.open()
    except OSError:
        return identifiers
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                identifiers.add(str(json.loads(line)["id"]))
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    return identifiers


def failed_record_events(queue_dir: Path, successful_ids: set[str]) -> dict[str, float]:
    events: dict[str, float] = {}
    for path in (queue_dir / "failed").glob("task-*.json"):
        try:
            task = read_json(path)
            timestamp = float(task.get("failed_at", task.get("last_attempt_at", path.stat().st_mtime)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        for raw_id in task.get("record_ids", []):
            record_id = str(raw_id)
            if record_id not in successful_ids:
                events[record_id] = timestamp
    return events


def throughput(
    event_times: list[float],
    settled: int,
    started_at: float,
    ended_at: float,
    now: float,
    window_seconds: float,
    terminal: bool,
) -> float | None:
    elapsed = max(ended_at - started_at, 1.0)
    lifetime = 3600.0 * settled / elapsed if settled else None
    if terminal or len(event_times) < 2:
        return lifetime
    window_start = max(started_at, now - window_seconds)
    recent = [timestamp for timestamp in event_times if timestamp >= window_start]
    recent_elapsed = max(now - window_start, 1.0)
    if len(recent) >= 2 and recent_elapsed >= 60:
        return 3600.0 * len(recent) / recent_elapsed
    return lifetime


def queue_state(
    experiment: Experiment,
    run_dir: Path,
    now: float,
    eta_window_seconds: float,
) -> None:
    queue_dir = run_dir / "queue"
    manifest_path = queue_dir / "manifest.json"
    if not manifest_path.is_file():
        experiment.state = "INVALID" if experiment.expected == 0 or experiment.count_error else "WAITING"
        return
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        experiment.state = "INVALID"
        return

    experiment.expected = int(manifest.get("record_count", experiment.expected or 0))
    experiment.dataset = str(manifest.get("dataset", experiment.dataset))
    experiment.split = str(manifest.get("split", experiment.split))
    states = {}
    for state in ("pending", "claimed", "completed", "failed"):
        states[state] = len(list((queue_dir / state).glob("task-*.json")))
    experiment.pending_tasks = states["pending"]
    experiment.active_tasks = states["claimed"]
    experiment.completed_tasks = states["completed"]
    experiment.failed_tasks = states["failed"]
    experiment.total_tasks = int(manifest.get("task_count", sum(states.values())))

    successes = result_events(run_dir)
    integrity_error = None
    records_path = queue_dir / "records.jsonl"
    if records_path.is_file():
        expected_ids = jsonl_ids(records_path)
        extra_ids = set(successes) - expected_ids
        if len(expected_ids) != experiment.expected:
            integrity_error = (
                f"queue contains {len(expected_ids)} unique IDs but manifest expects {experiment.expected}"
            )
        elif extra_ids:
            integrity_error = f"{len(extra_ids)} prediction IDs are outside the current queue"
        successes = {record_id: timestamp for record_id, timestamp in successes.items() if record_id in expected_ids}
    failed = failed_record_events(queue_dir, set(successes))
    experiment.successful = len(successes)
    experiment.excluded_failed = len(failed)
    start = float(manifest.get("created_at", path_mtime(manifest_path, now)))
    metrics_path = run_dir / "metrics.json"
    terminal = metrics_path.is_file()
    end = path_mtime(metrics_path, now) if terminal else now
    experiment.elapsed_seconds = max(0.0, end - start)

    if experiment.expected == 0:
        experiment.state = "INVALID"
    elif terminal:
        try:
            coverage = read_json(metrics_path).get("coverage", {})
            partial = coverage.get("status") == "partial" or experiment.excluded_failed > 0
        except (OSError, json.JSONDecodeError):
            partial = experiment.excluded_failed > 0
        experiment.state = "PARTIAL" if partial else "COMPLETE"
    elif experiment.pending_tasks or experiment.active_tasks:
        experiment.state = "RUNNING" if experiment.active_tasks else "QUEUED"
    elif experiment.total_tasks and experiment.completed_tasks + experiment.failed_tasks >= experiment.total_tasks:
        terminal_paths = list((queue_dir / "completed").glob("task-*.json"))
        terminal_paths.extend((queue_dir / "failed").glob("task-*.json"))
        last_terminal = max((path_mtime(path, now) for path in terminal_paths), default=now)
        experiment.state = "UNSCORED" if now - last_terminal > 600 else "FINALIZING"
    else:
        experiment.state = "PREPARING"
    if integrity_error:
        experiment.state = "INVALID"
        experiment.count_error = integrity_error

    event_times = [timestamp for timestamp in successes.values() if timestamp is not None]
    event_times.extend(failed.values())
    experiment.rate_per_hour = throughput(
        event_times,
        experiment.settled,
        start,
        end,
        now,
        eta_window_seconds,
        terminal,
    )
    remaining = experiment.remaining
    if remaining == 0:
        experiment.eta_seconds = 0.0
    elif remaining is not None and experiment.rate_per_hour and experiment.rate_per_hour > 0:
        experiment.eta_seconds = 3600.0 * remaining / experiment.rate_per_hour
    if integrity_error:
        experiment.eta_seconds = None


def infer_model(run_name: str, dataset: str) -> str:
    marker = f"-{dataset}-"
    return run_name.split(marker, 1)[0] if marker in run_name else run_name


def discover_experiments(output_root: Path, now: float, eta_window_seconds: float) -> list[Experiment]:
    experiments: dict[str, Experiment] = {}
    for path in sorted((output_root / "_suite_plans").glob("*.json")):
        try:
            plan = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        model = str(plan.get("served_model_name", plan.get("model_key", path.stem)))
        allocated_devices = int(plan.get("device_count", 0))
        for raw in plan.get("experiments", []):
            run_name = str(raw["run_name"])
            experiments[run_name] = Experiment(
                run_name=run_name,
                model=model,
                dataset=str(raw["dataset"]),
                split=str(raw.get("split", "val")),
                sequence=int(raw.get("sequence", 0)),
                allocated_devices=allocated_devices,
                expected=raw.get("expected_count"),
                count_error=raw.get("count_error"),
                run_dir=str(output_root / run_name),
            )

    for manifest_path in sorted(output_root.glob("*/queue/manifest.json")):
        run_dir = manifest_path.parent.parent
        if run_dir.name.startswith("_"):
            continue
        try:
            manifest = read_json(manifest_path)
            dataset = str(manifest.get("dataset", "unknown"))
        except (OSError, json.JSONDecodeError):
            dataset = "unknown"
        if run_dir.name not in experiments:
            experiments[run_dir.name] = Experiment(
                run_name=run_dir.name,
                model=infer_model(run_dir.name, dataset),
                dataset=dataset,
                split="val",
                run_dir=str(run_dir),
            )

    for experiment in experiments.values():
        queue_state(experiment, Path(experiment.run_dir), now, eta_window_seconds)

    model_rates: dict[str, float] = {}
    for model in {experiment.model for experiment in experiments.values()}:
        candidates = [
            experiment.rate_per_hour
            for experiment in experiments.values()
            if experiment.model == model
            and experiment.state in {"RUNNING", "QUEUED", "FINALIZING"}
            and experiment.rate_per_hour
        ]
        if not candidates:
            candidates = [
                experiment.rate_per_hour
                for experiment in experiments.values()
                if experiment.model == model and experiment.rate_per_hour
            ]
        if candidates:
            model_rates[model] = candidates[0]
    for experiment in experiments.values():
        if experiment.state == "WAITING" and experiment.expected is not None:
            rate = model_rates.get(experiment.model)
            experiment.rate_per_hour = rate
            if rate:
                experiment.eta_seconds = 3600.0 * experiment.expected / rate
    return sorted(experiments.values(), key=lambda item: (item.model.lower(), item.sequence, item.dataset))


def discover_devices(
    experiments: list[Experiment],
    now: float,
    max_age: float,
) -> tuple[list[DeviceRow], list[str]]:
    devices: list[DeviceRow] = []
    errors: list[str] = []
    for experiment in experiments:
        current = Path(experiment.run_dir) / "device_memory" / "current"
        for path in sorted(current.glob("*.json")):
            try:
                snapshot = read_json(path)
                age = now - float(snapshot["timestamp"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if not snapshot.get("active") or age > max_age:
                continue
            if snapshot.get("error"):
                errors.append(f"{experiment.run_name}/{snapshot.get('node_id', path.stem)}: {snapshot['error']}")
            for device in snapshot.get("devices", []):
                total = float(device["total_mb"])
                used = float(device["used_mb"])
                devices.append(
                    DeviceRow(
                        experiment=experiment.run_name,
                        model=experiment.model,
                        benchmark=f"{experiment.dataset}:{experiment.split}",
                        node=str(snapshot.get("node_id", path.stem)),
                        index=int(device["index"]),
                        backend=str(snapshot.get("backend", "unknown")),
                        used_mb=used,
                        total_mb=total,
                        utilization_percent=100.0 * used / total if total else 0.0,
                        age_seconds=max(0.0, age),
                    )
                )
    return sorted(devices, key=lambda item: (item.model.lower(), item.node, item.index)), errors


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def clipped(value: str, width: int) -> str:
    return value if len(value) <= width else value[: max(1, width - 1)] + "…"


def overall_summary(experiments: list[Experiment]) -> dict[str, Any]:
    known = [experiment for experiment in experiments if experiment.expected is not None]
    expected = sum(int(experiment.expected or 0) for experiment in known)
    settled = sum(experiment.settled for experiment in known)
    states: dict[str, int] = {}
    for experiment in experiments:
        states[experiment.state] = states.get(experiment.state, 0) + 1

    if not experiments:
        status = "NO EXPERIMENTS"
    elif any(experiment.state in {"INVALID", "UNSCORED"} for experiment in experiments):
        status = "ATTENTION"
    elif all(experiment.state in TERMINAL_STATES for experiment in experiments):
        status = "COMPLETE WITH EXCLUSIONS" if states.get("PARTIAL") else "COMPLETE"
    elif any(
        experiment.state in {"RUNNING", "QUEUED", "FINALIZING", "PREPARING"}
        for experiment in experiments
    ):
        status = "RUNNING"
    else:
        status = "WAITING"

    pipelines: dict[str, list[Experiment]] = {}
    for experiment in experiments:
        pipelines.setdefault(experiment.model, []).append(experiment)
    pipeline_etas: list[float] = []
    unknown_eta = False
    for pipeline in pipelines.values():
        remaining = [item for item in pipeline if item.state not in TERMINAL_STATES]
        if any(item.eta_seconds is None for item in remaining):
            unknown_eta = True
        else:
            pipeline_etas.append(sum(float(item.eta_seconds or 0) for item in remaining))
    eta = None if unknown_eta or not experiments else max(pipeline_etas, default=0.0)
    aggregate_rate = sum(
        float(experiment.rate_per_hour or 0)
        for experiment in experiments
        if experiment.state in {"RUNNING", "QUEUED", "FINALIZING"}
    )
    return {
        "status": status,
        "experiment_count": len(experiments),
        "states": states,
        "expected": expected,
        "settled": settled,
        "progress_percent": 100.0 * settled / expected if expected else 0.0,
        "aggregate_rate_per_hour": aggregate_rate,
        "eta_seconds": eta,
        "eta_has_unknown_components": unknown_eta,
        "expected_active_devices": sum(
            max(
                (
                    experiment.allocated_devices
                    for experiment in experiments
                    if experiment.model == model
                    and experiment.state
                    in {"RUNNING", "QUEUED", "FINALIZING", "PREPARING"}
                ),
                default=0,
            )
            for model in {experiment.model for experiment in experiments}
        ),
    }


def render(
    experiments: list[Experiment],
    devices: list[DeviceRow],
    memory_errors: list[str],
    now: float,
    show_devices: bool = True,
) -> str:
    overall = overall_summary(experiments)
    lines = [
        f"Evaluation suite  {datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')}",
        (
            f"Overall: {overall['status']}  settled {overall['settled']}/{overall['expected']} "
            f"({overall['progress_percent']:.1f}%)  rate {overall['aggregate_rate_per_hour']:.1f}/h  "
            f"ETA {format_duration(overall['eta_seconds'])}"
        ),
        "States: " + "  ".join(f"{state.lower()}={count}" for state, count in sorted(overall["states"].items())),
        "",
        "Experiments",
        (
            f"{'STATE':<11} {'MODEL':<20} {'BENCHMARK':<22} {'SETTLED':>13} "
            f"{'PROGRESS':>9} {'TASK S/A/P':>13} {'EXCL':>5} {'RATE/H':>9} {'ETA':>8}"
        ),
        "-" * 121,
    ]
    for experiment in experiments:
        expected = "?" if experiment.expected is None else str(experiment.expected)
        records = f"{experiment.settled}/{expected}"
        progress = (
            f"{100.0 * experiment.settled / experiment.expected:.1f}%"
            if experiment.expected
            else "--"
        )
        tasks = (
            f"{experiment.completed_tasks + experiment.failed_tasks}/"
            f"{experiment.active_tasks}/{experiment.pending_tasks}"
            if experiment.total_tasks
            else "--"
        )
        benchmark = f"{experiment.dataset}:{experiment.split}"
        eta = format_duration(experiment.eta_seconds)
        if experiment.state == "WAITING" and experiment.eta_seconds is not None:
            eta = "~" + eta
        rate = f"{experiment.rate_per_hour:.1f}" if experiment.rate_per_hour else "--"
        lines.append(
            f"{experiment.state:<11} {clipped(experiment.model, 20):<20} "
            f"{clipped(benchmark, 22):<22} {records:>13} {progress:>9} {tasks:>13} "
            f"{experiment.excluded_failed:>5} {rate:>9} {eta:>8}"
        )
    if not experiments:
        lines.append("No suite plans or experiment queues found.")

    experiment_warnings = []
    for experiment in experiments:
        if experiment.count_error:
            experiment_warnings.append(
                f"{experiment.model} / {experiment.dataset}:{experiment.split}: {experiment.count_error}"
            )
        elif experiment.state == "UNSCORED":
            experiment_warnings.append(
                f"{experiment.model} / {experiment.dataset}:{experiment.split}: "
                "queue drained more than 10 minutes ago but metrics.json is absent"
            )
    if experiment_warnings:
        lines.append("\nExperiment warnings:")
        lines.extend(f"  - {warning}" for warning in experiment_warnings)

    if not show_devices:
        lines.append("\nDevice table hidden (--hide-devices).")
        if overall["eta_has_unknown_components"]:
            lines.append(
                "ETA note: overall ETA is pending until every active model pipeline has a measured rate."
            )
        return "\n".join(lines)

    lines.extend(["", "Device memory (all fresh active devices)"])
    if devices:
        used = sum(device.used_mb for device in devices)
        total = sum(device.total_mb for device in devices)
        lines.append(
            f"Aggregate: {used / 1024:.1f}/{total / 1024:.1f} GiB "
            f"({100.0 * used / total if total else 0.0:.1f}%) across {len(devices)} devices; "
            f"expected active allocation={overall['expected_active_devices'] or 'unknown'}"
        )
        lines.append(
            f"{'MODEL':<18} {'BENCHMARK':<20} {'NODE':<24} {'DEV':>4} {'TYPE':<7} "
            f"{'USED GiB':>10} {'TOTAL GiB':>10} {'OCCUPIED':>9} {'AGE':>6}"
        )
        lines.append("-" * 124)
        for device in devices:
            lines.append(
                f"{clipped(device.model, 18):<18} {clipped(device.benchmark, 20):<20} "
                f"{clipped(device.node, 24):<24} "
                f"{device.index:>4} {device.backend:<7} {device.used_mb / 1024:>10.1f} "
                f"{device.total_mb / 1024:>10.1f} {device.utilization_percent:>8.1f}% "
                f"{device.age_seconds:>5.0f}s"
            )
    else:
        lines.append(
            "No fresh active device-memory snapshots; "
            f"expected active allocation={overall['expected_active_devices'] or 'unknown'}."
        )
    if memory_errors:
        lines.append("Monitor warnings:")
        lines.extend(f"  - {error}" for error in memory_errors)
    if overall["eta_has_unknown_components"]:
        lines.append("\nETA note: overall ETA is pending until every active model pipeline has a measured rate.")
    return "\n".join(lines)


def snapshot(args: argparse.Namespace) -> tuple[list[Experiment], list[DeviceRow], list[str], float]:
    now = time.time()
    experiments = discover_experiments(
        Path(args.output_root),
        now,
        eta_window_seconds=args.eta_window_minutes * 60.0,
    )
    devices, errors = discover_devices(experiments, now, args.memory_max_age)
    return experiments, devices, errors, now


def main() -> None:
    args = parse_args()
    if args.watch < 0 or args.memory_max_age <= 0 or args.eta_window_minutes <= 0:
        raise SystemExit("watch must be >= 0; memory age and ETA window must be > 0")
    if args.json and args.watch:
        raise SystemExit("--json cannot be combined with --watch")
    try:
        while True:
            experiments, devices, errors, now = snapshot(args)
            if args.json:
                print(
                    json.dumps(
                        {
                            "timestamp": now,
                            "overall": overall_summary(experiments),
                            "experiments": [asdict(experiment) for experiment in experiments],
                            "devices": [] if args.hide_devices else [asdict(device) for device in devices],
                            "memory_errors": errors,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                return
            output = render(
                experiments,
                devices,
                errors,
                now,
                show_devices=not args.hide_devices,
            )
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(output, flush=True)
            if not args.watch:
                return
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
