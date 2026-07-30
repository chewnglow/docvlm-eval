#!/usr/bin/env python3
"""Launch the multi-node Ascend evaluation from one Ray job."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


DIST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIST_DIR))

from device_plan import MAX_TOTAL_DEVICES, MIN_TOTAL_DEVICES  # noqa: E402


FORWARDED_ENV = (
    "OPENAI_API_KEY",
    "BENCHMARKS",
    "MAX_IMAGES",
    "MAX_TOKENS",
    "MAX_MODEL_LEN",
    "LIMIT_MM_PER_PROMPT",
    "REQUEST_CONCURRENCY",
    "REQUEST_TIMEOUT",
    "MAX_RETRIES",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "GPU_MEMORY_UTILIZATION",
    "MM_PROCESSOR_CACHE_GB",
    "ENABLE_PREFIX_CACHING",
    "MAX_RECORDS_PER_TASK",
    "IMAGE_DATA_CACHE_MB",
    "LEASE_SECONDS",
    "MAX_TASK_ATTEMPTS",
    "QUEUE_WAIT_POLLS",
    "QUEUE_WAIT_INTERVAL",
    "SERVER_WAIT_TIMEOUT",
    "SCORE_EXTRA_ARGS",
    "VLLM_EXTRA_ARGS",
    "BASE_PORT",
)


@dataclass(frozen=True)
class ClusterNode:
    node_id: str
    node_ip: str
    capacity: int


@dataclass(frozen=True)
class NodeAssignment:
    family: str
    model_key: str
    node_id: str
    node_ip: str
    node_rank: int
    local_devices: int
    reserved_devices: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Qwen and Step evaluation groups across an existing Ray cluster. "
            "Run this once through Ray Jobs or on the Ray head node."
        )
    )
    parser.add_argument("--address", default="auto", help="Ray address; default: auto")
    parser.add_argument("--total-devices", type=int, required=True)
    parser.add_argument("--qwen-devices", type=int)
    parser.add_argument("--step-devices", type=int)
    parser.add_argument("--node-capacity", type=int, default=8)
    parser.add_argument(
        "--resource-key",
        default="NPU",
        help="Ray custom resource advertised by each Ascend compute node.",
    )
    parser.add_argument("--qwen-model-key", default="qwen3.5-9b")
    parser.add_argument("--step-model-key", default="step3vl-10b")
    parser.add_argument("--qwen-model-path")
    parser.add_argument("--step-model-path")
    parser.add_argument("--qwen-served-model-name")
    parser.add_argument("--step-served-model-name")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--project-root",
        default=".",
        help=(
            "Project path as seen by Ray workers. Keep '.' when submitting with "
            "'ray job submit --working-dir .'."
        ),
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON", "python3"),
        help="Python executable available at the same path on every node.",
    )
    parser.add_argument(
        "--keep-servers",
        action="store_true",
        help="Leave vLLM servers running after the suite finishes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Ray node allocation without starting model servers.",
    )
    return parser.parse_args()


def resolve_split(
    total_devices: int,
    qwen_devices: int | None,
    step_devices: int | None,
) -> tuple[int, int]:
    if not MIN_TOTAL_DEVICES <= total_devices <= MAX_TOTAL_DEVICES:
        raise ValueError(
            f"TOTAL_DEVICES must be between {MIN_TOTAL_DEVICES} and "
            f"{MAX_TOTAL_DEVICES}, got {total_devices}"
        )
    if qwen_devices is None and step_devices is None:
        qwen_devices = (total_devices + 1) // 2
        step_devices = total_devices - qwen_devices
    elif qwen_devices is None:
        qwen_devices = total_devices - int(step_devices)
    elif step_devices is None:
        step_devices = total_devices - int(qwen_devices)

    qwen_devices = int(qwen_devices)
    step_devices = int(step_devices)
    if qwen_devices < 1 or step_devices < 1:
        raise ValueError("Ray orchestration requires at least one device for each model group")
    if qwen_devices + step_devices != total_devices:
        raise ValueError(
            "QWEN_DEVICES + STEP_DEVICES must equal TOTAL_DEVICES: "
            f"{qwen_devices} + {step_devices} != {total_devices}"
        )
    return qwen_devices, step_devices


def discover_compute_nodes(
    ray_nodes: Iterable[dict[str, Any]],
    resource_key: str,
) -> list[ClusterNode]:
    if not resource_key.strip():
        raise ValueError("Ray resource key must not be empty")
    nodes: list[ClusterNode] = []
    for raw in ray_nodes:
        if not raw.get("Alive"):
            continue
        resources = raw.get("Resources") or {}
        capacity = int(float(resources.get(resource_key, 0)))
        if capacity < 1:
            continue
        node_id = str(raw.get("NodeID") or "")
        node_ip = str(raw.get("NodeManagerAddress") or "")
        if not node_id or not node_ip:
            continue
        nodes.append(ClusterNode(node_id=node_id, node_ip=node_ip, capacity=capacity))
    return sorted(nodes, key=lambda node: (node.node_ip, node.node_id))


def build_assignments(
    nodes: list[ClusterNode],
    *,
    qwen_devices: int,
    step_devices: int,
    node_capacity: int,
    qwen_model_key: str,
    step_model_key: str,
) -> list[NodeAssignment]:
    if node_capacity < 1:
        raise ValueError("NODE_DEVICE_CAPACITY must be >= 1")

    group_specs = (
        ("qwen", qwen_model_key, qwen_devices),
        ("step", step_model_key, step_devices),
    )
    required_nodes = sum(math.ceil(devices / node_capacity) for _, _, devices in group_specs)
    eligible = [node for node in nodes if node.capacity >= node_capacity]
    if len(eligible) < required_nodes:
        capacities = ", ".join(f"{node.node_ip}:{node.capacity}" for node in nodes) or "none"
        raise ValueError(
            f"Need {required_nodes} compute nodes with at least {node_capacity} "
            f"devices each; found {len(eligible)}. "
            f"Advertised capacities: {capacities}"
        )

    assignments: list[NodeAssignment] = []
    node_index = 0
    for family, model_key, group_devices in group_specs:
        remaining = group_devices
        node_rank = 0
        while remaining:
            node = eligible[node_index]
            local_devices = min(node_capacity, remaining)
            assignments.append(
                NodeAssignment(
                    family=family,
                    model_key=model_key,
                    node_id=node.node_id,
                    node_ip=node.node_ip,
                    node_rank=node_rank,
                    local_devices=local_devices,
                    # Reserve the complete node so another experiment cannot use
                    # devices left idle by a partial final model-group node.
                    reserved_devices=node.capacity,
                )
            )
            node_index += 1
            node_rank += 1
            remaining -= local_devices
    return assignments


def safe_node_name(assignment: NodeAssignment) -> str:
    value = f"ray-{assignment.family}-{assignment.node_rank}-{assignment.node_ip}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def forwarded_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    return {key: source[key] for key in FORWARDED_ENV if key in source}


def validate_data_paths(
    output_root: str,
    dataset_root: str,
    model_paths: Iterable[str | None],
) -> None:
    required = {"--output-root": output_root, "--dataset-root": dataset_root}
    for option, value in required.items():
        if not Path(value).expanduser().is_absolute():
            raise ValueError(f"{option} must be an absolute path visible on every node")
    for value in model_paths:
        if value and not Path(value).expanduser().is_absolute():
            raise ValueError("model paths must be absolute paths visible on every model node")


def build_node_environment(
    assignment: NodeAssignment,
    *,
    total_devices: int,
    qwen_devices: int,
    step_devices: int,
    node_capacity: int,
    output_root: str,
    dataset_root: str,
    python: str,
    common_env: dict[str, str],
    model_path: str | None = None,
    served_model_name: str | None = None,
) -> dict[str, str]:
    env = dict(common_env)
    node_name = safe_node_name(assignment)
    control_root = Path(output_root) / "_ray_nodes" / assignment.model_key / node_name
    env.update(
        {
            "TOTAL_DEVICES": str(total_devices),
            "QWEN_DEVICES": str(qwen_devices),
            "STEP_DEVICES": str(step_devices),
            "NODE_DEVICE_CAPACITY": str(node_capacity),
            "NODE_RANK": str(assignment.node_rank),
            "NODE_ID": node_name,
            "DEVICES_PER_NODE": str(assignment.local_devices),
            "MODEL_KEY": assignment.model_key,
            "OUTPUT_ROOT": output_root,
            "DATASET_ROOT": dataset_root,
            "PYTHON": python,
            "LOG_DIR": str(control_root / "server_logs"),
            "PID_DIR": str(control_root / "pids"),
        }
    )
    if model_path:
        env["MODEL_PATH"] = model_path
    if served_model_name:
        env["SERVED_MODEL_NAME"] = served_model_name
    return env


def run_node_suite(
    assignment_data: dict[str, Any],
    node_env: dict[str, str],
    project_root: str,
    keep_servers: bool,
) -> dict[str, Any]:
    """Run on a Ray worker pinned to one compute node."""
    assignment = NodeAssignment(**assignment_data)
    root = Path(project_root).expanduser().resolve()
    suite = root / "eval" / "distributed" / "run_ascend_suite.sh"
    stop = root / "eval" / "distributed" / "stop_vllm_servers.sh"
    if not suite.is_file():
        raise FileNotFoundError(
            f"{suite} is unavailable on Ray node {assignment.node_ip}. "
            "Submit with '--working-dir .' or use a shared --project-root."
        )

    env = os.environ.copy()
    env.update(node_env)
    print(
        f"[ray-node] {assignment.node_ip}: {assignment.model_key}, "
        f"rank={assignment.node_rank}, devices={assignment.local_devices}",
        flush=True,
    )
    try:
        completed = subprocess.run(["bash", str(suite)], cwd=root, env=env, check=False)
    finally:
        if not keep_servers and stop.is_file():
            subprocess.run(["bash", str(stop)], cwd=root, env=env, check=False)

    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, ["bash", str(suite)])
    return {
        "node_ip": assignment.node_ip,
        "model_key": assignment.model_key,
        "node_rank": assignment.node_rank,
        "local_devices": assignment.local_devices,
    }


def print_plan(assignments: list[NodeAssignment], resource_key: str) -> None:
    print(f"Ray allocation using resource {resource_key!r}:")
    for assignment in assignments:
        print(
            f"  {assignment.node_ip:<15} {assignment.family:<5} "
            f"rank={assignment.node_rank:<2} devices={assignment.local_devices} "
            f"reserve={assignment.reserved_devices}"
        )


def main() -> None:
    args = parse_args()
    try:
        qwen_devices, step_devices = resolve_split(
            args.total_devices,
            args.qwen_devices,
            args.step_devices,
        )
        validate_data_paths(
            args.output_root,
            args.dataset_root,
            (args.qwen_model_path, args.step_model_path),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ImportError as exc:
        raise SystemExit(
            "Ray is required for cluster orchestration. Install "
            "'ray[default]' from eval/requirements-benchmark.txt."
        ) from exc

    ray.init(address=args.address)
    try:
        nodes = discover_compute_nodes(ray.nodes(), args.resource_key)
        assignments = build_assignments(
            nodes,
            qwen_devices=qwen_devices,
            step_devices=step_devices,
            node_capacity=args.node_capacity,
            qwen_model_key=args.qwen_model_key,
            step_model_key=args.step_model_key,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print_plan(assignments, args.resource_key)
    if args.dry_run:
        return

    common_env = forwarded_environment()
    remote_runner = ray.remote(max_retries=0, num_cpus=0)(run_node_suite)
    futures: list[Any] = []
    for assignment in assignments:
        if assignment.family == "qwen":
            model_path = args.qwen_model_path
            served_model_name = args.qwen_served_model_name
        else:
            model_path = args.step_model_path
            served_model_name = args.step_served_model_name
        node_env = build_node_environment(
            assignment,
            total_devices=args.total_devices,
            qwen_devices=qwen_devices,
            step_devices=step_devices,
            node_capacity=args.node_capacity,
            output_root=args.output_root,
            dataset_root=args.dataset_root,
            python=args.python,
            common_env=common_env,
            model_path=model_path,
            served_model_name=served_model_name,
        )
        futures.append(
            remote_runner.options(
                resources={args.resource_key: float(assignment.reserved_devices)},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=assignment.node_id,
                    soft=False,
                ),
            ).remote(
                asdict(assignment),
                node_env,
                args.project_root,
                args.keep_servers,
            )
        )

    failures: list[str] = []
    remaining = list(futures)
    while remaining:
        ready, remaining = ray.wait(remaining, num_returns=1)
        try:
            result = ray.get(ready[0])
            print(
                f"[ray-complete] {result['node_ip']} {result['model_key']} "
                f"rank={result['node_rank']}",
                flush=True,
            )
        except Exception as exc:  # Ray wraps the remote exception.
            failures.append(str(exc))
            print(f"[ray-failed] {exc}", file=sys.stderr, flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} Ray node launcher(s) failed")
    print("All Ray node launchers completed successfully.")


if __name__ == "__main__":
    main()
