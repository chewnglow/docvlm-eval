#!/usr/bin/env python3
"""Compute a flexible two-model device allocation and this node's active devices."""

from __future__ import annotations

import argparse
import math


MIN_TOTAL_DEVICES = 16
MAX_TOTAL_DEVICES = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-devices", type=int, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--node-capacity", type=int, default=8)
    parser.add_argument("--qwen-devices", type=int, default=None)
    parser.add_argument("--step-devices", type=int, default=None)
    return parser.parse_args()


def model_family(model_key: str) -> str:
    normalized = model_key.lower()
    if normalized.startswith("qwen3.5-9b"):
        return "qwen"
    if normalized.startswith("step3vl-10b") or normalized.startswith("step3vl10b"):
        return "step"
    raise ValueError(f"Cannot infer model family from MODEL_KEY={model_key!r}")


def make_plan(
    *,
    total_devices: int,
    model_key: str,
    node_rank: int,
    node_capacity: int,
    qwen_devices: int | None = None,
    step_devices: int | None = None,
) -> dict[str, int | str]:
    if not MIN_TOTAL_DEVICES <= total_devices <= MAX_TOTAL_DEVICES:
        raise ValueError(
            f"TOTAL_DEVICES must be between {MIN_TOTAL_DEVICES} and {MAX_TOTAL_DEVICES}, "
            f"got {total_devices}"
        )
    if node_rank < 0:
        raise ValueError("NODE_RANK must be >= 0")
    if node_capacity < 1:
        raise ValueError("NODE_DEVICE_CAPACITY must be >= 1")

    if qwen_devices is None and step_devices is None:
        qwen_devices = (total_devices + 1) // 2
        step_devices = total_devices - qwen_devices
    elif qwen_devices is None:
        qwen_devices = total_devices - int(step_devices)
    elif step_devices is None:
        step_devices = total_devices - int(qwen_devices)

    qwen_devices = int(qwen_devices)
    step_devices = int(step_devices)
    if qwen_devices < 0 or step_devices < 0:
        raise ValueError("QWEN_DEVICES and STEP_DEVICES must be >= 0")
    if qwen_devices + step_devices != total_devices:
        raise ValueError(
            f"QWEN_DEVICES + STEP_DEVICES must equal TOTAL_DEVICES: "
            f"{qwen_devices} + {step_devices} != {total_devices}"
        )

    family = model_family(model_key)
    group_devices = qwen_devices if family == "qwen" else step_devices
    if group_devices == 0:
        raise ValueError(f"The selected {family} model has zero allocated devices")
    group_nodes = math.ceil(group_devices / node_capacity)
    node_start = node_rank * node_capacity
    local_devices = max(0, min(node_capacity, group_devices - node_start))
    return {
        "family": family,
        "group_devices": group_devices,
        "group_nodes": group_nodes,
        "local_devices": local_devices,
        "qwen_devices": qwen_devices,
        "step_devices": step_devices,
    }


def main() -> None:
    args = parse_args()
    try:
        plan = make_plan(
            total_devices=args.total_devices,
            model_key=args.model_key,
            node_rank=args.node_rank,
            node_capacity=args.node_capacity,
            qwen_devices=args.qwen_devices,
            step_devices=args.step_devices,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        plan["group_devices"],
        plan["group_nodes"],
        plan["local_devices"],
        plan["qwen_devices"],
        plan["step_devices"],
    )


if __name__ == "__main__":
    main()
