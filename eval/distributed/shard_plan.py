#!/usr/bin/env python3
"""Print offset/limit for contiguous sharding."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit("--shard-id must be in [0, num-shards)")
    base = args.total // args.num_shards
    rem = args.total % args.num_shards
    offset = args.shard_id * base + min(args.shard_id, rem)
    limit = base + (1 if args.shard_id < rem else 0)
    print(f"{offset} {limit}")


if __name__ == "__main__":
    main()

