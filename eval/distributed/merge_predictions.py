#!/usr/bin/env python3
"""Merge shard JSONL files in shard-id order, de-duplicating by id."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def shard_key(path: Path) -> tuple[int, str]:
    match = re.search(r"shard-(\d+)-of-", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def main() -> None:
    args = parse_args()
    shard_dir = Path(args.shard_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    kept = 0
    with output.open("w") as out:
      for shard in sorted(shard_dir.glob("shard-*-of-*.jsonl"), key=shard_key):
          with shard.open() as f:
              for line in f:
                  if not line.strip():
                      continue
                  total += 1
                  row = json.loads(line)
                  key = str(row.get("id", f"__missing__{total}"))
                  if key in seen:
                      continue
                  seen.add(key)
                  kept += 1
                  out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"merged {kept} unique rows from {total} shard rows -> {output}")


if __name__ == "__main__":
    main()

