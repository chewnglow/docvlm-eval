#!/usr/bin/env python3
"""Wait until an OpenAI-compatible server exposes /v1/models."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--interval", type=float, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadline = time.time() + args.timeout
    url = args.base_url.rstrip("/") + "/models"
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "data" in payload:
                print(f"ready: {args.base_url}")
                return
            last_error = f"unexpected response: {payload!r}"
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(args.interval)
    raise SystemExit(f"Timed out waiting for {args.base_url}: {last_error}")


if __name__ == "__main__":
    main()

