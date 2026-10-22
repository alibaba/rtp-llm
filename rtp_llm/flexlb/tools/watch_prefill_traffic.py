#!/usr/bin/env python3
"""Read the live Prefill arrival/queue trend endpoint as timestamped JSON lines."""

import argparse
import json
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Master HTTP base URL")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--samples", type=int, default=0, help="0 keeps watching until Ctrl-C"
    )
    args = parser.parse_args()
    if args.interval <= 0 or args.samples < 0:
        parser.error("interval must be positive and samples nonnegative")
    endpoint = args.url.rstrip("/") + "/rtp_llm/debug/prefill_traffic"
    count = 0
    while args.samples == 0 or count < args.samples:
        start = time.monotonic()
        try:
            with urlopen(endpoint, timeout=5) as response:
                snapshot = json.load(response)
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
        except (URLError, TimeoutError, ValueError) as error:
            print(f"traffic snapshot failed: {error}", file=sys.stderr, flush=True)
            if args.samples:
                raise SystemExit(1)
        count += 1
        if args.samples == 0 or count < args.samples:
            time.sleep(max(0, args.interval - (time.monotonic() - start)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
