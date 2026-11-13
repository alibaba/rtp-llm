#!/usr/bin/env python3
"""Read the Prefill arrival/queue trend as timestamped JSON lines.

Two modes: `--url` polls a master HTTP endpoint (`/rtp_llm/debug/prefill_traffic`,
provided by the glm MasterDiagnosticsServer) and `--input` replays snapshot JSON
lines from a file or `-` (stdin), so captured trends can be inspected offline.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def poll(url, interval, samples):
    endpoint = url.rstrip("/") + "/rtp_llm/debug/prefill_traffic"
    count = 0
    while samples == 0 or count < samples:
        start = time.monotonic()
        try:
            with urlopen(endpoint, timeout=5) as response:
                snapshot = json.load(response)
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
        except (URLError, TimeoutError, ValueError) as error:
            print(f"traffic snapshot failed: {error}", file=sys.stderr, flush=True)
            if samples:
                raise SystemExit(1)
        count += 1
        if samples == 0 or count < samples:
            time.sleep(max(0, interval - (time.monotonic() - start)))


def replay(source, interval, samples):
    stream = sys.stdin if source == "-" else Path(source).open()
    count = 0
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                snapshot = json.loads(line)
            except ValueError as error:
                print(f"invalid snapshot line: {error}", file=sys.stderr, flush=True)
                if samples:
                    raise SystemExit(1)
                continue
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            count += 1
            if samples and count >= samples:
                break
            if interval:
                time.sleep(interval)
    finally:
        if stream is not sys.stdin:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Master HTTP base URL with the trend endpoint")
    source.add_argument(
        "--input",
        help="file of snapshot JSON lines, or '-' for stdin (offline replay)",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--samples", type=int, default=0, help="0 keeps watching until Ctrl-C"
    )
    args = parser.parse_args()
    if args.interval < 0 or args.samples < 0:
        parser.error("interval must be nonnegative and samples nonnegative")
    if args.url:
        if args.interval <= 0:
            parser.error("url mode requires a positive interval")
        poll(args.url, args.interval, args.samples)
    else:
        replay(args.input, args.interval, args.samples)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
