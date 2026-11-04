#!/usr/bin/env python3
"""Drive an existing fixed-QPS replay without owning its stop lifecycle."""

import argparse
import json
import math
import signal
import subprocess
import sys
import time


def target_qps(elapsed, minimum, maximum, period):
    midpoint = (minimum + maximum) / 2.0
    amplitude = (maximum - minimum) / 2.0
    return midpoint - amplitude * math.cos(2.0 * math.pi * elapsed / period)


def run_command(tool, *arguments):
    result = subprocess.run(
        [tool, *arguments], capture_output=True, text=True, timeout=15
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    response = json.loads(result.stdout)
    if not response.get("accepted"):
        raise RuntimeError(response.get("error", "replay command was rejected"))
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", default="/usr/local/bin/navi_sched_replay")
    parser.add_argument("--min-qps", type=float, required=True)
    parser.add_argument("--max-qps", type=float, required=True)
    parser.add_argument("--period-seconds", type=float, required=True)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--elapsed-offset-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--stop-on-exit", action="store_true")
    args = parser.parse_args()
    if not 0 < args.min_qps <= args.max_qps:
        parser.error("require 0 < min-qps <= max-qps")
    if args.period_seconds <= 0 or args.step_seconds <= 0:
        parser.error("period-seconds and step-seconds must be positive")
    if args.elapsed_offset_seconds < 0:
        parser.error("elapsed-offset-seconds must not be negative")
    if args.duration_seconds < 0:
        parser.error("duration-seconds must not be negative")

    state = run_command(args.tool, "query")
    if state.get("action") != "start" or state.get("mode") != "fixed_qps":
        parser.error("an active fixed_qps replay is required")

    stopping = False

    def request_exit(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_exit)
    signal.signal(signal.SIGTERM, request_exit)
    origin = time.monotonic() - args.elapsed_offset_seconds
    tick = math.floor(args.elapsed_offset_seconds / args.step_seconds) + 1
    print("attached to active replay; signal exits driver without stopping replay", flush=True)
    while not stopping:
        deadline = origin + tick * args.step_seconds
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        if stopping:
            break
        elapsed = tick * args.step_seconds
        if args.duration_seconds and elapsed >= args.duration_seconds:
            break
        qps = target_qps(elapsed, args.min_qps, args.max_qps, args.period_seconds)
        try:
            run_command(args.tool, "update", "--total-qps", "%.6f" % qps)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            print("replay QPS update failed: %s" % error, file=sys.stderr, flush=True)
        tick = max(tick + 1, math.floor((time.monotonic() - origin) / args.step_seconds) + 1)
    if args.stop_on_exit:
        run_command(args.tool, "stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
