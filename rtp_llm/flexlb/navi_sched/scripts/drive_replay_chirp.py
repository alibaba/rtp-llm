#!/usr/bin/env python3
"""Run a bounded NaviSched replay with warmup and a continuous frequency sweep."""

import argparse
import json
import math
import signal
import subprocess
import sys
import time


# The deployed fixed-QPS control endpoint requires a strictly positive rate.
ZERO_QPS_FLOOR = 0.000001


def control_qps(target_qps):
    return max(ZERO_QPS_FLOOR, target_qps)


def wave(elapsed, minimum, maximum, base_period, fast_period, warmup_cycles,
         sweep_duration):
    warmup_duration = warmup_cycles * base_period
    slow_frequency = 1.0 / base_period
    fast_frequency = 1.0 / fast_period
    half_sweep = sweep_duration / 2.0
    if elapsed <= warmup_duration:
        cycles = elapsed * slow_frequency
        frequency = slow_frequency
        stage = "warmup"
    else:
        sweep_time = min(elapsed - warmup_duration, sweep_duration)
        slope = (fast_frequency - slow_frequency) / half_sweep
        if sweep_time <= half_sweep:
            cycles = (warmup_cycles + slow_frequency * sweep_time +
                      0.5 * slope * sweep_time * sweep_time)
            frequency = slow_frequency + slope * sweep_time
            stage = "frequency_up"
        else:
            remaining = sweep_time - half_sweep
            cycles = (warmup_cycles + (slow_frequency + fast_frequency) *
                      half_sweep / 2.0 + fast_frequency * remaining -
                      0.5 * slope * remaining * remaining)
            frequency = fast_frequency - slope * remaining
            stage = "frequency_down"
    midpoint = (minimum + maximum) / 2.0
    amplitude = (maximum - minimum) / 2.0
    qps = midpoint - amplitude * math.cos(2.0 * math.pi * cycles)
    return min(maximum, max(minimum, qps)), 1.0 / frequency, stage


def run_command(tool, *arguments):
    result = subprocess.run([tool, *arguments], capture_output=True, text=True,
                            timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    response = json.loads(result.stdout)
    if not response.get("accepted"):
        raise RuntimeError(response.get("error", "replay command rejected"))
    return response


def history_arguments(start_time_us, end_time_us, recent_query_count):
    if start_time_us is not None or end_time_us is not None:
        if (start_time_us is None or end_time_us is None or
                start_time_us <= 0 or end_time_us <= start_time_us):
            raise ValueError("require 0 < start-time-us < end-time-us")
        if recent_query_count is not None:
            raise ValueError("time range cannot be combined with recent-query-count")
        return ["--history-mode", "time_range", "--start-time-us", str(start_time_us),
                "--end-time-us", str(end_time_us)]
    count = 1000 if recent_query_count is None else recent_query_count
    if count <= 0:
        raise ValueError("recent-query-count must be positive")
    return ["--history-mode", "recent_count", "--recent-query-count", str(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", default="/usr/local/bin/navi_sched_replay")
    parser.add_argument("--min-qps", type=float, default=0.0)
    parser.add_argument("--max-qps", type=float, default=400.0)
    parser.add_argument("--base-period-seconds", type=float, default=300.0)
    parser.add_argument("--fast-period-seconds", type=float, default=45.0)
    parser.add_argument("--warmup-cycles", type=int, default=3)
    parser.add_argument("--sweep-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--recent-query-count", type=int)
    parser.add_argument("--start-time-us", type=int)
    parser.add_argument("--end-time-us", type=int)
    parser.add_argument("--drain-seconds", type=float, default=60.0)
    parser.add_argument("--step-hold-seconds", type=float, default=300.0)
    args = parser.parse_args()
    try:
        selection = history_arguments(args.start_time_us, args.end_time_us,
                                      args.recent_query_count)
    except ValueError as error:
        parser.error(str(error))
    if not (math.isfinite(args.min_qps) and math.isfinite(args.max_qps) and
            0 <= args.min_qps < args.max_qps):
        parser.error("require 0 <= min-qps < max-qps")
    if not (math.isfinite(args.base_period_seconds) and
            math.isfinite(args.fast_period_seconds) and
            math.isfinite(args.sweep_duration_seconds) and
            math.isfinite(args.step_seconds) and
            0 < args.fast_period_seconds < args.base_period_seconds and
            args.sweep_duration_seconds > 0 and args.step_seconds > 0 and
            args.warmup_cycles >= 1 and
            args.drain_seconds > 0 and args.step_hold_seconds > 0):
        parser.error("invalid periods, duration, step, warmup cycles, or history count")

    total_duration = (args.warmup_cycles * args.base_period_seconds +
                      args.sweep_duration_seconds)
    control_duration = (total_duration + args.drain_seconds +
                        2 * args.step_hold_seconds + 30.0)
    state = run_command(args.tool, "query")
    if state.get("action") == "start":
        parser.error("another replay is active")

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = False
    try:
        initial_qps, _, _ = wave(0.0, args.min_qps, args.max_qps,
                                 args.base_period_seconds, args.fast_period_seconds,
                                 args.warmup_cycles, args.sweep_duration_seconds)
        response = run_command(
            args.tool, "start", "--mode", "fixed_qps", *selection,
            "--total-qps", "%.6f" % control_qps(initial_qps), "--loop-count", "0",
            "--duration-ms", str(math.ceil(control_duration * 1000.0)))
        started = True
        if response.get("eligibleClientCount", 0) < 1:
            raise RuntimeError("no replayable clients")
        origin = time.monotonic()
        print(json.dumps({"event": "started", "wall_time_us": time.time_ns() // 1000,
                          "warmup_seconds": args.warmup_cycles * args.base_period_seconds,
                          "sweep_seconds": args.sweep_duration_seconds,
                          "min_qps": args.min_qps, "max_qps": args.max_qps,
                          "fast_period_seconds": args.fast_period_seconds,
                          "drain_seconds": args.drain_seconds,
                          "step_hold_seconds": args.step_hold_seconds,
                          "history_selection": selection,
                          "eligible_clients": response["eligibleClientCount"]}), flush=True)
        tick = 1
        while not stopping:
            deadline = origin + tick * args.step_seconds
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            if stopping:
                break
            elapsed = time.monotonic() - origin
            if elapsed >= total_duration:
                break
            qps, period, stage = wave(
                elapsed, args.min_qps, args.max_qps, args.base_period_seconds,
                args.fast_period_seconds, args.warmup_cycles,
                args.sweep_duration_seconds)
            run_command(args.tool, "update", "--total-qps",
                        "%.6f" % control_qps(qps))
            print(json.dumps({"event": "point", "wall_time_us": time.time_ns() // 1000,
                              "elapsed_seconds": round(elapsed, 6),
                              "stage": stage, "period_seconds": round(period, 6),
                              "qps": round(qps, 6),
                              "lag_ms": round(max(0.0, elapsed - tick * args.step_seconds)
                                              * 1000.0, 3)}), flush=True)
            tick = max(tick + 1, math.floor((time.monotonic() - origin) /
                                            args.step_seconds) + 1)
        if stopping:
            raise RuntimeError("interrupted")
        run_command(args.tool, "update", "--total-qps",
                    "%.6f" % ZERO_QPS_FLOOR)
        print(json.dumps({"event": "drain_start", "target_qps": 0.0,
                          "control_qps": ZERO_QPS_FLOOR,
                          "wall_time_us": time.time_ns() // 1000}), flush=True)

        def hold(seconds):
            deadline = time.monotonic() + seconds
            while not stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                time.sleep(min(1.0, remaining))
            raise RuntimeError("interrupted")

        hold(args.drain_seconds)
        for qps in (100.0, 50.0):
            run_command(args.tool, "update", "--total-qps", "%.6f" % qps)
            print(json.dumps({"event": "step_start", "qps": qps,
                              "wall_time_us": time.time_ns() // 1000}), flush=True)
            hold(args.step_hold_seconds)
        print(json.dumps({"event": "steps_complete",
                          "wall_time_us": time.time_ns() // 1000}), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"event": "failed", "error": str(error),
                          "wall_time_us": time.time_ns() // 1000}),
              file=sys.stderr, flush=True)
        return 1
    finally:
        if started:
            try:
                run_command(args.tool, "stop")
                print(json.dumps({"event": "stopped",
                                  "wall_time_us": time.time_ns() // 1000}), flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                print(json.dumps({"event": "stop_failed", "error": str(error)}),
                      file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
