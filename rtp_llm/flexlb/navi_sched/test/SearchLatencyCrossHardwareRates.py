#!/usr/bin/env python3
"""Search fixed residual-off rates on separate 308 and 5000 replay inputs."""

import argparse
import concurrent.futures
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from AblateLatencyOnlineRates import evaluate


def replay_pair(args, candidate, temp_dir):
    weight, scale = candidate
    result = {"weight": weight, "scale": scale}
    for name in ("mi", "rtx"):
        result[name] = evaluate(
            args.binary, getattr(args, name),
            getattr(args, name + "_start_us"),
            getattr(args, name + "_end_us"),
            args.cycle_seconds, weight, scale, 0.0, temp_dir,
        )
    return result


def score(row, baseline):
    return sum(row[name]["ALL"]["mae_ms"] /
               baseline[name]["ALL"]["mae_ms"] for name in ("mi", "rtx")) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("mi", type=Path)
    parser.add_argument("rtx", type=Path)
    parser.add_argument("output", type=Path)
    for name in ("mi", "rtx"):
        parser.add_argument("--" + name + "-start-us", type=int, required=True)
        parser.add_argument("--" + name + "-end-us", type=int, required=True)
    parser.add_argument("--cycle-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--baseline-weight", type=float, default=64)
    parser.add_argument("--baseline-scale", type=float, default=2.75)
    args = parser.parse_args()
    if (args.workers < 1 or args.cycle_seconds < 1 or
            args.mi_end_us <= args.mi_start_us or
            args.rtx_end_us <= args.rtx_start_us):
        parser.error("invalid worker count, cycle duration, or time window")
    candidates = list(itertools.product(args.weights, args.scales))
    baseline_candidate = (args.baseline_weight, args.baseline_scale)
    if baseline_candidate not in candidates:
        parser.error("grid must contain the baseline")
    if any(not math.isfinite(value) or value <= 0
           for candidate in candidates for value in candidate):
        parser.error("rates and weights must be finite and positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp_dir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(replay_pair, args, candidate, temp_dir): candidate
                       for candidate in candidates}
            for future in concurrent.futures.as_completed(futures):
                candidate = futures[future]
                try:
                    row = future.result()
                    results.append(row)
                    print("done", *candidate, row["mi"]["ALL"]["mae_ms"],
                          row["rtx"]["ALL"]["mae_ms"], flush=True)
                except Exception as error:
                    detail = (error.stderr if isinstance(error, subprocess.CalledProcessError)
                              else str(error))
                    failures.append({"weight": candidate[0], "scale": candidate[1],
                                     "error": detail})
                    print("failed", *candidate, detail, flush=True)
    baseline = next((row for row in results if
                     (row["weight"], row["scale"]) == baseline_candidate), None)
    if baseline is None:
        raise RuntimeError("baseline replay failed")
    for row in results:
        for name in ("mi", "rtx"):
            if row[name]["ALL"]["count"] != baseline[name]["ALL"]["count"]:
                raise RuntimeError("candidate changed sample count: " + str(row))
        row["relative_mae"] = score(row, baseline)
    results.sort(key=lambda row: row["relative_mae"])
    args.output.write_text(json.dumps({"baseline": baseline_candidate,
                                       "results": results, "failures": failures},
                                      indent=2) + "\n", encoding="utf-8")
    for row in results[:15]:
        print("rank", row["weight"], row["scale"], row["relative_mae"],
              row["mi"]["ALL"]["mae_ms"], row["rtx"]["ALL"]["mae_ms"])


if __name__ == "__main__":
    main()
