#!/usr/bin/env python3
"""Coarse-search residual-off controllers, then check finalists on both GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from AblateLatencyOnlineRates import evaluate


WINDOWS = {
    "mi": (1791899400000000, 1791903600000000),
    "rtx": (1791795000000000, 1791839100000000),
}


def run(binary, data, window, weight, mode, temp_dir):
    result = evaluate(binary, data, *window, 300, weight, 2.55, 0.0,
                      temp_dir, mode=mode)
    return {key: result["ALL"][key] for key in
            ("count", "mae_ms", "bias_ms", "abs_p95_ms", "under_p95_ms",
             "over_p95_ms")}


def candidates(family):
    yield ("fixed", 80, "counterfactual_fixed")
    if family == "decode":
        for weight, count_rate, token_rate in itertools.product(
                (16, 80, 320), (0, 1, 4), (0, 1, 4)):
            if count_rate == token_rate == 0:
                continue
            mode = "parameter_rates:17={},18={}".format(
                count_rate, token_rate)
            yield ("decode", weight, mode)
        return
    for weight, gain, ceiling, threshold in itertools.product(
            (16, 80, 320), (4, 16, 64), (8, 32), (0.05, 0.2)):
        mode = ("production_error_rate_hinge:1,{},{},0.03,{}"
                .format(gain, ceiling, threshold))
        yield ("error_hinge", weight, mode)
    for weight, gain, ceiling in itertools.product(
            (16, 80, 320), (4, 16, 64), (8, 32)):
        mode = "production_error_rate:1,{},{},0.03".format(gain, ceiling)
        yield ("error_rate", weight, mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("mi", type=Path)
    parser.add_argument("rtx", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=12)
    parser.add_argument("--family", choices=("controller", "decode"),
                        default="controller")
    args = parser.parse_args()
    if args.workers < 1 or args.finalists < 1:
        parser.error("workers and finalists must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    with TemporaryDirectory(dir=args.output.parent) as temp_dir:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(run, args.binary, args.mi, WINDOWS["mi"],
                                 weight, mode, temp_dir): (name, weight, mode)
                     for name, weight, mode in candidates(args.family)}
            for task in as_completed(tasks):
                name, weight, mode = tasks[task]
                try:
                    row = {"strategy": name, "weight": weight,
                           "mode": mode, "mi": task.result()}
                    rows.append(row)
                    print(name, weight, mode, row["mi"]["mae_ms"], flush=True)
                except Exception as error:
                    failures.append({"mode": mode, "weight": weight,
                                     "error": str(error)})
        rows.sort(key=lambda row: row["mi"]["mae_ms"])
        baseline = next((row for row in rows if row["strategy"] == "fixed"), None)
        if baseline is None:
            raise RuntimeError("fixed baseline failed: {}".format(failures))
        finalists = rows[:args.finalists]
        if baseline not in finalists:
            finalists.append(baseline)
        with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
            tasks = {pool.submit(run, args.binary, args.rtx, WINDOWS["rtx"],
                                 row["weight"], row["mode"], temp_dir): row
                     for row in finalists}
            for task in as_completed(tasks):
                row = tasks[task]
                try:
                    row["rtx"] = task.result()
                    print("rtx", row["strategy"], row["weight"],
                          row["rtx"]["mae_ms"], flush=True)
                except Exception as error:
                    failures.append({"mode": row["mode"],
                                     "weight": row["weight"],
                                     "hardware": "rtx", "error": str(error)})
    for row in rows:
        if row["mi"]["count"] != baseline["mi"]["count"]:
            raise RuntimeError("MI sample count changed across candidates")
        if "rtx" in row and row["rtx"]["count"] != baseline["rtx"]["count"]:
            raise RuntimeError("RTX sample count changed across candidates")
    args.output.write_text(json.dumps({"family": args.family, "windows": WINDOWS,
                                      "results": rows,
                                      "failures": failures}, indent=2) + "\n")


if __name__ == "__main__":
    main()
