#!/usr/bin/env python3
"""Compare fixed-rate replay candidates on verified residual-off cycles."""

import argparse
import concurrent.futures
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile


DECODE_MODES = {
    "off": None,
    "off_explicit": "parameter_rates:17=0,18=0",
    "count": "parameter_rates:17=1,18=0",
    "tokens": "parameter_rates:17=0,18=1",
    "both": "parameter_rates:17=1,18=1",
}


def summarize(rows):
    if not rows:
        return None
    errors = [prediction - actual for _, _, actual, prediction, _ in rows]
    by_bucket = {}
    for host, bucket, actual, prediction, _ in rows:
        by_bucket.setdefault((host, bucket), []).append(prediction - actual)
    bucket_means = [sum(values) / len(values) for values in by_bucket.values()
                    if len(values) >= 2]
    return {
        "count": len(rows),
        "mae_ms": sum(abs(error) for error in errors) / len(errors),
        "bias_prediction_minus_actual_ms": sum(errors) / len(errors),
        "mean_over_ms": sum(max(error, 0) for error in errors) / len(errors),
        "mean_under_ms": sum(max(-error, 0) for error in errors) / len(errors),
        "max_host_bucket_over_ms": max(bucket_means, default=0),
        "max_host_bucket_under_ms": -min(bucket_means, default=0),
    }


def evaluate(binary, data, windows, candidate, temp_dir,
             mode="counterfactual_fixed"):
    weight, scale, decode_mode = candidate
    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".tsv") as output:
        command = [str(binary), str(data), output.name, str(weight), "20261008",
                   str(scale), "0", "all", "0", mode]
        if DECODE_MODES[decode_mode]:
            command.append(DECODE_MODES[decode_mode])
        run = subprocess.run(command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        if run.returncode:
            return {"weight": weight, "scale": scale, "decode": decode_mode,
                    "error": run.stderr.strip()[-1000:],
                    "exit_code": run.returncode}
        groups = {name: [] for name, _, _ in windows}
        phases = {name: {phase: [] for phase in ("rising", "peak", "falling")}
                  for name, _, _ in windows}
        with open(output.name, encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("\t")
                if fields[0] != "P":
                    continue
                if len(fields) < 13:
                    raise RuntimeError("malformed replay prediction for {}: {!r}".format(
                        candidate, line[:200]))
                actual, prediction = float(fields[6]), float(fields[9])
                if not math.isfinite(actual) or not math.isfinite(prediction):
                    return {"weight": weight, "scale": scale, "decode": decode_mode,
                            "error": "non-finite replay prediction"}
                ready_us = int(fields[5]) - round(actual * 1000)
                for name, begin, end in windows:
                    if not begin <= ready_us < end:
                        continue
                    second = (ready_us - begin) / 1_000_000
                    row = (fields[1], int(second // 10), actual, prediction,
                           abs(prediction - float(fields[12])))
                    groups[name].append(row)
                    phase = "rising" if second < 120 else "peak" if second < 180 else "falling"
                    phases[name][phase].append(row)
                    break
    all_rows = list(itertools.chain.from_iterable(groups.values()))
    baseline_differences = [row[4] for row in all_rows]
    window_summaries = {
        name: {"all": summarize(groups[name]),
               **{phase: summarize(rows) for phase, rows in phases[name].items()}}
        for name, _, _ in windows
    }
    all_summary = summarize(all_rows)
    for key in ("max_host_bucket_over_ms", "max_host_bucket_under_ms"):
        all_summary[key] = max(window["all"][key]
                               for window in window_summaries.values())
    return {
        "weight": weight,
        "scale": scale,
        "decode": decode_mode,
        "all": all_summary,
        "windows": window_summaries,
        "online_prediction_difference_ms": {
            "mean": sum(baseline_differences) / len(baseline_differences),
            "max": max(baseline_differences),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", action="append", required=True,
                        help="NAME:BEGIN_UNIX_SECONDS:END_UNIX_SECONDS")
    parser.add_argument("--weights", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4])
    parser.add_argument("--decode-modes", nargs="+", choices=DECODE_MODES,
                        default=list(DECODE_MODES))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-online-diff-ms", type=float, default=20)
    args = parser.parse_args()
    windows = []
    for value in args.window:
        name, begin, end = value.split(":")
        begin_us, end_us = int(begin) * 1_000_000, int(end) * 1_000_000
        if end_us - begin_us != 300_000_000:
            parser.error("each window must contain one 300-second cycle")
        windows.append((name, begin_us, end_us))
    if (args.workers < 1 or not math.isfinite(args.max_online_diff_ms) or
            args.max_online_diff_ms < 0 or any(not math.isfinite(value) or value <= 0
                               for value in args.weights + args.scales)):
        parser.error("workers, weights and scales must be positive")
    decode_modes = list(dict.fromkeys([*args.decode_modes, "off"]))
    candidates = list(itertools.product(args.weights, args.scales, decode_modes))
    with tempfile.TemporaryDirectory(prefix="navi-off-grid-") as temp_dir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(
                lambda candidate: evaluate(args.binary, args.data, windows,
                                           candidate, temp_dir), candidates))
        control = evaluate(args.binary, args.data, windows,
                           (4, 1, "off_explicit"), temp_dir)
    baseline = next(row for row in rows if row["weight"] == 4 and
                    row["scale"] == 1 and row["decode"] == "off")
    if (baseline["online_prediction_difference_ms"]["max"] > args.max_online_diff_ms or
            baseline["all"]["count"] == 0 or
            baseline["all"]["count"] != control["all"]["count"] or
            abs(baseline["all"]["mae_ms"] - control["all"]["mae_ms"]) > 1e-6):
        raise RuntimeError("baseline replay or disabled-decode control mismatch")
    rows.sort(key=lambda row: row["all"]["mae_ms"] if "all" in row else math.inf)
    args.output.write_text(json.dumps({"windows": windows, "baseline": baseline,
                                       "results": rows}, indent=2) + "\n")
    print("baseline", json.dumps(baseline["all"]))
    print("failed candidates", sum("error" in row for row in rows))
    for row in rows[:15]:
        print(row["weight"], row["scale"], row["decode"],
              round(row["all"]["mae_ms"], 1),
              round(sum(window["falling"]["mean_over_ms"]
                        for window in row["windows"].values()) / len(windows), 1))


if __name__ == "__main__":
    main()
