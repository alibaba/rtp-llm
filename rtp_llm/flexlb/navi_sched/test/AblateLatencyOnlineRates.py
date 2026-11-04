#!/usr/bin/env python3
"""Compare fixed online-rate candidates on an exactly replayable log window."""

import argparse
import concurrent.futures
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def summarize(rows):
    errors = [prediction - target for target, prediction in rows]
    over = [error for error in errors if error > 0]
    under = [-error for error in errors if error < 0]
    return {
        "count": len(rows),
        "bias_ms": sum(errors) / len(errors),
        "mae_ms": sum(map(abs, errors)) / len(errors),
        "wape": sum(map(abs, errors)) / sum(target for target, _ in rows),
        "abs_p95_ms": percentile(list(map(abs, errors)), 0.95),
        "over_p95_ms": percentile(over, 0.95),
        "under_p95_ms": percentile(under, 0.95),
        "over_count": len(over),
        "under_count": len(under),
    }


def build_replay_command(binary, data, output, weight, scale, residual,
                         mode="counterfactual_fixed"):
    return [
        str(binary),
        str(data),
        str(output),
        str(weight),
        "20261008",
        str(scale),
        str(residual),
        "all",
        "0",
        mode,
    ]


def cycle_index(event_time_us, start_us, cycle_seconds):
    return (event_time_us - start_us) // (cycle_seconds * 1_000_000)


def score_time_us(event_time_us, target_ms, basis):
    if basis == "ready":
        return event_time_us - round(target_ms * 1000.0)
    if basis == "event":
        return event_time_us
    raise ValueError(f"invalid score time basis: {basis}")


def classify_architecture(data, host):
    if data.name == "mi":
        return "MI"
    if data.name in ("rtx", "rtx_stable"):
        return "RTX"
    if host.startswith("mi-") or "mi308" in host:
        return "MI"
    if host.startswith("rtx-") or "rtx5000" in host:
        return "RTX"
    raise ValueError(f"hardware is unknown for host {host}; use a grouped input directory")


def evaluate(binary, data, start_us, end_us, cycle_seconds, weight, scale,
             residual, temp_dir, score_time_basis="ready",
             mode="counterfactual_fixed"):
    cycle_count = math.ceil(
        (end_us - start_us) / (cycle_seconds * 1_000_000)
    )
    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".tsv") as output:
        subprocess.run(
            build_replay_command(
                binary, data, output.name, weight, scale, residual, mode
            ),
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        groups = {"ALL": [], "MI": [], "RTX": []}
        cycles = {
            name: [[] for _ in range(cycle_count)] for name in groups
        }
        hosts = {}
        host_cycles = {}
        with open(output.name, encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("\t")
                if fields[0] != "P":
                    continue
                event_time = int(fields[5])
                target_ms = float(fields[6])
                score_time = score_time_us(
                    event_time, target_ms, score_time_basis
                )
                if not start_us <= score_time < end_us:
                    continue
                host = fields[1].lower()
                arch = classify_architecture(data, host)
                row = (target_ms, float(fields[9]))
                groups["ALL"].append(row)
                groups[arch].append(row)
                hosts.setdefault(host, []).append(row)
                host_cycles.setdefault(
                    host, [[] for _ in range(cycle_count)]
                )
                cycle = cycle_index(score_time, start_us, cycle_seconds)
                if 0 <= cycle < cycle_count:
                    cycles["ALL"][cycle].append(row)
                    cycles[arch][cycle].append(row)
                    host_cycles[host][cycle].append(row)
    result = {arch: summarize(rows) for arch, rows in groups.items() if rows}
    result["score_time_basis"] = score_time_basis
    result["cycles"] = {
        arch: [summarize(rows) if rows else None for rows in chunks]
        for arch, chunks in cycles.items()
    }
    result["hosts"] = {
        host: {
            "all": summarize(rows),
            "cycles": [summarize(chunk) if chunk else None
                       for chunk in host_cycles[host]],
        }
        for host, rows in sorted(hosts.items())
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--end-us", type=int, required=True)
    parser.add_argument("--cycle-seconds", type=int, default=300)
    parser.add_argument("--score-time-basis", choices=("ready", "event"),
                        default="ready")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--weights", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6])
    parser.add_argument("--residuals", type=float, nargs="+",
                        default=[0.03, 0.08, 0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--baseline-weight", type=float, default=4.0)
    parser.add_argument("--baseline-scale", type=float, default=1.0)
    parser.add_argument("--baseline-residual", type=float, default=0.8)
    args = parser.parse_args()
    if (args.end_us <= args.start_us or args.workers < 1 or
            args.cycle_seconds < 1):
        parser.error("invalid time window, cycle duration, or worker count")

    candidates = list(itertools.product(args.weights, args.scales, args.residuals))
    baseline_candidate = (args.baseline_weight, args.baseline_scale,
                          args.baseline_residual)
    if baseline_candidate not in candidates:
        parser.error("grid must contain the verified baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp_dir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(evaluate, args.binary, args.data, args.start_us,
                            args.end_us, args.cycle_seconds, *candidate,
                            temp_dir, args.score_time_basis): candidate
                for candidate in candidates
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                weight, scale, residual = futures[future]
                results.append({"weight": weight, "scale": scale,
                                "residual": residual, **future.result()})
    results.sort(key=lambda row: (row.get("MI", row["ALL"])["mae_ms"],
                                  row.get("MI", row["ALL"])["abs_p95_ms"]))
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    baseline = next(row for row in results if (row["weight"], row["scale"],
                                                row["residual"]) == baseline_candidate)
    print("baseline", json.dumps(baseline, ensure_ascii=False))
    eligible = [row for row in results
                if "RTX" not in baseline or
                (row["RTX"]["mae_ms"] <= baseline["RTX"]["mae_ms"]
                 and row["RTX"]["abs_p95_ms"] <= baseline["RTX"]["abs_p95_ms"])]
    print("eligible", len(eligible), "of", len(results))
    for row in eligible[:15]:
        print(row["weight"], row["scale"], row["residual"],
              *[(arch, row[arch]["mae_ms"], row[arch]["abs_p95_ms"],
                 row[arch]["over_p95_ms"], row[arch]["under_p95_ms"])
                for arch in ("MI", "RTX") if arch in row])


if __name__ == "__main__":
    main()
