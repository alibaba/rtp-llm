#!/usr/bin/env python3
"""Summarize exact online parameter paths by replay cycle and hardware."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from AnalyzeLatencyAdamScale import (
    PARAMETER_COUNT,
    PARAMETER_SCALES,
    load_host_architecture,
    parse_trace_line,
)


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def analyze(trace_path, host_architecture, start_us, period_us, cycle_count):
    records = defaultdict(list)
    with trace_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.startswith("D\t"):
                continue
            try:
                record = parse_trace_line(line)
            except ValueError as error:
                raise ValueError(f"{trace_path}:{line_number}: {error}") from error
            architecture = host_architecture.get(record["host"])
            if architecture is None:
                raise ValueError(
                    f"trace host is absent from manifest: {record['host']}")
            ready_time_us = record["event_time_us"] - round(
                record["target_ms"] * 1000.0)
            cycle = (ready_time_us - start_us) // period_us
            if cycle < 0 or cycle >= cycle_count:
                continue
            records[(architecture, record["host"], int(cycle))].append(
                (ready_time_us, record))

    host_paths = []
    scales = np.asarray(PARAMETER_SCALES)
    for (architecture, host, cycle), entries in sorted(records.items()):
        entries.sort(key=lambda item: item[0])
        before = np.asarray(entries[0][1]["values"]["parameter_before"])
        after = np.asarray(entries[-1][1]["values"]["parameter_after"])
        all_before = np.asarray([
            entry[1]["values"]["parameter_before"] for entry in entries
        ])
        all_updates = np.asarray([
            entry[1]["values"]["actual_update"] for entry in entries
        ])
        effective_before = before * scales
        effective_after = after * scales
        effective_mean_abs = np.mean(np.abs(all_before * scales), axis=0)
        effective_net = effective_after - effective_before
        effective_variation = np.sum(
            np.abs(all_updates * scales), axis=0)
        host_paths.append({
            "architecture": architecture,
            "host": host,
            "cycle": cycle,
            "update_count": len(entries),
            "parameters": [
                {
                    "index": index,
                    "effective_start": float(effective_before[index]),
                    "effective_end": float(effective_after[index]),
                    "effective_abs_mean": float(effective_mean_abs[index]),
                    "effective_net_change": float(effective_net[index]),
                    "effective_total_variation": float(
                        effective_variation[index]),
                    "absolute_net_over_abs_mean": float(
                        abs(effective_net[index]) /
                        max(effective_mean_abs[index], 1e-12)),
                    "absolute_net_over_total_variation": float(
                        abs(effective_net[index]) /
                        max(effective_variation[index], 1e-12)),
                }
                for index in range(PARAMETER_COUNT)
            ],
        })

    architectures = {}
    for architecture in sorted(set(host_architecture.values())):
        cycles = {}
        for cycle in range(cycle_count):
            selected = [path for path in host_paths
                        if path["architecture"] == architecture and
                        path["cycle"] == cycle]
            if not selected:
                continue
            cycles[str(cycle)] = {
                "host_count": len(selected),
                "update_count": sum(path["update_count"] for path in selected),
                "parameters": [
                    {
                        "index": index,
                        "effective_abs_mean": summarize([
                            path["parameters"][index]["effective_abs_mean"]
                            for path in selected
                        ]),
                        "absolute_effective_net_change": summarize([
                            abs(path["parameters"][index]["effective_net_change"])
                            for path in selected
                        ]),
                        "effective_total_variation": summarize([
                            path["parameters"][index]["effective_total_variation"]
                            for path in selected
                        ]),
                        "absolute_net_over_abs_mean": summarize([
                            path["parameters"][index]["absolute_net_over_abs_mean"]
                            for path in selected
                        ]),
                        "absolute_net_over_total_variation": summarize([
                            path["parameters"][index]["absolute_net_over_total_variation"]
                            for path in selected
                        ]),
                    }
                    for index in range(PARAMETER_COUNT)
                ],
            }
        architectures[architecture] = {"cycles": cycles}
    return {
        "start_us": start_us,
        "period_us": period_us,
        "cycle_count": cycle_count,
        "architectures": architectures,
        "host_paths": host_paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--period-seconds", type=float, default=300.0)
    parser.add_argument("--cycle-count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (not math.isfinite(args.period_seconds) or
            args.period_seconds <= 0.0 or args.cycle_count <= 0):
        parser.error("invalid period or cycle count")
    result = analyze(
        args.trace,
        load_host_architecture(args.manifest),
        args.start_us,
        round(args.period_seconds * 1_000_000),
        args.cycle_count,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
