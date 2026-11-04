#!/usr/bin/env python3
"""Replay a production-controller weight/base-cap/residual-cap grid."""

import argparse
import concurrent.futures
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


EXCLUDED_HOSTS = {"11_225_199_135", "11_225_203_0"}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def summarize(errors, targets):
    positive = [error for error in errors if error > 0]
    return {
        "count": len(errors),
        "mae_ms": sum(map(abs, errors)) / len(errors),
        "bias_ms": sum(errors) / len(errors),
        "wape": sum(map(abs, errors)) / sum(targets),
        "abs_p95_ms": percentile([abs(error) for error in errors], 0.95),
        "over_p95_ms": percentile(positive, 0.95),
        "over_count": len(positive),
    }


def replay(binary, data, candidate, sweep_start_us, sweep_end_us, host_arch,
           excluded_hosts, switch_at_us, temp_dir):
    weight, base_cap, residual_cap, error_ratio = candidate
    mode = (f"production_switch:{switch_at_us},{weight},{base_cap},"
            f"{residual_cap},{error_ratio}"
            if switch_at_us is not None else
            f"production_cap:{base_cap},{residual_cap}")
    initial_weight = 4 if switch_at_us is not None else weight
    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".tsv") as output:
        subprocess.run(
            [str(binary), str(data), output.name, str(initial_weight), "20261008", "16",
             "0.8", "all", "0", mode],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        buckets = {}
        integrity = {}
        with open(output.name, encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\n").split("\t")
                if fields[0] != "P" or fields[1] in excluded_hosts:
                    continue
                target = float(fields[6])
                ready_us = int(fields[5]) - round(target * 1000)
                if not sweep_start_us <= ready_us < sweep_end_us:
                    continue
                host = fields[1]
                arch = host_arch[host]
                error = float(fields[9]) - target
                half = "first_half" if ready_us < (sweep_start_us + sweep_end_us) // 2 else "second_half"
                for key in (arch, host, half, f"{arch}_{half}"):
                    errors, targets = buckets.setdefault(key, ([], []))
                    errors.append(error)
                    targets.append(target)
                if candidate == (4, 192, 1.0, 0.1):
                    online_diff = abs(float(fields[9]) - float(fields[12]))
                    count, total, maximum = integrity.get(host, (0, 0.0, 0.0))
                    integrity[host] = (count + 1, total + online_diff,
                                       max(maximum, online_diff))
    return {
        "weight": weight,
        "base_cap": base_cap,
        "residual_cap": residual_cap,
        "error_ratio": error_ratio,
        "metrics": {key: summarize(*values) for key, values in buckets.items()},
        "online_integrity": {
            host: {"count": count, "mae_ms": total / count, "max_ms": maximum}
            for host, (count, total, maximum) in integrity.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sweep-start-us", type=int, required=True)
    parser.add_argument("--sweep-end-us", type=int, required=True)
    parser.add_argument("--weights", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--base-caps", type=int, nargs="+", default=[96, 128, 192])
    parser.add_argument("--residual-caps", type=float, nargs="+", default=[0.4, 0.8, 1.0])
    parser.add_argument("--residual-error-ratios", type=float, nargs="+", default=[0.1])
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--switch-at-us", type=int)
    parser.add_argument("--include-all-hosts", action="store_true")
    args = parser.parse_args()
    if (args.sweep_end_us <= args.sweep_start_us or args.workers < 1 or
            any(not math.isfinite(value) or value < 0.0
                for value in args.residual_error_ratios)):
        parser.error("invalid sweep interval or worker count")
    if args.switch_at_us is None and args.residual_error_ratios != [0.1]:
        parser.error("residual error ratio changes require --switch-at-us")
    if args.include_all_hosts and args.switch_at_us is None:
        parser.error("including all hosts requires an anchored production switch")
    if args.switch_at_us is not None and args.switch_at_us != args.sweep_start_us:
        parser.error("production switch must match the sweep start")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    host_arch = {}
    for node in manifest["nodes"]:
        host = node["ip"].replace(".", "_")
        deployment = node["deployment"].upper()
        if "MI308" in deployment:
            host_arch[host] = "MI"
        elif "RTX_PRO_5000" in deployment:
            host_arch[host] = "RTX"
        else:
            parser.error("unknown deployment for host {}: {}".format(host, deployment))
    if len(host_arch) != 20 or sum(arch == "MI" for arch in host_arch.values()) != 10:
        parser.error("expected 10 MI and 10 RTX hosts in manifest")
    candidates = list(itertools.product(args.weights, args.base_caps,
                                        args.residual_caps,
                                        args.residual_error_ratios))
    if (4, 192, 1.0, 0.1) not in candidates:
        parser.error("grid must include the production baseline (4, 192, 1.0, 0.1)")
    excluded_hosts = set() if args.include_all_hosts else EXCLUDED_HOSTS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp_dir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(
                lambda candidate: replay(args.binary, args.data, candidate,
                                         args.sweep_start_us, args.sweep_end_us,
                                         host_arch, excluded_hosts,
                                         args.switch_at_us, temp_dir),
                candidates,
            ))
    baseline = next(row for row in results if (row["weight"], row["base_cap"],
                                                 row["residual_cap"],
                                                 row["error_ratio"]) ==
                    (4, 192, 1.0, 0.1))
    expected_hosts = 20 if args.include_all_hosts else 18
    if len(baseline["online_integrity"]) != expected_hosts or any(
        row["mae_ms"] > 0.1 for row in baseline["online_integrity"].values()
    ):
        raise RuntimeError("baseline replay does not match online nodes")
    args.output.write_text(json.dumps({"sweep_start_us": args.sweep_start_us,
                                       "sweep_end_us": args.sweep_end_us,
                                       "switch_at_us": args.switch_at_us,
                                       "host_arch": host_arch,
                                       "excluded_hosts": sorted(excluded_hosts),
                                       "results": results}, indent=2) + "\n",
                           encoding="utf-8")
    for row in sorted(results, key=lambda value: value["metrics"]["MI"]["mae_ms"])[:10]:
        metrics = row["metrics"]
        print(row["weight"], row["base_cap"], row["residual_cap"],
              row["error_ratio"],
              "MI", round(metrics["MI"]["mae_ms"], 1),
              "MI_2H", round(metrics["MI_second_half"]["mae_ms"], 1),
              "RTX", round(metrics["RTX"]["mae_ms"], 1),
              "RTX_2H", round(metrics["RTX_second_half"]["mae_ms"], 1))


if __name__ == "__main__":
    main()
