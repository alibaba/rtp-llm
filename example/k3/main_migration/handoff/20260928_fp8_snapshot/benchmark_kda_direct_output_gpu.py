#!/usr/bin/env python3
"""Measure the packed cuLA output-copy change on identical GPU inputs."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from compare_native_kda_packed_gpu import load_module
from cula.kda import chunk_kda


def quantiles(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def stable(samples):
    recent = samples[-3:]
    middle = statistics.median(recent)
    return all(abs(value / middle - 1) <= 0.05 for value in recent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pages", type=int, default=17)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.pages <= 4 or args.warmups < 10 or args.samples < 10:
        parser.error("require >4 pages, >=10 warmups, and >=10 samples")

    torch.manual_seed(20260928)
    length = (args.pages - 1) * args.block_size + 3
    shape = (length, args.heads, 128)
    q, k, v, g = (
        torch.randn(shape, device="cuda:0", dtype=torch.bfloat16).mul_(0.1)
        for _ in range(4)
    )
    beta = torch.randn((length, args.heads), device="cuda:0", dtype=torch.bfloat16).mul_(0.1)
    a_log = torch.zeros(args.heads, device="cuda:0", dtype=torch.float32)
    dt_bias = torch.zeros(args.heads * 128, device="cuda:0", dtype=torch.float32)
    inputs = (q, k, v, g, beta, a_log, dt_bias)
    modules = {
        "baseline": load_module("native_kda_baseline", args.baseline),
        "candidate": load_module("native_kda_candidate", args.candidate),
    }
    paths = {}
    for name, module in modules.items():
        segments = tuple(
            module.StateSegment(
                start=page * args.block_size,
                end=min((page + 1) * args.block_size, length),
                cache_block=page + 1,
            )
            for page in range(args.pages)
        )
        paths[name] = {
            "module": module,
            "cache": torch.zeros(
                (args.pages + 1, args.heads, 128, 128),
                device="cuda:0", dtype=torch.float32,
            ),
            "sequences": (module.StateSequence(None, segments),),
        }

    def timed(name):
        path = paths[name]
        path["cache"].zero_()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            result = path["module"]._cula_paged_prefill(
                chunk_kda, *inputs, -5.0, path["cache"],
                path["sequences"], args.block_size,
            )
        end.record()
        end.synchronize()
        if result.shape != v.shape:
            raise RuntimeError("Unexpected cuLA output shape")
        return start.elapsed_time(end)

    warmups = {}
    for name in paths:
        timed(name)  # Materialize CUDA and JIT before the warmup set.
        values = []
        for _ in range(30):
            values.append(timed(name))
            if len(values) >= args.warmups and stable(values):
                break
        if not stable(values):
            raise RuntimeError(f"{name} warmup did not converge: {values[-3:]}")
        warmups[name] = values

    samples = {name: [] for name in paths}
    for index in range(args.samples):
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        for name in order:
            samples[name].append(timed(name))

    base = statistics.median(samples["baseline"])
    candidate = statistics.median(samples["candidate"])
    report = {
        "length": length,
        "heads": args.heads,
        "block_size": args.block_size,
        "pages": args.pages,
        "warmup_iterations": {name: len(values) for name, values in warmups.items()},
        "warmup_last_three_ms": {name: values[-3:] for name, values in warmups.items()},
        "sample_count_per_variant": args.samples,
        "baseline": quantiles(samples["baseline"]),
        "candidate": quantiles(samples["candidate"]),
        "median_speedup_pct": 100 * (base - candidate) / base,
        "device": torch.cuda.get_device_name(0),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("baseline", "candidate")}, indent=2))
    print(f"baseline={base:.3f} ms candidate={candidate:.3f} ms")


if __name__ == "__main__":
    main()
