#!/usr/bin/env python3
"""Isolate MLA FP8 cache-write latency from the surrounding PD model."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from rtp_llm.ops import compute_ops


TOKENS = 65536
WIDTH = 576
CALLS_PER_SAMPLE = 10


def stable(values):
    median = statistics.median(values)
    return all(abs(value / median - 1) <= 0.05 for value in values)


def run_case(name, block_size, physical_padding, active_divisor, kv_c, k_pe, scale):
    slots = torch.arange(TOKENS, device="cuda", dtype=torch.int64)
    if active_divisor != 1:
        slots = torch.where(slots % active_divisor == 0,
                            slots // active_divisor, -1)
    active_tokens = TOKENS // active_divisor
    blocks = (active_tokens + block_size - 1) // block_size
    cache = torch.empty_strided(
        (blocks, block_size, WIDTH),
        (block_size * WIDTH * physical_padding, WIDTH, 1),
        dtype=torch.uint8,
        device="cuda",
    )
    cache.zero_()

    def one():
        compute_ops.concat_and_cache_mla(kv_c, k_pe, cache, slots, "fp8", scale)

    one()
    torch.cuda.synchronize()
    for token in (0, TOKENS // 2, TOKENS - active_divisor):
        if token % active_divisor:
            continue
        slot = token // active_divisor
        entry = cache[slot // block_size, slot % block_size]
        assert bool(torch.any(entry[:512] != 0))
        assert bool(torch.any(entry[512:] != 0))

    def timed():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(CALLS_PER_SAMPLE):
            one()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / CALLS_PER_SAMPLE

    timed()  # Lazy runtime initialization is not a warmup sample.
    warmups = []
    for _ in range(40):
        warmups.append(timed())
        if len(warmups) >= 10 and stable(warmups[-3:]):
            break
    else:
        raise RuntimeError(f"{name}: device warmup did not converge")
    samples = [timed() for _ in range(50)]
    return {
        "name": name,
        "block_size": block_size,
        "physical_padding": physical_padding,
        "active_divisor": active_divisor,
        "warmup_count": len(warmups),
        "last3_warmup_ms": warmups[-3:],
        "samples_per_case": len(samples),
        "calls_per_sample": CALLS_PER_SAMPLE,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    kv_c = torch.full((TOKENS, 512), 1.25, dtype=torch.bfloat16, device="cuda")
    k_pe = torch.full((TOKENS, 64), 0.75, dtype=torch.bfloat16, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device="cuda")
    cases = [
        ("128_full", 128, 1, 1),
        ("4096_full", 4096, 1, 1),
        ("128_padded4_full", 128, 4, 1),
        ("128_one_eighth_active", 128, 1, 8),
    ]
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tokens": TOKENS,
        "cache_format": "ordinary E4M3 FP8, 512 latent + 64 RoPE bytes per token",
        "scope": "single-rank cache writer diagnostic, not PD or whole-model latency",
        "cases": [run_case(*case, kv_c, k_pe, scale) for case in cases],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({row["name"]: row["median_ms"] for row in report["cases"]}))


if __name__ == "__main__":
    main()
