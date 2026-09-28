#!/usr/bin/env python3
"""Compare packed and grouped cuLA checkpoints on one CUDA device."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from cula.kda import chunk_kda


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(module, inputs, block_size):
    q, k, v, g, beta, a_log, dt_bias = inputs
    length, heads = q.shape[:2]
    pages = (length + block_size - 1) // block_size
    segments = tuple(
        module.StateSegment(
            start=page * block_size,
            end=min((page + 1) * block_size, length),
            cache_block=page + 1,
        )
        for page in range(pages)
    )
    cache = torch.zeros((pages + 1, heads, 128, 128),
                        dtype=torch.float32, device=q.device)
    with torch.inference_mode():
        output = module._cula_paged_prefill(
            chunk_kda, q, k, v, g, beta, a_log, dt_bias, -5.0,
            cache, (module.StateSequence(None, segments),), block_size,
        )
        torch.cuda.synchronize()
    return output, cache[1:]


def compare(left, right, atol, rtol):
    diff = (left.float() - right.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "reference_abs_max": left.float().abs().max().item(),
        "all_finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        "within_tolerance": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--pages", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    args = parser.parse_args()
    if args.pages <= 4 or args.heads <= 0 or args.block_size % 64:
        parser.error("require more than four pages, positive heads, and 64-aligned pages")
    torch.manual_seed(20260928)
    device = torch.device("cuda:0")
    length = (args.pages - 1) * args.block_size + 3
    shape = (length, args.heads, 128)
    q, k, v, g = (
        torch.randn(shape, device=device, dtype=torch.bfloat16).mul_(0.1)
        for _ in range(4)
    )
    beta = torch.randn((length, args.heads), device=device,
                       dtype=torch.bfloat16).mul_(0.1)
    a_log = torch.zeros(args.heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(args.heads * 128, dtype=torch.float32, device=device)
    inputs = (q, k, v, g, beta, a_log, dt_bias)
    baseline = load_module("native_kda_baseline", args.baseline)
    candidate = load_module("native_kda_candidate", args.candidate)
    baseline_output, baseline_cache = run(baseline, inputs, args.block_size)
    candidate_output, candidate_cache = run(candidate, inputs, args.block_size)
    result = {
        "length": length, "heads": args.heads,
        "block_size": args.block_size, "pages": args.pages,
        "output": compare(baseline_output, candidate_output, 0.05, 0.05),
        "checkpoint_state": compare(baseline_cache, candidate_cache, 0.05, 0.05),
    }
    result["passed"] = all(
        value["all_finite"] and value["within_tolerance"]
        for value in (result["output"], result["checkpoint_state"])
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
