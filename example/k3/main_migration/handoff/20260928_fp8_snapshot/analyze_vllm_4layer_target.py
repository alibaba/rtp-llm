#!/usr/bin/env python3
"""Summarize the fixed vLLM four-layer target Prefill GPU annotations."""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def family(name: str) -> str:
    key = name.lower()
    if "nccldevkernel" in key:
        return "NCCL.BF16"
    if "tokenspeed_mlafmha" in key:
        return "MLA.TokenSpeed"
    if "flash_kda" in key:
        return "KDA.FlashKDA"
    if "causal_conv1d" in key:
        return "KDA.short_conv"
    if "moe::" in key or key.startswith("bmm_"):
        return "MoE"
    if "nvjet" in key:
        return "GEMM.NVJet_unattributed"
    if "attn_res" in key:
        return "AttnRes"
    if "rms_norm" in key or "layer_norm_gated" in key or "situ_and_mul" in key:
        return "Norm_or_Gate"
    if "quant" in key or "fusedkimik3mlaqkv" in key:
        return "FP8.producer_or_quant"
    if "copy" in key or "memcpy" in key:
        return "Tensor.copy"
    return "Other_unattributed"


def analyze(directory: Path) -> dict:
    ranks = []
    for rank in range(8):
        paths = list(directory.glob(f"*rank{rank}.*.json.gz"))
        if len(paths) != 1:
            raise ValueError(f"expected one trace for rank {rank}, got {paths}")
        with gzip.open(paths[0], "rt") as stream:
            events = json.load(stream)["traceEvents"]
        scopes = sorted((event for event in events
                         if event.get("cat") == "gpu_user_annotation"
                         and event.get("name", "").startswith("execute_context_1(")
                         and event.get("ph") == "X"), key=lambda event: event["ts"])
        if len(scopes) != 3:
            raise ValueError(f"rank {rank}: expected three target scopes, got {len(scopes)}")
        kernels = [event for event in events if event.get("cat") == "kernel"]
        rows = []
        for scope in scopes:
            selected = [event for event in kernels
                        if scope["ts"] <= event["ts"] < scope["ts"] + scope["dur"]]
            if not selected:
                raise ValueError(f"rank {rank}: target scope has no GPU kernels")
            sums = defaultdict(float)
            counts = defaultdict(int)
            for event in selected:
                name = family(event["name"])
                sums[name] += event["dur"] / 1000
                counts[name] += 1
            rows.append({"start_us": scope["ts"], "span_ms": scope["dur"] / 1000,
                         "kernel_count": len(selected),
                         "family_cumulative_ms": dict(sorted(sums.items())),
                         "family_kernel_count": dict(sorted(counts.items()))})
        ranks.append({"rank": rank, "trace": paths[0].name, "requests": rows})

    requests = []
    for index in range(3):
        per_rank = [rank["requests"][index] for rank in ranks]
        worst = max(range(8), key=lambda rank: per_rank[rank]["span_ms"])
        requests.append({"request_index": index,
                         "rank_start_spread_us": max(row["start_us"] for row in per_rank)
                         - min(row["start_us"] for row in per_rank),
                         "worst_rank": worst,
                         "max_rank_gpu_span_ms": per_rank[worst]["span_ms"],
                         "worst_rank_family_cumulative_ms":
                         per_rank[worst]["family_cumulative_ms"]})
    return {"source": "vLLM 3df4 PD TP8/EP8 target only, no MTP",
            "timing_basis": "GPU user annotation span; family sums overlap across streams",
            "ranks": ranks, "matched_requests": requests,
            "median_max_rank_gpu_span_ms": statistics.median(
                row["max_rank_gpu_span_ms"] for row in requests)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"matched": len(result["matched_requests"]),
                      "median_max_rank_gpu_span_ms": result["median_max_rank_gpu_span_ms"]}))
