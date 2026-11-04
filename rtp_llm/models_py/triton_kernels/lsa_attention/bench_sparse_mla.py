"""Microbenchmark for the LSA sparse MLA baseline (CUDA-event timing).

Reports, per case: latency, achieved TFLOP/s and MFU against the B300/SM103
canonical BF16 dense peak (2.5 PFLOPS, adapt-model performance-standard §0),
plus achieved GB/s and MBU against an 8.0 TB/s HBM placeholder that still needs
a sourced peak before being treated as a gate.

Usage (inside a GPU container with torch/triton):

    python bench_sparse_mla.py --device cuda:0 --warmup 10 --iters 50

The script is intentionally dependency-light and prints one JSON object per
line so results can be archived in build_logs for later gates_calc processing.
"""

from __future__ import annotations

import argparse
import json
from typing import Tuple

import torch

from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_triton import (
    sparse_mla_fwd,
    sparse_mla_fwd_split,
)

PEAK_BF16_TFLOPS = 2.5e15  # B300/SM103 BF16 dense, canonical (2026-10-16)
PEAK_HBM_BYTES = 8.0e12  # B300 HBM3e vendor peak; MBU vs this is UNVERIFIED


def build_case(
    t_count: int,
    heads: int,
    dq: int,
    dv: int,
    n_count: int,
    slots: int,
    device: torch.device,
    seed: int = 0,
    pad_ratio: float = 0.01,
    index_mode: str = "lsa",
    init_tokens: int = 16,
    local_tokens: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one benchmark case.

    ``index_mode="lsa"`` reproduces the real LSA slot structure: sink + local
    sliding window (contiguous, shared by neighbouring tokens) plus a dynamic
    sparse remainder sampled uniformly from the rest of the prefix — the
    workload the SI design targets. ``index_mode="random"`` is the adversarial
    uniform-random baseline kept for reference.
    """
    gen = torch.Generator().manual_seed(seed)
    q_abs = (torch.randn(t_count, heads, dq, generator=gen) * 0.1).to(device).bfloat16()
    kv = (torch.randn(n_count, dq, generator=gen) * 0.1).to(device).bfloat16()

    if index_mode == "random":
        idx = torch.randint(0, n_count, (t_count, 1, slots), generator=gen)
    elif index_mode == "lsa":
        # Row positions: prefill rows are consecutive (chunk), decode rows sit
        # at the end of the context.
        if t_count >= n_count:
            raise ValueError("lsa index mode expects n_count > t_count")
        pos = torch.arange(n_count - t_count, n_count)
        idx = torch.full((t_count, 1, slots), -1, dtype=torch.int64)
        n_sink = min(init_tokens, slots)
        idx[:, 0, :n_sink] = torch.arange(n_sink)
        for row in range(t_count):
            position = int(pos[row])
            n_local = min(local_tokens, slots - n_sink, position + 1)
            n_sparse = slots - n_sink - n_local
            cursor = n_sink
            if n_local > 0:
                idx[row, 0, cursor : cursor + n_local] = torch.arange(
                    position - n_local + 1, position + 1
                )
                cursor += n_local
            if n_sparse > 0:
                # Dynamic sparse slots drawn from the prefix outside sink/local.
                high = max(position - n_local, n_sink)
                if high > n_sink:
                    idx[row, 0, cursor : cursor + n_sparse] = torch.randint(
                        n_sink, high, (n_sparse,), generator=gen
                    )
                else:
                    idx[row, 0, cursor : cursor + n_sparse] = torch.randint(
                        0, n_count, (n_sparse,), generator=gen
                    )
    else:
        raise ValueError(f"unknown index_mode {index_mode!r}")
    if pad_ratio > 0:
        pad = torch.rand(t_count, 1, slots, generator=gen) < pad_ratio
        idx = idx.masked_fill(pad, -1)
    return q_abs, kv, idx.to(device)


def bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # milliseconds


def flops_and_bytes(
    t_count: int,
    heads: int,
    dq: int,
    dv: int,
    slots: int,
    effective_slots: float,
    splits: int = 0,
) -> Tuple[float, float]:
    flops = 2.0 * (dq + dv) * heads * t_count * effective_slots
    # KV gather: each selected row is read ONCE per query token (the DV-wide
    # value slice and the rope slice are disjoint parts of the DQ-wide row).
    kv_bytes = t_count * effective_slots * dq * 2
    q_bytes = t_count * heads * dq * 2
    out_bytes = t_count * heads * dv * 2
    idx_bytes = t_count * slots * 4
    # Split-KV partial traffic (bf16 partials written then read by the merge).
    partial_bytes = 0
    if splits > 1:
        partial_bytes = 2 * t_count * splits * heads * dv * 2
    return flops, kv_bytes + q_bytes + out_bytes + idx_bytes + partial_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--split", type=int, default=0,
                        help=">1 enables the split-KV decode path with that many splits")
    parser.add_argument("--index-mode", default="lsa", choices=["lsa", "random"])
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "event": "device",
                "name": props.name,
                "capability": [props.major, props.minor],
                "memory_gb": round(props.total_memory / 2**30, 1),
            }
        )
    )

    cases = [
        ("prefill_t2048_s2048_ctx128k", dict(t_count=2048, n_count=131072, slots=2048)),
        ("prefill_t512_s2048_ctx128k", dict(t_count=512, n_count=131072, slots=2048)),
        ("decode_b32_s2048_ctx128k", dict(t_count=32, n_count=131072, slots=2048)),
        ("decode_b64_s2048_ctx128k", dict(t_count=64, n_count=131072, slots=2048)),
        ("decode_b32_s2048_ctx32k", dict(t_count=32, n_count=32768, slots=2048)),
        ("spec_b32x6_s2048_ctx128k", dict(t_count=192, n_count=131072, slots=2048)),
        ("decode_b128_s2048_ctx128k", dict(t_count=128, n_count=131072, slots=2048)),
        ("decode_b256_s2048_ctx128k", dict(t_count=256, n_count=131072, slots=2048)),
    ]

    for name, kw in cases:
        q_abs, kv, idx = build_case(
            heads=32, dq=576, dv=512, device=device, seed=hash(name) % 1000,
            index_mode=args.index_mode, **kw
        )
        effective = float((idx >= 0).sum().item())

        def run() -> None:
            if args.split > 1 and kw["t_count"] <= 256:
                sparse_mla_fwd_split(
                    q_abs,
                    kv,
                    idx,
                    kv_lora_rank=512,
                    out_dtype=torch.float32,
                    splits=args.split,
                    block_n=args.block_n,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                )
            else:
                # prefer the bit-identical fast single-pass kernel when present (baseline-log #5.89)
                try:
                    from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_fast_triton import (
                        sparse_fast_fwd as _sp_fwd,
                    )
                except Exception:
                    _sp_fwd = sparse_mla_fwd
                _sp_fwd(
                    q_abs,
                    kv,
                    idx,
                    kv_lora_rank=512,
                    out_dtype=torch.float32,
                    block_n=args.block_n,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                )

        ms = bench_ms(run, args.warmup, args.iters)
        splits_used = args.split if (args.split > 1 and kw["t_count"] <= 256) else 0
        flops, nbytes = flops_and_bytes(
            heads=32, dq=576, dv=512, slots=kw["slots"],
            t_count=kw["t_count"], effective_slots=effective / kw["t_count"],
            splits=splits_used,
        )
        seconds = ms / 1e3
        tflops = flops / seconds
        gbps = nbytes / seconds / 1e9
        record = {
            "event": "case",
            "case": name,
            "t_count": kw["t_count"],
            "n_count": kw["n_count"],
            "slots": kw["slots"],
            "latency_ms": round(ms, 4),
            "achieved_tflops": round(tflops / 1e12, 3),
            "mfu_vs_2.5PF": round(tflops / PEAK_BF16_TFLOPS, 6),
            "achieved_gbps": round(gbps, 1),
            "mbu_vs_8TBs_unverified": round(nbytes / seconds / PEAK_HBM_BYTES, 6),
        }
        print(json.dumps(record), flush=True)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()