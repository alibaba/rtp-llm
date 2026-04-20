"""Benchmark: fp8_fp4_mega_moe on Qwen3.5-397B-A17B dimensions.

Qwen3.5-397B-A17B:  hidden=4096, intermediate=1024, experts=512, top_k=10
  1024 % 512 == 0  → compatible with mega_moe alignment requirement ✓

Usage (4x GB200, EP=4):

  # Baseline: BF16 local GEMM, no EP comm
  torchrun --nproc_per_node=4 benchmark/bench_mega_moe_qwen35_397b.py

  # mega_moe: FP8A + FP4W + fused NVLink EP
  USE_MEGA_MOE=1 torchrun --nproc_per_node=4 benchmark/bench_mega_moe_qwen35_397b.py

  # Larger batch (8 GPUs, DP=2 EP=4):
  USE_MEGA_MOE=1 torchrun --nproc_per_node=8 benchmark/bench_mega_moe_qwen35_397b.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ── Distributed helpers ──────────────────────────────────────────────────────

def rank() -> int:
    return dist.get_rank()

def world_size() -> int:
    return dist.get_world_size()

def is_rank0() -> bool:
    return rank() == 0

def log(msg: str):
    if is_rank0():
        print(f"[rank0] {msg}", flush=True)

def setup_dist() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


# ── FP4 weight preparation ────────────────────────────────────────────────────

def cast_grouped_to_fp4(
    bf16_weights: torch.Tensor,  # [G, N, K] BF16
) -> Tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4

    num_groups, n, k = bf16_weights.shape
    packed_list: List[torch.Tensor] = []
    sf_list: List[torch.Tensor] = []
    for i in range(num_groups):
        p, s = per_token_cast_to_fp4(
            bf16_weights[i].float(),
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=False,
        )
        packed_list.append(p)
        sf_list.append(s)

    packed = torch.stack(packed_list)
    sf = torch.stack(sf_list)
    sf = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), num_groups)
    return packed, sf


def prepare_mega_moe_weights(
    w1_bf16: torch.Tensor,  # [E, 2I, H]
    w2_bf16: torch.Tensor,  # [E, H, I]
) -> Tuple[Tuple, Tuple]:
    import deep_gemm

    l1 = cast_grouped_to_fp4(w1_bf16)
    l2 = cast_grouped_to_fp4(w2_bf16)
    return deep_gemm.transform_weights_for_mega_moe(l1, l2)


# ── Baseline: BF16 local GEMM (no EP comm) ───────────────────────────────────

def baseline_moe_forward(
    hidden_states: torch.Tensor,   # [T, H] BF16
    topk_weights: torch.Tensor,    # [T, K] float32
    topk_ids: torch.Tensor,        # [T, K] int64 (local expert IDs 0..E_local-1)
    w1: torch.Tensor,              # [E_local, 2I, H]
    w2: torch.Tensor,              # [E_local, H, I]
) -> torch.Tensor:
    T, H = hidden_states.shape
    E_local = w1.shape[0]
    I = w1.shape[1] // 2

    out = torch.zeros_like(hidden_states, dtype=torch.float32)
    for e in range(E_local):
        tok_mask = (topk_ids == e).any(dim=-1)
        if not tok_mask.any():
            continue
        tok_idx = tok_mask.nonzero(as_tuple=True)[0]
        x_e = hidden_states[tok_idx].float()  # [t_e, H]
        gate_up = x_e @ w1[e].float().T       # [t_e, 2I]
        gate, up = gate_up[:, :I], gate_up[:, I:]
        l1_out = F.silu(gate) * up             # [t_e, I]
        l2_out = l1_out @ w2[e].float().T     # [t_e, H]
        for slot in range(topk_ids.shape[1]):
            slot_mask = topk_ids[tok_idx, slot] == e
            if slot_mask.any():
                w_slot = topk_weights[tok_idx[slot_mask], slot].float().unsqueeze(1)
                out[tok_idx[slot_mask]] += w_slot * l2_out[slot_mask]
    return out.bfloat16()


# ── mega_moe forward ──────────────────────────────────────────────────────────

def mega_moe_forward(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    l1_weights: Tuple,
    l2_weights: Tuple,
    symm_buffer,
) -> torch.Tensor:
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp8

    T, H = hidden_states.shape
    x_fp8, x_sf = per_token_cast_to_fp8(
        hidden_states.float(), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )

    buf = symm_buffer
    buf.x[:T].copy_(x_fp8)
    buf.x_sf[:T].copy_(x_sf)
    buf.topk_idx[:T].copy_(topk_ids)
    buf.topk_weights[:T].copy_(topk_weights)

    y = torch.empty((T, H), dtype=torch.bfloat16, device=hidden_states.device)
    deep_gemm.fp8_fp4_mega_moe(
        y, l1_weights, l2_weights, buf,
        activation_clamp=10.0, fast_math=True,
    )
    return y


# ── Main ──────────────────────────────────────────────────────────────────────

def bench(args):
    local_rank = setup_dist()
    ws = world_size()
    r = rank()
    device = torch.device(f"cuda:{local_rank}")

    cap = torch.cuda.get_device_capability()
    log(f"GPU: sm{cap[0]}{cap[1]}, world_size={ws}")

    use_mega_moe = os.environ.get("USE_MEGA_MOE", "0") == "1"

    # Qwen3.5-397B-A17B dimensions
    hidden       = args.hidden
    intermediate = args.intermediate
    num_experts  = args.num_experts
    top_k        = args.top_k
    total_tokens = args.seq_len

    if use_mega_moe:
        if intermediate % 512 != 0:
            print(f"ERROR: mega_moe requires intermediate % 512 == 0, got {intermediate}")
            sys.exit(1)
        if cap[0] < 10:
            print(f"ERROR: mega_moe requires SM100+ (GB200), got SM{cap[0]}{cap[1]}")
            sys.exit(1)

    # With EP=ws, each rank owns a subset of experts
    num_local_experts = num_experts // ws
    expert_start = r * num_local_experts
    per_rank_tokens = total_tokens  # each rank processes all tokens (all-to-all handles routing)

    log(f"Config: hidden={hidden}, intermediate={intermediate}, "
        f"experts={num_experts} ({num_local_experts}/rank), top_k={top_k}")
    log(f"Tokens: total={total_tokens} ({per_rank_tokens}/rank)")
    log(f"Kernel: {'mega_moe (FP8A+FP4W+fused-EP)' if use_mega_moe else 'baseline (BF16 local GEMM)'}")

    # Create synthetic BF16 weights
    log("Creating synthetic weights...")
    w1_bf16 = torch.randn(
        num_local_experts, 2 * intermediate, hidden,
        dtype=torch.bfloat16, device=device
    ) * 0.02
    w2_bf16 = torch.randn(
        num_local_experts, hidden, intermediate,
        dtype=torch.bfloat16, device=device
    ) * 0.02

    def make_inputs():
        hs = torch.randn(per_rank_tokens, hidden, dtype=torch.bfloat16, device=device)
        tw = torch.softmax(
            torch.randn(per_rank_tokens, top_k, device=device), dim=-1
        ).float()
        ti = torch.randint(
            0, num_experts, (per_rank_tokens, top_k), device=device, dtype=torch.int32
        )
        return hs, tw, ti

    if use_mega_moe:
        import deep_gemm

        log("Converting weights to FP4...")
        l1_weights, l2_weights = prepare_mega_moe_weights(w1_bf16, w2_bf16)

        block_m = deep_gemm._C.get_block_m_for_mega_moe(
            ws, num_experts, per_rank_tokens, top_k
        )
        aligned_tokens = math.ceil(per_rank_tokens / block_m) * block_m
        log(f"block_m={block_m}, aligned_tokens={aligned_tokens}")

        log("Allocating NVLink symmetric buffer...")
        symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            num_experts=num_experts,
            num_max_tokens_per_rank=aligned_tokens,
            num_topk=top_k,
            hidden=hidden,
            intermediate_hidden=intermediate,
            activation="swiglu",
        )

        def run():
            hs, tw, ti = make_inputs()
            return mega_moe_forward(hs, tw, ti, l1_weights, l2_weights, symm_buffer)

    else:
        def run():
            hs, tw, ti = make_inputs()
            local_ti = (ti % num_local_experts).long()
            return baseline_moe_forward(hs, tw, local_ti, w1_bf16, w2_bf16)

    # Warmup
    dist.barrier()
    log(f"Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    log(f"Benchmarking ({args.repeats} iters)...")
    timings = []
    for _ in range(args.repeats):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e3)

    if is_rank0():
        timings.sort()
        p50 = timings[len(timings) // 2]
        p90 = timings[int(len(timings) * 0.9)]
        # FLOPs: 2 experts per token on average per local rank, 3 matmuls each
        # Rough estimate: 2 * seq_len * hidden * intermediate * 3 (gate/up/down)
        total_flops = (
            2 * total_tokens * top_k * intermediate * hidden * 2 / num_experts
        ) * 3  # 3 GEMMs per selected expert slot
        tflops = total_flops / (p50 / 1e3) / 1e12

        kernel_name = (
            "mega_moe (FP8A+FP4W+fused-EP)"
            if use_mega_moe
            else "baseline BF16 local GEMM"
        )
        print("\n" + "=" * 70)
        print(f"Qwen3.5-397B-A17B MoE Layer Benchmark")
        print(f"  Kernel:        {kernel_name}")
        print(f"  EP ranks:      {ws}")
        print(f"  hidden/inter:  {hidden}/{intermediate}")
        print(f"  experts:       {num_experts} total, {top_k} top-k")
        print(f"  total tokens:  {total_tokens}")
        print("-" * 70)
        print(f"  p50 latency:   {p50:.2f} ms")
        print(f"  p90 latency:   {p90:.2f} ms")
        print(f"  min latency:   {min(timings):.2f} ms")
        print(f"  est. TFLOPS:   {tflops:.1f}")
        print("=" * 70)

    dist.barrier()
    dist.destroy_process_group()
    if use_mega_moe:
        symm_buffer.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Qwen3.5-397B-A17B defaults
    parser.add_argument("--hidden",       type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--num-experts",  type=int, default=512)
    parser.add_argument("--top-k",        type=int, default=10)
    parser.add_argument("--seq-len",      type=int, default=65536,
                        help="Total tokens across all ranks")
    parser.add_argument("--warmup",       type=int, default=3)
    parser.add_argument("--repeats",      type=int, default=10)
    args = parser.parse_args()
    bench(args)
