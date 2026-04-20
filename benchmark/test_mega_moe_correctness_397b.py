"""Correctness test: mega_moe vs BF16 baseline on real Qwen3.5-397B-A17B layer-1 weights.

Loads the actual FP8 MoE expert weights from the 397B checkpoint, dequantizes
them to BF16, then runs both:
  - Baseline: pure-PyTorch BF16 grouped GEMM
  - mega_moe: DeepGEMM fp8_fp4_mega_moe fused EP+GEMM+SwiGLU kernel

Usage (single-rank correctness, then 4-rank perf):
  python benchmark/test_mega_moe_correctness_397b.py --mode correctness
  torchrun --nproc_per_node=4 benchmark/test_mega_moe_correctness_397b.py --mode perf
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

MODEL_DIR = "/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8"
HIDDEN      = 4096
INTER       = 1024
NUM_EXPERTS = 512
TOP_K       = 10
BLOCK_M_ALIGN = 128


# ── FP8 dequantization (block_size = 128×128) ─────────────────────────────────

def dequant_fp8_block(w_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize block-quantized FP8 weight to BF16.

    w_fp8:     [N, K]  float8_e4m3fn
    scale_inv: [N//128, K//128] bfloat16 – multiply to recover BF16
    Returns:   [N, K]  bfloat16
    """
    n, k = w_fp8.shape
    s = scale_inv.to(torch.float32)
    s_expanded = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    s_expanded = s_expanded[:n, :k]
    return (w_fp8.to(torch.float32) * s_expanded).to(torch.bfloat16)


# ── Weight loading ─────────────────────────────────────────────────────────────

def load_layer_experts_bf16(
    layer: int,
    expert_start: int,
    expert_end: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load experts [expert_start, expert_end) for `layer` and dequantize to BF16.

    Returns:
        w1: [E_local, 2*INTER, HIDDEN]  BF16  (gate+up stacked)
        w2: [E_local, HIDDEN, INTER]    BF16  (down projection)
    """
    import safetensors.torch as st, json

    idx = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
    wmap = idx["weight_map"]

    E_local = expert_end - expert_start
    w1 = torch.zeros(E_local, 2 * INTER, HIDDEN, dtype=torch.bfloat16, device=device)
    w2 = torch.zeros(E_local, HIDDEN, INTER, dtype=torch.bfloat16, device=device)

    # Which shard files we need
    needed_shards: dict[str, list[int]] = {}
    pfx = f"model.language_model.layers.{layer}.mlp.experts."
    for eid in range(expert_start, expert_end):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"{pfx}{eid}.{proj}.weight"
            shard = wmap[key]
            needed_shards.setdefault(shard, []).append(eid)

    # Load shard by shard to avoid loading everything at once
    loaded_shards: dict[str, dict] = {}
    for shard in sorted(set(needed_shards.keys())):
        shard_path = os.path.join(MODEL_DIR, shard)
        loaded_shards[shard] = st.load_file(shard_path, device="cpu")

    for eid in range(expert_start, expert_end):
        local_e = eid - expert_start

        def _load(proj: str) -> torch.Tensor:
            key_w = f"{pfx}{eid}.{proj}.weight"
            key_s = f"{pfx}{eid}.{proj}.weight_scale_inv"
            shard_name = wmap[key_w]
            sd = loaded_shards[shard_name]
            return dequant_fp8_block(sd[key_w], sd[key_s])

        gate_bf16 = _load("gate_proj")  # [I, H]
        up_bf16   = _load("up_proj")    # [I, H]
        down_bf16 = _load("down_proj")  # [H, I]

        w1[local_e, :INTER]      = gate_bf16.to(device)
        w1[local_e, INTER:]      = up_bf16.to(device)
        w2[local_e]              = down_bf16.to(device)

    return w1, w2


# ── BF16 reference MoE ────────────────────────────────────────────────────────

def ref_moe_forward(
    hidden_states: torch.Tensor,   # [T, H] BF16
    topk_weights: torch.Tensor,    # [T, K] float32
    topk_ids: torch.Tensor,        # [T, K] int64  (local expert IDs 0..E_local-1)
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
        x_e = hidden_states[tok_idx].float()
        gate_up = x_e @ w1[e].float().T        # [t_e, 2I]
        gate, up = gate_up[:, :I], gate_up[:, I:]
        l1_out = F.silu(gate) * up              # [t_e, I]
        l2_out = l1_out @ w2[e].float().T       # [t_e, H]
        for slot in range(topk_ids.shape[1]):
            slot_mask = topk_ids[tok_idx, slot] == e
            if slot_mask.any():
                w_slot = topk_weights[tok_idx[slot_mask], slot].float().unsqueeze(1)
                out[tok_idx[slot_mask]] += w_slot * l2_out[slot_mask]
    return out.bfloat16()


# ── mega_moe helpers ─────────────────────────────────────────────────────────

def cast_to_fp4(w_bf16: torch.Tensor):
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4

    G, n, k = w_bf16.shape
    packs, sfs = [], []
    for i in range(G):
        p, s = per_token_cast_to_fp4(w_bf16[i].float(), use_ue8m0=True,
                                      gran_k=32, use_packed_ue8m0=False)
        packs.append(p); sfs.append(s)
    packed = torch.stack(packs)
    sf     = torch.stack(sfs)
    sf     = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), G)
    return packed, sf


def make_mega_weights(w1_bf16, w2_bf16):
    import deep_gemm
    l1 = cast_to_fp4(w1_bf16)
    l2 = cast_to_fp4(w2_bf16)
    return deep_gemm.transform_weights_for_mega_moe(l1, l2)


def mega_moe_run(hidden_states, topk_weights, topk_ids, l1_w, l2_w, buf):
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp8

    T, H = hidden_states.shape
    x_fp8, x_sf = per_token_cast_to_fp8(hidden_states.float(), use_ue8m0=True,
                                          gran_k=32, use_packed_ue8m0=True)
    buf.x[:T].copy_(x_fp8)
    buf.x_sf[:T].copy_(x_sf)
    buf.topk_idx[:T].copy_(topk_ids)
    buf.topk_weights[:T].copy_(topk_weights)

    out = torch.empty((T, H), dtype=torch.bfloat16, device=hidden_states.device)
    deep_gemm.fp8_fp4_mega_moe(out, l1_w, l2_w, buf,
                                activation_clamp=10.0, fast_math=True)
    return out


# ── Correctness test (single rank, no EP comm) ────────────────────────────────

def test_correctness():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", init_method="env://",
                             world_size=1, rank=0)
    import deep_gemm

    print("[correctness] Loading Qwen3.5-397B layer-1 expert weights (512 experts) ...",
          flush=True)
    w1, w2 = load_layer_experts_bf16(layer=1, expert_start=0, expert_end=NUM_EXPERTS,
                                      device=device)
    print(f"  w1: {tuple(w1.shape)} BF16  ({w1.numel()*2/1e9:.2f} GB)", flush=True)
    print(f"  w2: {tuple(w2.shape)} BF16  ({w2.numel()*2/1e9:.2f} GB)", flush=True)

    # Synthetic routing: 2048 tokens, top-10 global expert IDs
    torch.manual_seed(42)
    T = 2048
    hidden = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
    topk_w = torch.softmax(torch.randn(T, TOP_K, device=device), dim=-1).float()
    topk_ids_global = torch.randint(0, NUM_EXPERTS, (T, TOP_K), device=device,
                                     dtype=torch.int32)
    topk_ids_long   = topk_ids_global.long()

    # --- Baseline BF16 ---
    print("[correctness] Running BF16 baseline ...", flush=True)
    t0 = time.perf_counter()
    ref_out = ref_moe_forward(hidden, topk_w, topk_ids_long, w1, w2)
    torch.cuda.synchronize()
    t_baseline = time.perf_counter() - t0
    print(f"  baseline: {t_baseline*1e3:.1f} ms", flush=True)

    # --- mega_moe ---
    print("[correctness] Converting weights to FP4 ...", flush=True)
    l1_w, l2_w = make_mega_weights(w1, w2)

    block_m = deep_gemm._C.get_block_m_for_mega_moe(1, NUM_EXPERTS, T, TOP_K)
    aligned_T = math.ceil(T / block_m) * block_m
    print(f"  block_m={block_m}, aligned_T={aligned_T}", flush=True)

    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=aligned_T,
        num_topk=TOP_K,
        hidden=HIDDEN,
        intermediate_hidden=INTER,
        activation="swiglu",
    )

    # Warmup
    for _ in range(2):
        mega_out = mega_moe_run(hidden, topk_w, topk_ids_global, l1_w, l2_w, buf)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    mega_out = mega_moe_run(hidden, topk_w, topk_ids_global, l1_w, l2_w, buf)
    torch.cuda.synchronize()
    t_mega = time.perf_counter() - t0
    print(f"  mega_moe: {t_mega*1e3:.1f} ms", flush=True)

    # --- Comparison ---
    ref_f = ref_out.float()
    mega_f = mega_out.float()

    cos_sim = F.cosine_similarity(ref_f, mega_f, dim=-1).mean().item()
    max_err = (ref_f - mega_f).abs().max().item()
    rms_err = ((ref_f - mega_f) ** 2).mean().sqrt().item()
    ref_rms = (ref_f ** 2).mean().sqrt().item()
    rel_err = rms_err / (ref_rms + 1e-8)

    print("\n" + "=" * 65)
    print("Correctness: mega_moe vs BF16 baseline on Qwen3.5-397B layer-1")
    print(f"  Tokens: {T}, Experts: {NUM_EXPERTS}, top-k: {TOP_K}")
    print("-" * 65)
    print(f"  Cosine similarity:  {cos_sim:.6f}  (1.0 = perfect)")
    print(f"  Max absolute error: {max_err:.4f}")
    print(f"  RMS error:          {rms_err:.4f}")
    print(f"  Relative RMS error: {rel_err:.4f}")
    print("-" * 65)
    print(f"  BF16 baseline:      {t_baseline*1e3:.1f} ms")
    print(f"  mega_moe:           {t_mega*1e3:.1f} ms")
    print(f"  Speedup:            {t_baseline/t_mega:.1f}x")
    print("=" * 65)

    # FP4+FP8 quantization typically gives 0.97-0.99 cosine similarity vs BF16.
    PASS_THRESHOLD = 0.95
    if cos_sim >= PASS_THRESHOLD:
        print(f"\n[PASS] cosine_sim={cos_sim:.4f} >= {PASS_THRESHOLD}")
    else:
        print(f"\n[FAIL] cosine_sim={cos_sim:.4f} < {PASS_THRESHOLD}")
        sys.exit(1)

    buf.destroy()
    dist.destroy_process_group()


# ── Performance test (4-rank EP) ─────────────────────────────────────────────

def test_perf(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    import deep_gemm

    ws = dist.get_world_size()
    r  = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    E_local = NUM_EXPERTS // ws
    expert_start = r * E_local

    if r == 0:
        print(f"[perf] EP={ws}, {E_local} experts/rank, loading weights ...", flush=True)

    w1, w2 = load_layer_experts_bf16(layer=1, expert_start=expert_start,
                                      expert_end=expert_start + E_local, device=device)
    dist.barrier()
    if r == 0:
        print("[perf] Weights loaded. Converting to FP4 ...", flush=True)

    l1_w, l2_w = make_mega_weights(w1, w2)
    del w1, w2
    dist.barrier()
    if r == 0:
        print("[perf] FP4 conversion done. Allocating NVLink buffer ...", flush=True)

    T = args.seq_len
    block_m = deep_gemm._C.get_block_m_for_mega_moe(ws, NUM_EXPERTS, T, TOP_K)
    aligned_T = math.ceil(T / block_m) * block_m

    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=aligned_T,
        num_topk=TOP_K,
        hidden=HIDDEN,
        intermediate_hidden=INTER,
        activation="swiglu",
    )

    torch.manual_seed(42 + r)

    def make_inputs():
        hs  = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
        tw  = torch.softmax(torch.randn(T, TOP_K, device=device), dim=-1).float()
        ti  = torch.randint(0, NUM_EXPERTS, (T, TOP_K), device=device, dtype=torch.int32)
        return hs, tw, ti

    # Warmup
    dist.barrier()
    if r == 0:
        print(f"[perf] Warmup ({args.warmup} iters) ...", flush=True)
    for _ in range(args.warmup):
        hs, tw, ti = make_inputs()
        mega_moe_run(hs, tw, ti, l1_w, l2_w, buf)
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    if r == 0:
        print(f"[perf] Benchmarking ({args.repeats} iters) ...", flush=True)
    timings = []
    for _ in range(args.repeats):
        hs, tw, ti = make_inputs()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mega_moe_run(hs, tw, ti, l1_w, l2_w, buf)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e3)

    if r == 0:
        timings.sort()
        p50 = timings[len(timings) // 2]
        p90 = timings[int(len(timings) * 0.9)]
        total_flops = 2 * T * TOP_K * INTER * HIDDEN * 2 / NUM_EXPERTS * 3
        tflops = total_flops / (p50 / 1e3) / 1e12
        print("\n" + "=" * 65)
        print("Performance: mega_moe on Qwen3.5-397B-A17B layer-1 (real weights)")
        print(f"  EP={ws}, tokens={T}, experts={NUM_EXPERTS}, top-k={TOP_K}")
        print("-" * 65)
        print(f"  p50 latency:   {p50:.2f} ms")
        print(f"  p90 latency:   {p90:.2f} ms")
        print(f"  min latency:   {min(timings):.2f} ms")
        print(f"  est. TFLOPS:   {tflops:.1f}")
        print("=" * 65)

    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["correctness", "perf"], default="correctness")
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--warmup",  type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        print(f"ERROR: mega_moe requires SM100+, got SM{cap[0]}{cap[1]}")
        sys.exit(1)

    if args.mode == "correctness":
        # Single-rank: set env vars for NCCL
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        test_correctness()
    else:
        test_perf(args)
