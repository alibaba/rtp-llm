"""64K prefill timeline: mega_moe on Qwen3.5-397B-A17B (EP=4, real FP8 weights).

Loads real FP8 expert weights from layer 1 (representative of all 60 MoE layers),
converts to FP4, benchmarks mega_moe kernel at 64K tokens with EP=4,
outputs per-iteration timing and full-model prefill latency estimate.

Usage:
  torchrun --nproc_per_node=4 benchmark/bench_mega_moe_397b_prefill_timeline.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

MODEL_DIR   = "/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8"
HIDDEN      = 4096
INTER       = 1024
NUM_EXPERTS = 512
TOP_K       = 10
NUM_MOE_LAYERS = 60    # Qwen3.5-397B-A17B: 60 decoder layers (all MoE)
SEQ_LEN     = 65536    # 64K prefill tokens


# ── Distributed helpers ──────────────────────────────────────────────────────

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def r0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


# ── FP8 dequantization ────────────────────────────────────────────────────────

def dequant_fp8_block(w_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    s = scale_inv.to(torch.float32).repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return (w_fp8.to(torch.float32) * s).to(torch.bfloat16)


# ── Load real 397B FP8 layer weights ─────────────────────────────────────────

def load_layer_experts(layer: int, expert_start: int, expert_end: int,
                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    import safetensors.torch as st

    idx   = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
    wmap  = idx["weight_map"]
    pfx   = f"model.language_model.layers.{layer}.mlp.experts."
    E_loc = expert_end - expert_start

    w1 = torch.zeros(E_loc, 2 * INTER, HIDDEN, dtype=torch.bfloat16, device=device)
    w2 = torch.zeros(E_loc, HIDDEN, INTER,      dtype=torch.bfloat16, device=device)

    shards: dict[str, dict] = {}
    for eid in range(expert_start, expert_end):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shard = wmap[f"{pfx}{eid}.{proj}.weight"]
            if shard not in shards:
                shards[shard] = st.load_file(os.path.join(MODEL_DIR, shard), device="cpu")

    for eid in range(expert_start, expert_end):
        le = eid - expert_start
        def _load(proj):
            kw  = f"{pfx}{eid}.{proj}.weight"
            ks  = f"{pfx}{eid}.{proj}.weight_scale_inv"
            sd  = shards[wmap[kw]]
            return dequant_fp8_block(sd[kw], sd[ks])
        gate = _load("gate_proj")
        up   = _load("up_proj")
        down = _load("down_proj")
        w1[le, :INTER]  = gate.to(device)
        w1[le, INTER:]  = up.to(device)
        w2[le]          = down.to(device)
    return w1, w2


# ── FP4 conversion ────────────────────────────────────────────────────────────

def to_fp4(w: torch.Tensor):
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4
    G, n, k = w.shape
    packs, sfs = [], []
    for i in range(G):
        p, s = per_token_cast_to_fp4(w[i].float(), use_ue8m0=True,
                                      gran_k=32, use_packed_ue8m0=False)
        packs.append(p); sfs.append(s)
    packed = torch.stack(packs)
    sf     = torch.stack(sfs)
    sf     = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), G)
    return packed, sf

def make_weights(w1, w2):
    import deep_gemm
    return deep_gemm.transform_weights_for_mega_moe(to_fp4(w1), to_fp4(w2))


# ── mega_moe forward ──────────────────────────────────────────────────────────

def moe_run(hs, topk_w, topk_ids, l1_w, l2_w, buf):
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp8
    T = hs.shape[0]
    xf, xs = per_token_cast_to_fp8(hs.float(), use_ue8m0=True,
                                    gran_k=32, use_packed_ue8m0=True)
    buf.x[:T].copy_(xf)
    buf.x_sf[:T].copy_(xs)
    buf.topk_idx[:T].copy_(topk_ids)
    buf.topk_weights[:T].copy_(topk_w)
    out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=hs.device)
    deep_gemm.fp8_fp4_mega_moe(out, l1_w, l2_w, buf,
                                activation_clamp=10.0, fast_math=True)
    return out


# ── Precise CUDA-event timing ─────────────────────────────────────────────────

def cuda_time_ms(fn, warmup=3, repeats=20) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    local_rank = setup()
    ws     = dist.get_world_size()
    rank   = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    import deep_gemm
    cap = torch.cuda.get_device_capability()
    r0(f"GPU: SM{cap[0]}{cap[1]} {torch.cuda.get_device_name(local_rank)}, EP={ws}")

    if cap[0] < 10:
        print("ERROR: requires SM100+"); sys.exit(1)

    # Per-rank expert slice
    E_loc  = NUM_EXPERTS // ws
    e_start = rank * E_loc

    # ── Load real FP8 weights (layer 1) ─────────────────────────────────────
    r0(f"\nLoading 397B layer-1 real FP8 weights "
       f"({E_loc} experts/rank, {NUM_EXPERTS} total) ...")
    t0 = time.perf_counter()
    w1, w2 = load_layer_experts(1, e_start, e_start + E_loc, device)
    dist.barrier()
    r0(f"  Loaded in {time.perf_counter()-t0:.1f}s  "
       f"w1={tuple(w1.shape)} w2={tuple(w2.shape)}")

    # ── FP4 conversion ───────────────────────────────────────────────────────
    r0("Converting FP8→BF16→FP4 ...")
    t0 = time.perf_counter()
    l1_w, l2_w = make_weights(w1, w2)
    del w1, w2
    dist.barrier()
    r0(f"  FP4 conversion: {time.perf_counter()-t0:.1f}s")

    # ── Allocate symmetric buffer ─────────────────────────────────────────────
    T        = SEQ_LEN
    block_m  = deep_gemm._C.get_block_m_for_mega_moe(ws, NUM_EXPERTS, T, TOP_K)
    aligned  = math.ceil(T / block_m) * block_m
    r0(f"block_m={block_m}, aligned_T={aligned}")

    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=aligned,
        num_topk=TOP_K,
        hidden=HIDDEN,
        intermediate_hidden=INTER,
        activation="swiglu",
    )

    # ── Synthetic inputs ─────────────────────────────────────────────────────
    torch.manual_seed(42 + rank)
    def make_inputs():
        hs  = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
        tw  = torch.softmax(torch.randn(T, TOP_K, device=device), dim=-1).float()
        ti  = torch.randint(0, NUM_EXPERTS, (T, TOP_K), device=device, dtype=torch.int32)
        return hs, tw, ti

    def run():
        hs, tw, ti = make_inputs()
        return moe_run(hs, tw, ti, l1_w, l2_w, buf)

    # ── Warmup ───────────────────────────────────────────────────────────────
    r0(f"\nWarmup (3 iters, {T} tokens) ...")
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    dist.barrier()

    # ── Benchmark with per-iteration timing ──────────────────────────────────
    REPEATS = 20
    r0(f"Benchmarking ({REPEATS} iters) ...")
    timings = cuda_time_ms(run, warmup=0, repeats=REPEATS)

    if rank == 0:
        timings_s = sorted(timings)
        p50 = timings_s[REPEATS // 2]
        p90 = timings_s[int(REPEATS * 0.9)]
        p10 = timings_s[int(REPEATS * 0.1)]

        # FLOPs: 2 × T × top_k × INTER × HIDDEN × 2 matmuls + 1 activation
        # Two GEMMs: L1=(T*top_k, 2I, H) + L2=(T*top_k, H, I) per selected slot
        # With EP each rank processes T*top_k/EP tokens on average
        flops = 2 * T * TOP_K * (2 * INTER * HIDDEN + INTER * HIDDEN) / ws
        tflops_p50 = flops / (p50 / 1e3) / 1e12

        # Full model estimate: 60 MoE layers × single-layer time
        full_moe_ms = NUM_MOE_LAYERS * p50

        print("\n" + "=" * 72)
        print("Qwen3.5-397B-A17B  ·  64K Prefill Timeline  ·  mega_moe  ·  EP=4")
        print(f"  GPU: SM{cap[0]}{cap[1]} × {ws},  tokens={T},  "
              f"experts={NUM_EXPERTS} ({E_loc}/rank),  top_k={TOP_K}")
        print(f"  Weights: real FP8 layer-1 → FP4 (representative of all {NUM_MOE_LAYERS} layers)")
        print("=" * 72)

        print("\n  Per-iteration timing (CUDA events, ms):")
        print("  " + "-" * 60)
        for i, t in enumerate(timings):
            bar = "█" * int(t / max(timings) * 30)
            print(f"  iter {i+1:02d}: {t:7.3f} ms  {bar}")

        print("\n  Summary (single MoE layer, 64K tokens, EP=4):")
        print(f"    p10:  {p10:.3f} ms")
        print(f"    p50:  {p50:.3f} ms  ← headline")
        print(f"    p90:  {p90:.3f} ms")
        print(f"    min:  {min(timings):.3f} ms")
        print(f"    max:  {max(timings):.3f} ms")
        print(f"    est TFLOPS (p50): {tflops_p50:.1f}")

        print(f"\n  Full 397B model prefill estimate ({NUM_MOE_LAYERS} MoE layers):")
        print(f"    MoE total:  {NUM_MOE_LAYERS} × {p50:.3f} ms = {full_moe_ms:.1f} ms  "
              f"({full_moe_ms/1000:.2f} s)")
        print(f"    Attn total: not benchmarked (GQA, non-bottleneck at 64K prefill)")
        print(f"    MoE throughput: {T / (full_moe_ms/1e3) / 1e3:.1f}K tokens/s")

        print("\n  Layer-by-layer MoE timeline (estimated, p50 per layer):")
        print("  " + "-" * 60)
        cumul = 0.0
        for layer in range(NUM_MOE_LAYERS):
            cumul += p50
            if layer < 5 or layer >= NUM_MOE_LAYERS - 5 or layer == NUM_MOE_LAYERS // 2:
                print(f"  Layer {layer:02d}: +{p50:.2f} ms  →  cumulative {cumul:.1f} ms")
            elif layer == 5:
                print(f"  ...    (layers 5–{NUM_MOE_LAYERS-6} each +{p50:.2f} ms)")
        print(f"\n  Total MoE at end: {cumul:.1f} ms")
        print("=" * 72)

    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
