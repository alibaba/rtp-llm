"""Full 60-layer forward-pass timeline for Qwen3.5-397B-A17B with mega_moe.

Runs a complete 60-layer forward pass with per-layer CUDA-event timing:
  - Attention (GQA): random weights, same shape as real model
  - MoE (mega_moe): real FP8→FP4 weights from layer-1, reused for all 60 layers
    (all layers share identical dimensions; kernel time is identical)

Timing breakdown per layer:
  QKV proj | SDPA | O proj | Gate (router) | mega_moe kernel | total

Usage:
  torchrun --nproc_per_node=4 --master_port=29531 \
    benchmark/bench_mega_moe_397b_full_timeline.py
"""

from __future__ import annotations

import json, math, os, sys, time
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

MODEL_DIR      = "/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8"
HIDDEN         = 4096
NUM_HEADS      = 32
NUM_KV_HEADS   = 2
HEAD_DIM       = HIDDEN // NUM_HEADS   # 128
INTER          = 1024
NUM_EXPERTS    = 512
TOP_K          = 10
NUM_LAYERS     = 60
SEQ_LEN        = 65536


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def r0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


# ── FP8 weight loading ────────────────────────────────────────────────────────

def dequant_fp8_block(w_fp8, scale_inv):
    n, k = w_fp8.shape
    s = scale_inv.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return (w_fp8.float() * s).bfloat16()

def load_layer1_experts(expert_start, expert_end, device):
    import safetensors.torch as st
    idx  = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    pfx  = "model.language_model.layers.1.mlp.experts."
    E    = expert_end - expert_start

    w1 = torch.zeros(E, 2 * INTER, HIDDEN, dtype=torch.bfloat16, device=device)
    w2 = torch.zeros(E, HIDDEN, INTER,      dtype=torch.bfloat16, device=device)

    shards = {}
    for eid in range(expert_start, expert_end):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            s = wmap[f"{pfx}{eid}.{proj}.weight"]
            if s not in shards:
                shards[s] = st.load_file(os.path.join(MODEL_DIR, s), device="cpu")

    for eid in range(expert_start, expert_end):
        le = eid - expert_start
        def _ld(proj):
            kw = f"{pfx}{eid}.{proj}.weight"
            ks = f"{pfx}{eid}.{proj}.weight_scale_inv"
            sd = shards[wmap[kw]]
            return dequant_fp8_block(sd[kw], sd[ks])
        gate, up, down = _ld("gate_proj"), _ld("up_proj"), _ld("down_proj")
        w1[le, :INTER]  = gate.to(device)
        w1[le, INTER:]  = up.to(device)
        w2[le]          = down.to(device)
    return w1, w2


# ── FP4 conversion ────────────────────────────────────────────────────────────

def to_fp4_weights(w1, w2):
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4

    def quant(w):
        G, n, k = w.shape
        packs, sfs = [], []
        for i in range(G):
            p, s = per_token_cast_to_fp4(w[i].float(), use_ue8m0=True,
                                          gran_k=32, use_packed_ue8m0=False)
            packs.append(p); sfs.append(s)
        packed = torch.stack(packs)
        sf     = deep_gemm.transform_sf_into_required_layout(
                     torch.stack(sfs), n, k, (1, 32), G)
        return packed, sf

    return deep_gemm.transform_weights_for_mega_moe(quant(w1), quant(w2))


# ── mega_moe forward ──────────────────────────────────────────────────────────

def moe_fwd(hs, topk_w, topk_ids, l1_w, l2_w, buf):
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


# ── CUDA-event helpers ────────────────────────────────────────────────────────

def evt():
    return torch.cuda.Event(enable_timing=True)

def elapsed(s, e):
    return s.elapsed_time(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    local_rank = setup()
    ws   = dist.get_world_size()
    rank = dist.get_rank()
    dev  = torch.device(f"cuda:{local_rank}")

    import deep_gemm
    cap = torch.cuda.get_device_capability()
    r0(f"GPU: SM{cap[0]}{cap[1]} {torch.cuda.get_device_name(local_rank)} × {ws}")

    if cap[0] < 10:
        print("ERROR: requires SM100+"); sys.exit(1)

    E_loc   = NUM_EXPERTS // ws
    e_start = rank * E_loc

    # ── Load real FP8 layer-1 expert weights ─────────────────────────────────
    r0(f"\nLoading 397B layer-1 real FP8 weights ({E_loc} experts/rank) ...")
    t0 = time.perf_counter()
    w1, w2 = load_layer1_experts(e_start, e_start + E_loc, dev)
    dist.barrier()
    r0(f"  done in {time.perf_counter()-t0:.1f}s")

    r0("Converting to FP4 ...")
    l1_w, l2_w = to_fp4_weights(w1, w2)
    del w1, w2

    # ── Symmetric buffer (reused across all 60 layers) ───────────────────────
    T       = SEQ_LEN
    block_m = deep_gemm._C.get_block_m_for_mega_moe(ws, NUM_EXPERTS, T, TOP_K)
    aligned = math.ceil(T / block_m) * block_m
    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=aligned,
        num_topk=TOP_K,
        hidden=HIDDEN,
        intermediate_hidden=INTER,
        activation="swiglu",
    )
    r0(f"block_m={block_m}, aligned_T={aligned}, buf allocated")

    # ── Random attention weights (shared across layers; GEMM time = f(shape)) ─
    # Weights: q=[HIDDEN, HIDDEN], k=[HIDDEN, NUM_KV_HEADS*HEAD_DIM],
    #          v=[HIDDEN, NUM_KV_HEADS*HEAD_DIM], o=[HIDDEN, HIDDEN]
    KV_DIM = NUM_KV_HEADS * HEAD_DIM    # 256
    Wq = torch.randn(HIDDEN, HIDDEN,   dtype=torch.bfloat16, device=dev) * 0.02
    Wk = torch.randn(HIDDEN, KV_DIM,   dtype=torch.bfloat16, device=dev) * 0.02
    Wv = torch.randn(HIDDEN, KV_DIM,   dtype=torch.bfloat16, device=dev) * 0.02
    Wo = torch.randn(HIDDEN, HIDDEN,   dtype=torch.bfloat16, device=dev) * 0.02
    # MoE gate: [HIDDEN, NUM_EXPERTS]
    Wg = torch.randn(HIDDEN, NUM_EXPERTS, dtype=torch.bfloat16, device=dev) * 0.02
    # RMS-norm weight (per-layer; one shared copy is fine for timing)
    Wn = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)

    # ── One forward pass through all 60 layers ────────────────────────────────
    def forward_pass():
        hs = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1

        # CUDA events: [layer][phase_start, phase_end]
        # phases: 0=qkv, 1=sdpa, 2=oproj, 3=gate, 4=moe, 5=total_end
        evts = [[evt() for _ in range(6)] for _ in range(NUM_LAYERS)]

        for layer in range(NUM_LAYERS):
            e = evts[layer]

            # ── Residual + RMSNorm ──────────────────────────────────────────
            # (fast, not timed separately)
            normed = F.rms_norm(hs, (HIDDEN,), Wn)

            # ── Attention ───────────────────────────────────────────────────
            e[0].record()
            q = normed @ Wq                              # [T, H]
            k = normed @ Wk                              # [T, KV_DIM]
            v = normed @ Wv                              # [T, KV_DIM]

            e[1].record()
            # Reshape for SDPA: [1, heads, T, head_dim]
            q = q.view(1, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(1, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(1, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            # GQA: repeat K,V
            k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
            v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).reshape(T, HIDDEN)

            e[2].record()
            hs = hs + attn_out @ Wo                      # residual

            # ── MoE routing ─────────────────────────────────────────────────
            normed2 = F.rms_norm(hs, (HIDDEN,), Wn)
            e[3].record()
            logits = normed2 @ Wg                        # [T, NUM_EXPERTS]
            topk_w, topk_ids = torch.topk(logits, TOP_K, dim=-1)
            topk_w = topk_w.softmax(dim=-1).float()
            topk_ids = topk_ids.int()

            # ── mega_moe ─────────────────────────────────────────────────────
            e[4].record()
            moe_out = moe_fwd(normed2, topk_w, topk_ids, l1_w, l2_w, buf)
            e[5].record()

            hs = hs + moe_out                            # residual

        return hs, evts

    # ── Warmup ───────────────────────────────────────────────────────────────
    r0(f"\nWarmup (2 full forward passes) ...")
    for _ in range(2):
        forward_pass()
    torch.cuda.synchronize()
    dist.barrier()

    # ── Timed run ─────────────────────────────────────────────────────────────
    r0("Timed forward pass (all 60 layers) ...")
    t_wall0 = time.perf_counter()
    _, evts = forward_pass()
    torch.cuda.synchronize()
    t_wall = (time.perf_counter() - t_wall0) * 1e3

    if rank == 0:
        # Collect timings from CUDA events
        t_qkv   = [elapsed(evts[l][0], evts[l][1]) for l in range(NUM_LAYERS)]
        t_sdpa  = [elapsed(evts[l][1], evts[l][2]) for l in range(NUM_LAYERS)]
        t_oproj = [elapsed(evts[l][2], evts[l][3]) for l in range(NUM_LAYERS)]
        t_gate  = [elapsed(evts[l][3], evts[l][4]) for l in range(NUM_LAYERS)]
        t_moe   = [elapsed(evts[l][4], evts[l][5]) for l in range(NUM_LAYERS)]
        t_total = [t_qkv[l]+t_sdpa[l]+t_oproj[l]+t_gate[l]+t_moe[l]
                   for l in range(NUM_LAYERS)]

        def avg(lst): return sum(lst) / len(lst)
        def sm(lst):  return sum(lst)

        print("\n" + "=" * 80)
        print("Qwen3.5-397B-A17B  ·  Full 60-Layer Forward Pass Timeline")
        print(f"  EP={ws}, tokens={T}, experts={NUM_EXPERTS} ({E_loc}/rank), top_k={TOP_K}")
        print(f"  Attention: random weights (same shape as real model)")
        print(f"  MoE: real FP8→FP4 layer-1 weights (representative of all layers)")
        print("=" * 80)

        print(f"\n  {'Layer':>5}  {'QKV':>8}  {'SDPA':>8}  {'Oproj':>8}  "
              f"{'Gate':>7}  {'mega_moe':>9}  {'Total':>8}  ms")
        print("  " + "-" * 72)
        for l in range(NUM_LAYERS):
            mark = ""
            if l == 0 or l == NUM_LAYERS - 1 or (l + 1) % 10 == 0:
                print(f"  {l:>5}  {t_qkv[l]:>8.2f}  {t_sdpa[l]:>8.2f}  "
                      f"{t_oproj[l]:>8.2f}  {t_gate[l]:>7.2f}  "
                      f"{t_moe[l]:>9.2f}  {t_total[l]:>8.2f}")
            elif l == 1:
                print("  " + " " * 5 + "   ...")

        print("  " + "-" * 72)
        print(f"  {'AVG':>5}  {avg(t_qkv):>8.2f}  {avg(t_sdpa):>8.2f}  "
              f"{avg(t_oproj):>8.2f}  {avg(t_gate):>7.2f}  "
              f"{avg(t_moe):>9.2f}  {avg(t_total):>8.2f}  (per layer)")
        print(f"  {'SUM':>5}  {sm(t_qkv):>8.1f}  {sm(t_sdpa):>8.1f}  "
              f"{sm(t_oproj):>8.1f}  {sm(t_gate):>7.1f}  "
              f"{sm(t_moe):>9.1f}  {sm(t_total):>8.1f}  (60-layer total, ms)")

        print(f"\n  Wall-clock (60 layers, Python+CUDA): {t_wall:.1f} ms")

        # MoE vs Attention breakdown
        attn_total = sm(t_qkv) + sm(t_sdpa) + sm(t_oproj)
        moe_total  = sm(t_gate) + sm(t_moe)
        print(f"\n  Compute breakdown (60-layer totals):")
        print(f"    Attention (QKV+SDPA+Oproj): {attn_total:.1f} ms  "
              f"({100*attn_total/sm(t_total):.1f}%)")
        print(f"    MoE (Gate+mega_moe):         {moe_total:.1f} ms  "
              f"({100*moe_total/sm(t_total):.1f}%)")

        # Cumulative timeline
        print(f"\n  Cumulative timeline (every 10 layers):")
        cumul = 0.0
        print(f"    {'Layer':>5}  {'Cumulative (ms)':>16}  {'MoE cumul (ms)':>16}")
        cumul_moe = 0.0
        for l in range(NUM_LAYERS):
            cumul     += t_total[l]
            cumul_moe += t_moe[l]
            if (l + 1) % 10 == 0 or l == 0:
                print(f"    {l:>5}  {cumul:>16.1f}  {cumul_moe:>16.1f}")

        print("=" * 80)

    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
