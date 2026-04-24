"""Baseline 64K prefill benchmark: Qwen3.5-397B-A17B FP8, NO mega_moe.

Implements standard MoE path (EP all-to-all dispatch + FP8 grouped GEMM + combine)
and attention (GQA) for parallelism modes on 4 GPUs.

Modes:
  --mode 4tp        : TP=4, EP=4. All 65536 tokens replicated per GPU, heads split (8/GPU).
                      Expert GEMM: FP8, 65536×top_k slots dispatched.
  --mode 4dp        : DP=4, EP=4. Each GPU handles its OWN 65536-token request.
                      Expert GEMM: FP8, 65536×top_k slots dispatched per GPU.
  --mode 4cp-single : CP=4 (ring-attention), EP=4.
                      Single 65536-token request split to 16384 tokens/GPU.
                      Expert GEMM: FP8, 16384×top_k slots dispatched.
  --mode 4cp-full   : CP=4+DP=4, EP=4.
                      Each GPU handles its OWN 65536-token request with ring-attn.
                      Expert GEMM: FP8, 65536×top_k slots dispatched per GPU.

Usage:
  torchrun --nproc_per_node=4 --master_port=29601 \\
    benchmark/bench_baseline_4gpu.py --mode 4tp
  torchrun --nproc_per_node=4 --master_port=29602 \\
    benchmark/bench_baseline_4gpu.py --mode 4dp
  torchrun --nproc_per_node=4 --master_port=29603 \\
    benchmark/bench_baseline_4gpu.py --mode 4cp-single
  torchrun --nproc_per_node=4 --master_port=29604 \\
    benchmark/bench_baseline_4gpu.py --mode 4cp-full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

# ── Model constants ────────────────────────────────────────────────────────────
MODEL_DIR    = "/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8"
HIDDEN       = 4096
INTER        = 1024
NUM_EXPERTS  = 512
TOP_K        = 10
NUM_LAYERS   = 60
SEQ_LEN      = 65536
NUM_HEADS    = 32
NUM_KV_HEADS = 2
HEAD_DIM     = HIDDEN // NUM_HEADS   # 128


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def r0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


# ── UE8M0 scale packing (ported from rtp_llm/models_py/kernels/cuda/deepgemm_wrapper.py) ─

@triton.jit
def _pack_ue8m0_gran1(
    sp, op, M, K, Kp, sb, sm, sk, ob, ok, om,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Pack float32 UE8M0 scale [B, M, K//128] → int32 col-major [B, M, K//128//4], gran_mn=1."""
    pm = tl.program_id(0); pk = tl.program_id(1); pb = tl.program_id(2)
    om_ = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    ok_ = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    k4  = ok_[:, None] * 4 + tl.arange(0, 4)[None, :]
    v   = tl.load(sp + pb*sb + om_[:, None, None]*sm + k4[None, :, :]*sk,
                  mask=(om_ < M)[:, None, None] & (k4 < K)[None, :, :], other=0.0)
    exp = ((v.to(tl.int32, bitcast=True) >> 23) & 0xFF)
    pkg = tl.sum(exp << (tl.arange(0, 4)[None, None, :] * 8), axis=2).to(tl.int32)
    tl.store(op + pb*ob + ok_[None, :]*ok + om_[:, None]*om, pkg,
             mask=(om_ < M)[:, None] & (ok_ < Kp)[None, :])


@triton.jit
def _pack_ue8m0_gran128(
    sp, op, M, K, Kp, sb, sm, sk, ob, ok, om,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Pack float32 UE8M0 scale [B, N//128, K//128] → int32 col-major [B, N, K//128//4], gran_mn=128."""
    pm = tl.program_id(0); pk = tl.program_id(1); pb = tl.program_id(2)
    om_ = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    ok_ = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    k4  = ok_[:, None] * 4 + tl.arange(0, 4)[None, :]
    row = om_ // 128
    v   = tl.load(sp + pb*sb + row[:, None, None]*sm + k4[None, :, :]*sk,
                  mask=(om_ < M)[:, None, None] & (k4 < K)[None, :, :], other=0.0)
    exp = ((v.to(tl.int32, bitcast=True) >> 23) & 0xFF)
    pkg = tl.sum(exp << (tl.arange(0, 4)[None, None, :] * 8), axis=2).to(tl.int32)
    tl.store(op + pb*ob + ok_[None, :]*ok + om_[:, None]*om, pkg,
             mask=(om_ < M)[:, None] & (ok_ < Kp)[None, :])


def pack_ue8m0(scale: torch.Tensor, gran_mn: int) -> torch.Tensor:
    """Convert float32 UE8M0 scale to packed int32 (TMA column-major layout).

    gran_mn=1  : per-token scale [B, M, K//128] → [B, M, K//128//4] int32
    gran_mn=128: per-block scale [B, N//128, K//128] → [B, N, K//128//4] int32
    """
    import deep_gemm
    squeeze = scale.dim() == 2
    if squeeze:
        scale = scale.unsqueeze(0)
    B, Ms, K = scale.shape
    M = Ms * gran_mn
    aligned_mn = deep_gemm.get_tma_aligned_size(M, 4)
    Kp = (K + 3) // 4
    storage = torch.zeros((B, Kp, aligned_mn), device=scale.device, dtype=torch.int32)
    packed  = storage.transpose(1, 2)   # [B, aligned_mn, Kp] — col-major strides
    BM, BK  = 64, 32
    grid    = (triton.cdiv(M, BM), triton.cdiv(Kp, BK), B)
    if gran_mn == 1:
        _pack_ue8m0_gran1[grid](
            scale, packed, M, K, Kp,
            scale.stride(0), scale.stride(1), scale.stride(2),
            packed.stride(0), packed.stride(2), packed.stride(1),
            BLOCK_M=BM, BLOCK_K=BK, num_warps=4, num_stages=2,
        )
    else:
        _pack_ue8m0_gran128[grid](
            scale, packed, M, K, Kp,
            scale.stride(0), scale.stride(1), scale.stride(2),
            packed.stride(0), packed.stride(2), packed.stride(1),
            BLOCK_M=BM, BLOCK_K=BK, num_warps=4, num_stages=2,
        )
    res = packed[:, :M, :]
    return res.squeeze(0) if squeeze else res


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round each value up to the nearest power-of-2 (UE8M0 format)."""
    return torch.pow(2.0, torch.ceil(torch.log2(x.clamp(min=1e-38))))


# ── FP8 weight loading (no dequant) ──────────────────────────────────────────

def load_layer1_experts_fp8(
    expert_start: int, expert_end: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load FP8 expert weights directly from checkpoint (no dequantization).

    Returns:
        w1_fp8:  [E, 2*INTER, HIDDEN] float8_e4m3fn
        w1_sf:   [E, 2*INTER, HIDDEN//128//4] int32  (UE8M0 packed)
        w2_fp8:  [E, HIDDEN, INTER]   float8_e4m3fn
        w2_sf:   [E, HIDDEN, INTER//128//4]   int32  (UE8M0 packed)
    """
    import safetensors.torch as st
    idx  = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    pfx  = "model.language_model.layers.1.mlp.experts."
    E    = expert_end - expert_start

    w1_fp8 = torch.zeros(E, 2 * INTER, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    w2_fp8 = torch.zeros(E, HIDDEN, INTER,      dtype=torch.float8_e4m3fn, device=device)
    # Scale: [E, N//128, K//128] float32  (will be packed later)
    w1_scale = torch.ones(E, (2*INTER)//128, HIDDEN//128, dtype=torch.float32, device=device)
    w2_scale = torch.ones(E, HIDDEN//128,    INTER//128,  dtype=torch.float32, device=device)

    shards: dict = {}
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
            return sd[kw].to(device), sd[ks].to(device)

        gate_fp8, gate_si = _ld("gate_proj")   # [INTER, H] fp8, [I//128, H//128] bf16
        up_fp8,   up_si   = _ld("up_proj")
        down_fp8, down_si = _ld("down_proj")   # [H, I] fp8, [H//128, I//128] bf16

        w1_fp8[le, :INTER]  = gate_fp8
        w1_fp8[le, INTER:]  = up_fp8
        # Merge gate/up scale into w1_scale (scale_inv is dequant scale)
        w1_scale[le, :INTER//128]       = ceil_to_ue8m0(gate_si.float())
        w1_scale[le, INTER//128:]       = ceil_to_ue8m0(up_si.float())
        w2_fp8[le]          = down_fp8
        w2_scale[le]        = ceil_to_ue8m0(down_si.float())

    # Pack scales to UE8M0 int32 column-major format
    w1_sf = pack_ue8m0(w1_scale, gran_mn=128)   # [E, 2*INTER, HIDDEN//128//4] int32
    w2_sf = pack_ue8m0(w2_scale, gran_mn=128)   # [E, HIDDEN,  INTER//128//4]  int32

    return w1_fp8, w1_sf, w2_fp8, w2_sf


# ── FP8 activation quantization ───────────────────────────────────────────────

def quant_act_fp8(x_bf16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize [E, M, K] bfloat16 → fp8 + packed int32 scale [E, M, K//128//4]."""
    from deep_gemm.utils.math import per_token_cast_to_fp8
    E = x_bf16.shape[0]
    fp8_list, sf_list = [], []
    for i in range(E):
        xf, xs = per_token_cast_to_fp8(x_bf16[i].float(), use_ue8m0=True)
        fp8_list.append(xf)
        sf_list.append(xs)
    x_fp8 = torch.stack(fp8_list)
    x_sf  = torch.stack(sf_list)        # [E, M, K//128] float32 (gran_mn=1)
    return x_fp8, pack_ue8m0(x_sf, gran_mn=1)  # [E, M, K//128//4] int32


# ── FP8 expert GEMM ──────────────────────────────────────────────────────────

def expert_fp8_gemm(
    recv_x:   torch.Tensor,         # [E, expected_M, K] fp8
    x_sf:     torch.Tensor,         # [E, expected_M, K//128//4] int32
    w1_fp8:   torch.Tensor,         # [E, 2*INTER, K] fp8
    w1_sf:    torch.Tensor,         # [E, 2*INTER, K//128//4] int32
    w2_fp8:   torch.Tensor,         # [E, H, INTER] fp8
    w2_sf:    torch.Tensor,         # [E, H, INTER//128//4] int32
    masked_m: torch.Tensor,         # [E] int32 — actual tokens per expert
    expected_m: int,                # max tokens per expert (padded)
) -> torch.Tensor:
    """Grouped FP8 MoE expert GEMM with masked layout.

    Returns expert_out [E, expected_M, H] bfloat16.
    """
    import deep_gemm
    E   = w2_fp8.shape[0]
    H   = w2_fp8.shape[1]  # N dim of down-proj (= HIDDEN)
    N2  = w1_fp8.shape[1]  # 2 * INTER

    gate_up = torch.zeros(E, expected_m, N2, dtype=torch.bfloat16, device=recv_x.device)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (recv_x, x_sf), (w1_fp8, w1_sf), gate_up, masked_m, expected_m,
        disable_ue8m0_cast=False,
    )

    gate, up = gate_up.chunk(2, dim=-1)
    h = F.silu(gate) * up                      # [E, expected_M, INTER]

    # Quantize intermediate for L2
    h_fp8, h_sf = quant_act_fp8(h)

    out = torch.zeros(E, expected_m, H, dtype=torch.bfloat16, device=recv_x.device)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (h_fp8, h_sf), (w2_fp8, w2_sf), out, masked_m, expected_m,
        disable_ue8m0_cast=False,
    )
    return out


# ── EP dispatch / combine ─────────────────────────────────────────────────────

def ep_dispatch_fp8(
    x: torch.Tensor,               # [T, H] bfloat16
    topk_ids: torch.Tensor,        # [T, K] int32
    ws: int,
    rank: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, dict]:
    """All-to-all dispatch, returns fp8-quantized recv tensors in masked layout.

    Returns:
        recv_fp8:   [E_loc, expected_m, H] float8_e4m3fn  (zero-padded)
        recv_sf:    [E_loc, expected_m, H//128//4] int32
        masked_m:   [E_loc] int32  — actual token count per expert
        expected_m: int            — max tokens per expert (for masked GEMM)
        send_meta:  dict for combine step
    """
    from deep_gemm.utils.math import per_token_cast_to_fp8
    T, H = x.shape
    K     = topk_ids.shape[1]
    E_loc = NUM_EXPERTS // ws

    flat_ids    = topk_ids.view(-1)                                         # [T*K]
    flat_x      = x.unsqueeze(1).expand(-1, K, -1).reshape(T * K, H)       # [T*K, H]
    expert_rank = flat_ids // E_loc
    local_eid   = flat_ids % E_loc

    order     = torch.argsort(expert_rank.long(), stable=True)
    sorted_x   = flat_x[order]
    sorted_eid = local_eid[order]
    sorted_dr  = expert_rank[order]

    send_counts = torch.zeros(ws, dtype=torch.int64, device=device)
    for r in range(ws):
        send_counts[r] = (sorted_dr == r).sum()
    recv_counts = torch.zeros(ws, dtype=torch.int64, device=device)
    dist.all_to_all_single(recv_counts, send_counts)

    total_recv   = recv_counts.sum().item()
    recv_tokens  = torch.empty(total_recv, H, dtype=x.dtype, device=device)
    recv_eid_all = torch.empty(total_recv, dtype=torch.int32, device=device)
    dist.all_to_all_single(recv_tokens, sorted_x,
                           output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())
    dist.all_to_all_single(recv_eid_all, sorted_eid,
                           output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())

    # Sort received tokens by local expert ID for contiguous per-expert blocks
    sort_ord   = torch.argsort(recv_eid_all, stable=True)
    recv_sorted = recv_tokens[sort_ord]    # [total_recv, H], sorted by expert
    eid_sorted  = recv_eid_all[sort_ord]

    # Tokens per expert (vectorized)
    tokens_per_expert = torch.bincount(eid_sorted.int(), minlength=E_loc)  # [E_loc]
    expected_m = int(tokens_per_expert.max().item())
    if expected_m == 0:
        expected_m = 1

    # Quantize all received tokens in ONE call (much faster than per-expert loop)
    K_scale = H // 128
    if total_recv > 0:
        fp8_flat, sf_flat = per_token_cast_to_fp8(
            recv_sorted.float(), use_ue8m0=True)   # [total_recv, H], [total_recv, H//128]
    else:
        fp8_flat = torch.zeros(0, H,       dtype=torch.float8_e4m3fn, device=device)
        sf_flat  = torch.zeros(0, K_scale, dtype=torch.float32,        device=device)

    # Build padded [E_loc, expected_m, H] / [E_loc, expected_m, K_scale] tensors
    # then pack scales once → TMA-aligned column-major int32 (required by GEMM kernel)
    recv_fp8 = torch.zeros(E_loc, expected_m, H,       dtype=torch.float8_e4m3fn, device=device)
    sf_3d    = torch.zeros(E_loc, expected_m, K_scale, dtype=torch.float32,        device=device)
    offset = 0
    for e in range(E_loc):
        cnt = int(tokens_per_expert[e].item())
        if cnt > 0:
            recv_fp8[e, :cnt] = fp8_flat[offset:offset + cnt]
            sf_3d[e,   :cnt]  = sf_flat[offset:offset + cnt]
        offset += cnt
    # pack_ue8m0 produces [E_loc, expected_m, K_scale//4] with TMA col-major strides
    recv_sf = pack_ue8m0(sf_3d, gran_mn=1)

    send_meta = {
        "send_counts": send_counts,
        "recv_counts": recv_counts,
        "order":       order,
        "n_flat":      flat_x.shape[0],
    }
    return recv_fp8, recv_sf, tokens_per_expert.int(), expected_m, send_meta


def ep_combine(
    expert_out:  torch.Tensor,     # [E_loc, expected_m, H]
    tokens_per_expert: torch.Tensor,  # [E_loc] int32
    send_meta:   dict,
    topk_weights: torch.Tensor,    # [T, K] float32
) -> torch.Tensor:
    """Gather expert outputs back and accumulate weighted sum."""
    sc   = send_meta["send_counts"]
    rc   = send_meta["recv_counts"]
    order = send_meta["order"]
    n_flat = send_meta["n_flat"]
    T, K  = topk_weights.shape
    H     = expert_out.shape[-1]
    E_loc = expert_out.shape[0]

    # Re-assemble flat received tensor [total_recv, H]
    total_recv = int(rc.sum())
    flat_recv  = torch.empty(total_recv, H, dtype=expert_out.dtype, device=expert_out.device)
    counts_cpu = tokens_per_expert.cpu().tolist()  # 1 GPU sync instead of E_loc
    offset = 0
    for e, cnt in enumerate(counts_cpu):
        cnt = int(cnt)
        if cnt > 0:
            flat_recv[offset:offset + cnt] = expert_out[e, :cnt]
        offset += cnt

    # All-to-all: send results back
    recv_buf = torch.empty(n_flat, H, dtype=expert_out.dtype, device=expert_out.device)
    dist.all_to_all_single(recv_buf, flat_recv,
                           output_split_sizes=sc.tolist(),
                           input_split_sizes=rc.tolist())

    unsorted = torch.empty_like(recv_buf)
    unsorted[order] = recv_buf
    out = (unsorted.view(T, K, H) * topk_weights.float().unsqueeze(-1)).sum(dim=1)
    return out.bfloat16()


# ── CUDA event helpers ────────────────────────────────────────────────────────

def evt():   return torch.cuda.Event(enable_timing=True)
def ms(s, e): return s.elapsed_time(e)


class LayerEvents:
    def __init__(self):
        self.e = [evt() for _ in range(8)]

    def record(self, i): self.e[i].record()

    def times(self):
        return (
            ms(self.e[0], self.e[1]),   # qkv
            ms(self.e[1], self.e[2]),   # sdpa
            ms(self.e[2], self.e[3]),   # oproj
            ms(self.e[3], self.e[4]),   # gate
            ms(self.e[4], self.e[5]),   # dispatch
            ms(self.e[5], self.e[6]),   # expert_gemm
            ms(self.e[6], self.e[7]),   # combine
        )


# ── Layer forward implementations ─────────────────────────────────────────────

def moe_fp8_layer(
    x: torch.Tensor,               # [T_loc, H] bfloat16  — normed hidden
    topk_ids: torch.Tensor,        # [T_loc, K] int32
    topk_weights: torch.Tensor,    # [T_loc, K] float32
    w1_fp8, w1_sf, w2_fp8, w2_sf,
    ws: int, rank: int, device: torch.device,
) -> Tuple[torch.Tensor, float, float, float]:
    """Standard MoE: EP dispatch → FP8 expert GEMM → EP combine.

    Returns (output, dispatch_ms, gemm_ms, combine_ms).
    """
    e0 = evt(); e1 = evt(); e2 = evt(); e3 = evt()
    e0.record()
    recv_fp8, recv_sf, masked_m, expected_m, send_meta = ep_dispatch_fp8(
        x, topk_ids, ws, rank, device)
    e1.record()
    expert_out = expert_fp8_gemm(recv_fp8, recv_sf, w1_fp8, w1_sf, w2_fp8, w2_sf, masked_m, expected_m)
    e2.record()
    out = ep_combine(expert_out, masked_m, send_meta, topk_weights)
    e3.record()
    torch.cuda.synchronize()
    return out, ms(e0, e1), ms(e1, e2), ms(e2, e3)


def forward_4tp(hs: torch.Tensor, Wq, Wk, Wv, Wo, Wg, Wn,
                w1_fp8, w1_sf, w2_fp8, w2_sf,
                ws: int, rank: int, dev) -> Tuple[torch.Tensor, LayerEvents]:
    """TP=4: full T tokens per GPU, heads split 8/GPU, EP all-to-all on full T."""
    ev = LayerEvents()
    T  = hs.shape[0]

    normed = F.rms_norm(hs, (HIDDEN,), Wn)
    ev.record(0)
    q = normed @ Wq; k = normed @ Wk; v = normed @ Wv   # [T, H/4], [T, KV_DIM]

    ev.record(1)
    H_loc    = HIDDEN // ws
    heads_tp = H_loc // HEAD_DIM
    q = q.view(1, T, heads_tp,    HEAD_DIM).transpose(1, 2)
    k = k.view(1, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(1, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.repeat_interleave(heads_tp // NUM_KV_HEADS, dim=1)
    v = v.repeat_interleave(heads_tp // NUM_KV_HEADS, dim=1)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_out = attn_out.transpose(1, 2).reshape(T, H_loc)

    ev.record(2)
    proj_out = attn_out @ Wo
    dist.all_reduce(proj_out)
    hs = hs + proj_out

    normed2  = F.rms_norm(hs, (HIDDEN,), Wn)
    ev.record(3)
    logits   = normed2 @ Wg
    topk_w, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_w   = topk_w.softmax(dim=-1).float()
    topk_ids = topk_ids.int()

    ev.record(4)
    recv_fp8, recv_sf, masked_m, expected_m, sm = ep_dispatch_fp8(
        normed2, topk_ids, ws, rank, dev)
    ev.record(5)
    expert_out = expert_fp8_gemm(recv_fp8, recv_sf, w1_fp8, w1_sf, w2_fp8, w2_sf, masked_m, expected_m)
    ev.record(6)
    moe_out = ep_combine(expert_out, masked_m, sm, topk_w)
    ev.record(7)

    hs = hs + moe_out
    return hs, ev


def forward_4dp(hs: torch.Tensor, Wq, Wk, Wv, Wo, Wg, Wn,
                w1_fp8, w1_sf, w2_fp8, w2_sf,
                ws: int, rank: int, dev) -> Tuple[torch.Tensor, LayerEvents]:
    """DP=4: each rank processes its OWN T_local tokens (full SDPA), EP dispatch."""
    ev = LayerEvents()
    T_loc = hs.shape[0]

    normed = F.rms_norm(hs, (HIDDEN,), Wn)
    ev.record(0)
    q = normed @ Wq; k = normed @ Wk; v = normed @ Wv

    ev.record(1)
    q = q.view(1, T_loc, NUM_HEADS,    HEAD_DIM).transpose(1, 2)
    k = k.view(1, T_loc, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(1, T_loc, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_out = attn_out.transpose(1, 2).reshape(T_loc, HIDDEN)

    ev.record(2)
    hs = hs + attn_out @ Wo   # no all-reduce in DP

    normed2  = F.rms_norm(hs, (HIDDEN,), Wn)
    ev.record(3)
    logits   = normed2 @ Wg
    topk_w, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_w   = topk_w.softmax(dim=-1).float()
    topk_ids = topk_ids.int()

    ev.record(4)
    recv_fp8, recv_sf, masked_m, expected_m, sm = ep_dispatch_fp8(
        normed2, topk_ids, ws, rank, dev)
    ev.record(5)
    expert_out = expert_fp8_gemm(recv_fp8, recv_sf, w1_fp8, w1_sf, w2_fp8, w2_sf, masked_m, expected_m)
    ev.record(6)
    moe_out = ep_combine(expert_out, masked_m, sm, topk_w)
    ev.record(7)

    hs = hs + moe_out
    return hs, ev


def forward_4cp(hs: torch.Tensor, Wq, Wk, Wv, Wo, Wg, Wn,
                w1_fp8, w1_sf, w2_fp8, w2_sf,
                ws: int, rank: int, dev,
                ring_attn: bool = True) -> Tuple[torch.Tensor, LayerEvents]:
    """CP=4 (ring-attention) + TP=4 + EP=4.
    ring_attn=True : ring-attn with ws steps (single request, T_loc=16K)
    ring_attn=False: plain full SDPA on T_loc (each rank its own request)
    """
    ev = LayerEvents()
    T_loc     = hs.shape[0]
    H_per     = HIDDEN // ws          # 1024 (TP split)
    heads_per = NUM_HEADS // ws       # 8 heads per GPU

    normed = F.rms_norm(hs, (HIDDEN,), Wn)
    ev.record(0)
    q = normed @ Wq; k = normed @ Wk; v = normed @ Wv

    ev.record(1)
    q4 = q.view(1, T_loc, heads_per,    HEAD_DIM).transpose(1, 2)
    k4 = k.view(1, T_loc, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v4 = v.view(1, T_loc, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    if ring_attn:
        k_cur = k4.contiguous(); v_cur = v4.contiguous()
        acc   = torch.zeros(1, heads_per, T_loc, HEAD_DIM, dtype=hs.dtype, device=dev)
        for step in range(ws):
            k_r = k_cur.repeat_interleave(heads_per // NUM_KV_HEADS, dim=1)
            v_r = v_cur.repeat_interleave(heads_per // NUM_KV_HEADS, dim=1)
            acc = acc + F.scaled_dot_product_attention(q4, k_r, v_r, is_causal=(step == 0))
            if step < ws - 1:
                sk, sv = k_cur.clone(), v_cur.clone()
                rk, rv = torch.empty_like(k_cur), torch.empty_like(v_cur)
                nxt = (rank + 1) % ws; prv = (rank - 1) % ws
                reqs = dist.batch_isend_irecv([
                    dist.P2POp(dist.isend, sk, nxt), dist.P2POp(dist.irecv, rk, prv),
                    dist.P2POp(dist.isend, sv, nxt), dist.P2POp(dist.irecv, rv, prv),
                ])
                for r in reqs: r.wait()
                k_cur = rk; v_cur = rv
        attn_out = acc.transpose(1, 2).reshape(T_loc, H_per)
    else:
        k4 = k4.repeat_interleave(heads_per // NUM_KV_HEADS, dim=1)
        v4 = v4.repeat_interleave(heads_per // NUM_KV_HEADS, dim=1)
        attn_out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(T_loc, H_per)

    ev.record(2)
    proj_out = attn_out @ Wo
    dist.all_reduce(proj_out)
    hs = hs + proj_out

    normed2  = F.rms_norm(hs, (HIDDEN,), Wn)
    ev.record(3)
    logits   = normed2 @ Wg
    topk_w, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_w   = topk_w.softmax(dim=-1).float()
    topk_ids = topk_ids.int()

    ev.record(4)
    recv_fp8, recv_sf, masked_m, expected_m, sm = ep_dispatch_fp8(
        normed2, topk_ids, ws, rank, dev)
    ev.record(5)
    expert_out = expert_fp8_gemm(recv_fp8, recv_sf, w1_fp8, w1_sf, w2_fp8, w2_sf, masked_m, expected_m)
    ev.record(6)
    moe_out = ep_combine(expert_out, masked_m, sm, topk_w)
    ev.record(7)

    hs = hs + moe_out
    return hs, ev


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["4tp", "4dp", "4cp-single", "4cp-full"],
                        required=True,
                        help="4tp|4dp|4cp-single|4cp-full")
    args = parser.parse_args()

    local_rank = setup()
    ws   = dist.get_world_size()
    rank = dist.get_rank()
    dev  = torch.device(f"cuda:{local_rank}")

    import deep_gemm
    cap = torch.cuda.get_device_capability()
    r0(f"GPU: SM{cap[0]}{cap[1]} {torch.cuda.get_device_name(local_rank)} × {ws}")
    r0(f"Mode: {args.mode}  SEQ_LEN={SEQ_LEN}  experts={NUM_EXPERTS} ({NUM_EXPERTS//ws}/rank)")

    if cap[0] < 10:
        print("ERROR: FP8 grouped GEMM (UE8M0) requires SM100+"); sys.exit(1)

    # ── Load FP8 expert weights ───────────────────────────────────────────────
    E_loc   = NUM_EXPERTS // ws
    e_start = rank * E_loc
    r0(f"\nLoading 397B layer-1 FP8 weights ({E_loc} experts/rank, no dequant) ...")
    t0 = time.perf_counter()
    w1_fp8, w1_sf, w2_fp8, w2_sf = load_layer1_experts_fp8(e_start, e_start + E_loc, dev)
    dist.barrier()
    r0(f"  done {time.perf_counter()-t0:.1f}s  "
       f"w1_fp8={tuple(w1_fp8.shape)} w1_sf={tuple(w1_sf.shape)}")

    # ── Random attention / gate weights ──────────────────────────────────────
    H_per  = HIDDEN // ws    # 1024 for TP/CP
    KV_DIM = NUM_KV_HEADS * HEAD_DIM

    use_tp_weights = args.mode in ("4tp", "4cp-single", "4cp-full")
    Wq = torch.randn(HIDDEN, H_per  if use_tp_weights else HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
    Wk = torch.randn(HIDDEN, KV_DIM,  dtype=torch.bfloat16, device=dev) * 0.02
    Wv = torch.randn(HIDDEN, KV_DIM,  dtype=torch.bfloat16, device=dev) * 0.02
    Wo = torch.randn(H_per  if use_tp_weights else HIDDEN, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
    Wg = torch.randn(HIDDEN, NUM_EXPERTS, dtype=torch.bfloat16, device=dev) * 0.02
    Wn = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)

    # ── Token counts per mode ─────────────────────────────────────────────────
    if args.mode == "4cp-single":
        T_local = SEQ_LEN // ws   # 16384
    else:
        T_local = SEQ_LEN         # 65536 for 4tp, 4dp, 4cp-full

    r0(f"T_local={T_local} per GPU")

    def make_input():
        return torch.randn(T_local, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1

    def run_layer(hs):
        ring = args.mode == "4cp-single"
        if args.mode == "4tp":
            return forward_4tp(hs, Wq, Wk, Wv, Wo, Wg, Wn, w1_fp8, w1_sf, w2_fp8, w2_sf, ws, rank, dev)
        elif args.mode == "4dp":
            return forward_4dp(hs, Wq, Wk, Wv, Wo, Wg, Wn, w1_fp8, w1_sf, w2_fp8, w2_sf, ws, rank, dev)
        else:  # 4cp-single or 4cp-full
            return forward_4cp(hs, Wq, Wk, Wv, Wo, Wg, Wn, w1_fp8, w1_sf, w2_fp8, w2_sf, ws, rank, dev, ring_attn=ring)

    def forward_pass():
        all_evs = []
        for _ in range(NUM_LAYERS):
            hs = make_input()   # fresh input each layer; prevents routing blow-up in benchmark
            hs, ev = run_layer(hs)
            all_evs.append(ev)
        return hs, all_evs

    # ── Warmup ────────────────────────────────────────────────────────────────
    r0(f"\nWarmup (2 full forward passes, T_local={T_local}) ...")
    for i in range(2):
        r0(f"  warmup {i+1}/2 ...")
        forward_pass()
        torch.cuda.synchronize(); dist.barrier()
    r0("Warmup done.")

    # ── Timed run ─────────────────────────────────────────────────────────────
    r0("Timed forward pass (60 layers) ...")
    t_wall0 = time.perf_counter()
    _, all_evs = forward_pass()
    torch.cuda.synchronize()
    t_wall = (time.perf_counter() - t_wall0) * 1e3
    dist.barrier()

    if rank == 0:
        t_qkv   = [all_evs[l].times()[0] for l in range(NUM_LAYERS)]
        t_sdpa  = [all_evs[l].times()[1] for l in range(NUM_LAYERS)]
        t_oproj = [all_evs[l].times()[2] for l in range(NUM_LAYERS)]
        t_gate  = [all_evs[l].times()[3] for l in range(NUM_LAYERS)]
        t_disp  = [all_evs[l].times()[4] for l in range(NUM_LAYERS)]
        t_gemm  = [all_evs[l].times()[5] for l in range(NUM_LAYERS)]
        t_comb  = [all_evs[l].times()[6] for l in range(NUM_LAYERS)]
        t_moe   = [t_disp[l] + t_gemm[l] + t_comb[l] for l in range(NUM_LAYERS)]
        t_total = [t_qkv[l]+t_sdpa[l]+t_oproj[l]+t_gate[l]+t_moe[l] for l in range(NUM_LAYERS)]

        def avg(lst): return sum(lst) / len(lst)
        def sm(lst):  return sum(lst)

        mode_desc = {
            "4tp":       f"TP=4, EP=4  (T={T_local}/GPU, {HIDDEN//ws} heads/GPU, FP8 expert GEMM)",
            "4dp":       f"DP=4, EP=4  (T={T_local}/GPU, {NUM_HEADS} heads/GPU, FP8 expert GEMM, each rank own request)",
            "4cp-single":f"CP=4+TP=4, EP=4  (T={T_local}/GPU={SEQ_LEN} total, ring-attn {ws} steps, FP8 expert GEMM)",
            "4cp-full":  f"CP=4+DP=4, EP=4  (T={T_local}/GPU, {HIDDEN//ws} heads/GPU, plain SDPA, FP8 expert GEMM, each rank own request)",
        }

        print("\n" + "=" * 92)
        print(f"Qwen3.5-397B-A17B  ·  Baseline FP8 (NO mega_moe)  ·  {args.mode.upper()}")
        print(f"  Config: {mode_desc[args.mode]}")
        print(f"  Weights: real FP8 from checkpoint, UE8M0 repacked (no dequant)")
        print(f"  Attn: random weights, same GEMM shapes as real model")
        print("=" * 92)

        print(f"\n  {'Layer':>5}  {'QKV':>7}  {'SDPA':>8}  {'Oproj':>7}  "
              f"{'Gate':>6}  {'Dispatch':>9}  {'ExpertGEMM':>11}  {'Combine':>8}  "
              f"{'MoE':>7}  {'Total':>8}  ms")
        print("  " + "-" * 90)
        for l in range(NUM_LAYERS):
            if l == 0 or l == NUM_LAYERS-1 or (l+1) % 10 == 0:
                print(f"  {l:>5}  {t_qkv[l]:>7.2f}  {t_sdpa[l]:>8.2f}  {t_oproj[l]:>7.2f}  "
                      f"{t_gate[l]:>6.2f}  {t_disp[l]:>9.2f}  {t_gemm[l]:>11.2f}  "
                      f"{t_comb[l]:>8.2f}  {t_moe[l]:>7.2f}  {t_total[l]:>8.2f}")
            elif l == 1:
                print("  " + " " * 5 + "   ...")

        print("  " + "-" * 90)
        print(f"  {'AVG':>5}  {avg(t_qkv):>7.2f}  {avg(t_sdpa):>8.2f}  {avg(t_oproj):>7.2f}  "
              f"{avg(t_gate):>6.2f}  {avg(t_disp):>9.2f}  {avg(t_gemm):>11.2f}  "
              f"{avg(t_comb):>8.2f}  {avg(t_moe):>7.2f}  {avg(t_total):>8.2f}  (per layer)")
        print(f"  {'SUM':>5}  {sm(t_qkv):>7.1f}  {sm(t_sdpa):>8.1f}  {sm(t_oproj):>7.1f}  "
              f"{sm(t_gate):>6.1f}  {sm(t_disp):>9.1f}  {sm(t_gemm):>11.1f}  "
              f"{sm(t_comb):>8.1f}  {sm(t_moe):>7.1f}  {sm(t_total):>8.1f}  (60-layer total)")

        print(f"\n  Wall-clock (60-layer forward pass): {t_wall:.1f} ms")

        attn_total = sm(t_qkv) + sm(t_sdpa) + sm(t_oproj)
        moe_total  = sm(t_gate) + sm(t_moe)
        print(f"\n  Compute breakdown (60-layer totals):")
        print(f"    Attention (QKV+SDPA+Oproj):        {attn_total:>8.1f} ms  ({100*attn_total/sm(t_total):.1f}%)")
        print(f"    MoE (Gate+Dispatch+GEMM+Combine):  {moe_total:>8.1f} ms  ({100*moe_total/sm(t_total):.1f}%)")
        print(f"      ├─ EP Dispatch:    {sm(t_disp):>6.1f} ms  ({100*sm(t_disp)/sm(t_total):.1f}%)")
        print(f"      ├─ Expert GEMMs:   {sm(t_gemm):>6.1f} ms  ({100*sm(t_gemm)/sm(t_total):.1f}%)")
        print(f"      └─ EP Combine:     {sm(t_comb):>6.1f} ms  ({100*sm(t_comb)/sm(t_total):.1f}%)")

        print(f"\n  Cumulative timeline (every 10 layers):")
        cumul = 0.0
        for l in range(NUM_LAYERS):
            cumul += t_total[l]
            if l == 0 or (l+1) % 10 == 0:
                print(f"    Layer {l:>2}: +{t_total[l]:.2f} ms  →  cumulative {cumul:.1f} ms")

        print("=" * 92)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
