"""Baseline 64K prefill benchmark: Qwen3.5-397B-A17B FP8, NO mega_moe.

Implements standard MoE path (EP all-to-all dispatch + grouped GEMM + combine)
and attention (GQA) for three parallelism modes on 4 GPUs:

  --mode 4tp  : TP=4, EP=4. All 65536 tokens per GPU, heads split (8/GPU).
                MoE: dispatch 65536×top_k tokens, all-to-all EP.
  --mode 4dp  : DP=4, EP=4. Each GPU handles 16384 tokens, full heads (32/GPU).
                MoE: dispatch 16384×top_k tokens per GPU, all-to-all EP.
  --mode 4cp  : CP=4 (ring-attention), TP=4, EP=4.
                Each GPU handles 16384 Q tokens for ring-attention (8 heads/GPU),
                MoE same as 4dp.

Usage:
  torchrun --nproc_per_node=4 --master_port=29601 \
    benchmark/bench_baseline_4gpu.py --mode 4tp
  torchrun --nproc_per_node=4 --master_port=29602 \
    benchmark/bench_baseline_4gpu.py --mode 4dp
  torchrun --nproc_per_node=4 --master_port=29603 \
    benchmark/bench_baseline_4gpu.py --mode 4cp
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


# ── Weight loading (FP8 → BF16) ──────────────────────────────────────────────

def dequant_fp8_block(w_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    s = scale_inv.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return (w_fp8.float() * s).bfloat16()


def load_layer1_experts(expert_start: int, expert_end: int,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    import safetensors.torch as st
    idx  = json.load(open(os.path.join(MODEL_DIR, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    pfx  = "model.language_model.layers.1.mlp.experts."
    E    = expert_end - expert_start
    w1   = torch.zeros(E, 2 * INTER, HIDDEN, dtype=torch.bfloat16, device=device)
    w2   = torch.zeros(E, HIDDEN,    INTER,  dtype=torch.bfloat16, device=device)

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
            return dequant_fp8_block(sd[kw], sd[ks])
        gate, up, down = _ld("gate_proj"), _ld("up_proj"), _ld("down_proj")
        w1[le, :INTER] = gate.to(device)
        w1[le, INTER:] = up.to(device)
        w2[le]         = down.to(device)
    return w1, w2


# ── EP dispatch helpers ───────────────────────────────────────────────────────

def ep_dispatch(x: torch.Tensor,
                topk_ids: torch.Tensor,
                ws: int, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter tokens to their target expert GPUs via all-to-all.

    Returns:
        recv_x:     [total_recv, H] - tokens received by this GPU
        recv_ids:   [total_recv]    - local expert id for each received token
        send_meta:  (send_counts, send_order) for combine step
    """
    T, H = x.shape
    K = topk_ids.shape[1]
    E_per = NUM_EXPERTS // ws

    # Flatten: each token appears K times
    flat_ids = topk_ids.view(-1)                # [T*K]
    flat_x   = x.unsqueeze(1).expand(-1, K, -1).reshape(T * K, H)  # [T*K, H]

    expert_rank = flat_ids // E_per             # which GPU owns this expert
    local_eid   = flat_ids % E_per              # expert index within that GPU

    # Build per-destination sorted order
    order = torch.argsort(expert_rank, stable=True)
    sorted_x     = flat_x[order]
    sorted_eid   = local_eid[order]
    sorted_drank = expert_rank[order]

    send_counts = torch.zeros(ws, dtype=torch.int64, device=x.device)
    for r in range(ws):
        send_counts[r] = (sorted_drank == r).sum()

    recv_counts = torch.zeros(ws, dtype=torch.int64, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts)

    total_recv = recv_counts.sum().item()
    recv_x   = torch.empty(total_recv, H, dtype=x.dtype, device=x.device)
    recv_eid = torch.empty(total_recv, dtype=torch.int64, device=x.device)

    dist.all_to_all_single(recv_x,   sorted_x,   output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())
    dist.all_to_all_single(recv_eid, sorted_eid,  output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())

    send_meta = (send_counts, order, flat_x.shape[0])
    return recv_x, recv_eid, send_meta


def ep_combine(expert_out: torch.Tensor, recv_counts: torch.Tensor,
               send_meta: Tuple, topk_weights: torch.Tensor, ws: int) -> torch.Tensor:
    """Gather expert outputs back and accumulate weighted sum."""
    send_counts, send_order, n_flat = send_meta
    H = expert_out.shape[1]

    # All-to-all: send results back to origin GPUs
    recv_buf = torch.empty(n_flat, H, dtype=expert_out.dtype, device=expert_out.device)
    dist.all_to_all_single(recv_buf, expert_out,
                           output_split_sizes=send_counts.tolist(),
                           input_split_sizes=recv_counts.tolist())

    # Undo the argsort order
    unsorted = torch.empty_like(recv_buf)
    unsorted[send_order] = recv_buf

    T = topk_weights.shape[0]
    K = topk_weights.shape[1]
    # Weighted sum: unsorted is [T*K, H], weights is [T, K]
    out = (unsorted.view(T, K, H) * topk_weights.float().unsqueeze(-1)).sum(dim=1)
    return out.to(expert_out.dtype)


def moe_standard(x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                 w1: torch.Tensor, w2: torch.Tensor, ws: int, rank: int) -> torch.Tensor:
    """Standard (non-fused) MoE forward: dispatch → grouped GEMM → combine."""
    T, H = x.shape
    E_per = NUM_EXPERTS // ws

    # ── EP dispatch ──────────────────────────────────────────────────────────
    recv_x, recv_eid, send_meta = ep_dispatch(x, topk_ids, ws, rank)

    # recv_counts is needed for combine
    send_counts = send_meta[0]
    recv_counts = torch.zeros(ws, dtype=torch.int64, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts)

    # ── Expert GEMM on received tokens ───────────────────────────────────────
    total_recv = recv_x.shape[0]
    expert_out = torch.zeros(total_recv, H, dtype=torch.bfloat16, device=x.device)

    # Group by expert and run matmul
    for eid in range(E_per):
        mask = (recv_eid == eid)
        cnt  = mask.sum().item()
        if cnt == 0:
            continue
        xi = recv_x[mask]               # [cnt, H]
        # L1: gate + up
        gate_up = xi @ w1[eid].T         # [cnt, 2*INTER]
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up            # SwiGLU: [cnt, INTER]
        # L2: down
        expert_out[mask] = h @ w2[eid].T  # [cnt, H]

    # ── EP combine ───────────────────────────────────────────────────────────
    out = ep_combine(expert_out, recv_counts, send_meta, topk_weights, ws)
    return out


# ── Ring-attention helper ─────────────────────────────────────────────────────

def ring_attention_step(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        acc_out: torch.Tensor, acc_lse: torch.Tensor,
                        step: int, total_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """One ring-attn step: compute attention of local Q against current ring K,V.
    Uses causal masking aware of ring step ordering (simplified: no masking per step,
    just measure the compute/comm pattern accurately)."""
    # q: [1, Hq, Tlocal, D], k: [1, Hkv, Tlocal, D] (after repeat to Hq)
    # For timing accuracy we just do the SDPA
    Hq = q.shape[1]
    k_rep = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    v_rep = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=(step == 0))
    return out, acc_lse  # simplified accumulation for timing


# ── CUDA event helpers ────────────────────────────────────────────────────────

def evt(): return torch.cuda.Event(enable_timing=True)
def elapsed(s, e): return s.elapsed_time(e)


# ── Single-layer forward (timed) ──────────────────────────────────────────────

class LayerEvents:
    def __init__(self):
        self.e = [evt() for _ in range(8)]

    def record(self, i): self.e[i].record()

    def times(self):
        t_qkv   = elapsed(self.e[0], self.e[1])
        t_sdpa  = elapsed(self.e[1], self.e[2])
        t_oproj = elapsed(self.e[2], self.e[3])
        t_gate  = elapsed(self.e[3], self.e[4])
        t_disp  = elapsed(self.e[4], self.e[5])
        t_gemm  = elapsed(self.e[5], self.e[6])
        t_comb  = elapsed(self.e[6], self.e[7])
        return t_qkv, t_sdpa, t_oproj, t_gate, t_disp, t_gemm, t_comb


def forward_4tp(hs_global: torch.Tensor, Wq, Wk, Wv, Wo, Wg, Wn,
                w1, w2, ws, rank, dev) -> Tuple[torch.Tensor, LayerEvents]:
    """TP=4 forward: full sequence [T,H] on every GPU, split heads."""
    ev = LayerEvents()
    T  = hs_global.shape[0]
    KV_DIM = NUM_KV_HEADS * HEAD_DIM

    normed = F.rms_norm(hs_global, (HIDDEN,), Wn)

    ev.record(0)
    q = normed @ Wq    # [T, H/4] (local heads)
    k = normed @ Wk    # [T, KV_DIM]
    v = normed @ Wv    # [T, KV_DIM]

    ev.record(1)
    H_local = HIDDEN // ws                     # 1024
    q = q.view(1, T, H_local // HEAD_DIM, HEAD_DIM).transpose(1, 2)   # [1, 8, T, 128]
    k = k.view(1, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)          # [1, 2, T, 128]
    v = v.view(1, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.repeat_interleave(H_local // HEAD_DIM // NUM_KV_HEADS, dim=1)  # [1, 8, T, 128]
    v = v.repeat_interleave(H_local // HEAD_DIM // NUM_KV_HEADS, dim=1)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_out = attn_out.transpose(1, 2).reshape(T, H_local)           # [T, H/4]

    ev.record(2)
    proj_out = attn_out @ Wo                   # [T, H]
    dist.all_reduce(proj_out)                  # TP all-reduce
    hs_global = hs_global + proj_out

    # MoE router
    normed2 = F.rms_norm(hs_global, (HIDDEN,), Wn)
    ev.record(3)
    logits  = normed2 @ Wg                    # [T, NUM_EXPERTS]
    topk_w, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_w  = topk_w.softmax(dim=-1).float()
    topk_ids = topk_ids.int()

    # Standard MoE (split into dispatch, gemm, combine)
    flat_ids = topk_ids.view(-1)
    flat_x   = normed2.unsqueeze(1).expand(-1, TOP_K, -1).reshape(T * TOP_K, HIDDEN)
    E_per = NUM_EXPERTS // ws
    expert_rank = flat_ids // E_per
    local_eid   = flat_ids % E_per
    order = torch.argsort(expert_rank.long(), stable=True)
    sorted_x = flat_x[order]; sorted_eid = local_eid[order]; sorted_dr = expert_rank[order]
    send_counts = torch.zeros(ws, dtype=torch.int64, device=dev)
    for r in range(ws): send_counts[r] = (sorted_dr == r).sum()
    recv_counts = torch.zeros(ws, dtype=torch.int64, device=dev)

    ev.record(4)  # dispatch start
    dist.all_to_all_single(recv_counts, send_counts)
    total_recv = recv_counts.sum().item()
    recv_x   = torch.empty(total_recv, HIDDEN, dtype=normed2.dtype, device=dev)
    recv_eid = torch.empty(total_recv, dtype=torch.int64, device=dev)
    dist.all_to_all_single(recv_x,   sorted_x,   output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())
    dist.all_to_all_single(recv_eid, sorted_eid.long(), output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())

    ev.record(5)  # gemm start
    expert_out = torch.zeros(total_recv, HIDDEN, dtype=torch.bfloat16, device=dev)
    for eid in range(E_per):
        mask = (recv_eid == eid)
        cnt  = mask.sum().item()
        if cnt == 0: continue
        xi = recv_x[mask]
        gu = xi @ w1[eid].T
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        expert_out[mask] = h @ w2[eid].T

    ev.record(6)  # combine start
    recv_buf = torch.empty(T * TOP_K, HIDDEN, dtype=expert_out.dtype, device=dev)
    dist.all_to_all_single(recv_buf, expert_out,
                           output_split_sizes=send_counts.tolist(),
                           input_split_sizes=recv_counts.tolist())
    unsorted = torch.empty_like(recv_buf)
    unsorted[order] = recv_buf
    moe_out = (unsorted.view(T, TOP_K, HIDDEN) * topk_w.unsqueeze(-1)).sum(dim=1).bfloat16()

    ev.record(7)
    hs_global = hs_global + moe_out
    return hs_global, ev


def forward_4dp(hs_local: torch.Tensor, Wq, Wk, Wv, Wo, Wg, Wn,
                w1, w2, ws, rank, dev) -> Tuple[torch.Tensor, LayerEvents]:
    """DP=4 forward: local [T/4, H] sequence, full heads, EP MoE."""
    ev = LayerEvents()
    T_loc = hs_local.shape[0]
    KV_DIM = NUM_KV_HEADS * HEAD_DIM

    normed = F.rms_norm(hs_local, (HIDDEN,), Wn)

    ev.record(0)
    q = normed @ Wq    # [T/4, H]
    k = normed @ Wk    # [T/4, KV_DIM]
    v = normed @ Wv    # [T/4, KV_DIM]

    ev.record(1)
    q = q.view(1, T_loc, NUM_HEADS,    HEAD_DIM).transpose(1, 2)
    k = k.view(1, T_loc, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(1, T_loc, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_out = attn_out.transpose(1, 2).reshape(T_loc, HIDDEN)

    ev.record(2)
    hs_local = hs_local + attn_out @ Wo   # no all-reduce for DP

    # MoE
    normed2  = F.rms_norm(hs_local, (HIDDEN,), Wn)
    ev.record(3)
    logits   = normed2 @ Wg
    topk_w, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_w   = topk_w.softmax(dim=-1).float()
    topk_ids = topk_ids.int()

    E_per = NUM_EXPERTS // ws
    flat_ids = topk_ids.view(-1)
    flat_x   = normed2.unsqueeze(1).expand(-1, TOP_K, -1).reshape(T_loc * TOP_K, HIDDEN)
    expert_rank = (flat_ids // E_per).long()
    local_eid   = (flat_ids % E_per).long()
    order = torch.argsort(expert_rank, stable=True)
    sorted_x = flat_x[order]; sorted_eid = local_eid[order]; sorted_dr = expert_rank[order]
    send_counts = torch.zeros(ws, dtype=torch.int64, device=dev)
    for r in range(ws): send_counts[r] = (sorted_dr == r).sum()
    recv_counts = torch.zeros(ws, dtype=torch.int64, device=dev)

    ev.record(4)
    dist.all_to_all_single(recv_counts, send_counts)
    total_recv = recv_counts.sum().item()
    recv_x   = torch.empty(total_recv, HIDDEN, dtype=normed2.dtype, device=dev)
    recv_eid = torch.empty(total_recv, dtype=torch.int64, device=dev)
    dist.all_to_all_single(recv_x,   sorted_x,   output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())
    dist.all_to_all_single(recv_eid, sorted_eid,  output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())

    ev.record(5)
    expert_out = torch.zeros(total_recv, HIDDEN, dtype=torch.bfloat16, device=dev)
    for eid in range(E_per):
        mask = (recv_eid == eid)
        cnt  = mask.sum().item()
        if cnt == 0: continue
        xi = recv_x[mask]
        gu = xi @ w1[eid].T
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        expert_out[mask] = h @ w2[eid].T

    ev.record(6)
    recv_buf = torch.empty(T_loc * TOP_K, HIDDEN, dtype=expert_out.dtype, device=dev)
    dist.all_to_all_single(recv_buf, expert_out,
                           output_split_sizes=send_counts.tolist(),
                           input_split_sizes=recv_counts.tolist())
    unsorted = torch.empty_like(recv_buf)
    unsorted[order] = recv_buf
    moe_out = (unsorted.view(T_loc, TOP_K, HIDDEN) * topk_w.unsqueeze(-1)).sum(dim=1).bfloat16()

    ev.record(7)
    hs_local = hs_local + moe_out
    return hs_local, ev


def forward_4cp(hs_local: torch.Tensor, Wq, Wk, Wv, Wo, Wg, Wn,
                w1, w2, ws, rank, dev) -> Tuple[torch.Tensor, LayerEvents]:
    """CP=4 (ring-attention) + TP=4 + EP=4 forward.
    Each GPU holds T/4 local tokens, 8 heads (TP split).
    Ring-attention: rotate K,V through ws GPUs.
    MoE: same EP all-to-all as 4dp.
    """
    ev = LayerEvents()
    T_loc = hs_local.shape[0]
    H_per = HIDDEN // ws             # 1024 (TP split)
    heads_per = NUM_HEADS // ws      # 8 heads per GPU
    KV_DIM    = NUM_KV_HEADS * HEAD_DIM

    normed = F.rms_norm(hs_local, (HIDDEN,), Wn)

    ev.record(0)
    q = normed @ Wq    # [T/4, H/4]  local heads
    k = normed @ Wk    # [T/4, KV_DIM]
    v = normed @ Wv    # [T/4, KV_DIM]

    ev.record(1)
    # Ring-attention: ws steps, rotate K,V
    q4 = q.view(1, T_loc, heads_per,    HEAD_DIM).transpose(1, 2)  # [1, 8, T/4, 128]
    k4 = k.view(1, T_loc, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [1, 2, T/4, 128]
    v4 = v.view(1, T_loc, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    k_cur = k4.contiguous()
    v_cur = v4.contiguous()
    acc_out = torch.zeros(1, heads_per, T_loc, HEAD_DIM, dtype=torch.bfloat16, device=dev)

    for step in range(ws):
        # Compute attention of local Q against current ring K,V
        k_rep = k_cur.repeat_interleave(heads_per // NUM_KV_HEADS, dim=1)
        v_rep = v_cur.repeat_interleave(heads_per // NUM_KV_HEADS, dim=1)
        is_causal = (step == 0)  # causal only for local (diagonal) block
        step_out = F.scaled_dot_product_attention(q4, k_rep, v_rep, is_causal=is_causal)
        acc_out = acc_out + step_out  # simplified accumulation (no softmax correction)

        if step < ws - 1:
            # Ring: send K,V to next rank, receive from prev rank
            send_k = k_cur.clone()
            send_v = v_cur.clone()
            recv_k = torch.empty_like(k_cur)
            recv_v = torch.empty_like(v_cur)
            next_rank = (rank + 1) % ws
            prev_rank = (rank - 1) % ws
            ops = [
                dist.P2POp(dist.isend, send_k, next_rank),
                dist.P2POp(dist.irecv, recv_k, prev_rank),
                dist.P2POp(dist.isend, send_v, next_rank),
                dist.P2POp(dist.irecv, recv_v, prev_rank),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs: req.wait()
            k_cur = recv_k
            v_cur = recv_v

    attn_out = acc_out.transpose(1, 2).reshape(T_loc, H_per)  # [T/4, H/4]

    ev.record(2)
    proj_out = attn_out @ Wo   # [T/4, H]
    dist.all_reduce(proj_out)  # TP all-reduce
    hs_local = hs_local + proj_out

    # MoE (same as 4dp: local T/4 tokens dispatched globally)
    normed2  = F.rms_norm(hs_local, (HIDDEN,), Wn)
    ev.record(3)
    logits   = normed2 @ Wg
    topk_w, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_w   = topk_w.softmax(dim=-1).float()
    topk_ids = topk_ids.int()

    E_per = NUM_EXPERTS // ws
    flat_ids = topk_ids.view(-1)
    flat_x   = normed2.unsqueeze(1).expand(-1, TOP_K, -1).reshape(T_loc * TOP_K, HIDDEN)
    expert_rank = (flat_ids // E_per).long()
    local_eid   = (flat_ids % E_per).long()
    order = torch.argsort(expert_rank, stable=True)
    sorted_x = flat_x[order]; sorted_eid = local_eid[order]; sorted_dr = expert_rank[order]
    send_counts = torch.zeros(ws, dtype=torch.int64, device=dev)
    for r in range(ws): send_counts[r] = (sorted_dr == r).sum()
    recv_counts = torch.zeros(ws, dtype=torch.int64, device=dev)

    ev.record(4)
    dist.all_to_all_single(recv_counts, send_counts)
    total_recv = recv_counts.sum().item()
    recv_x   = torch.empty(total_recv, HIDDEN, dtype=normed2.dtype, device=dev)
    recv_eid = torch.empty(total_recv, dtype=torch.int64, device=dev)
    dist.all_to_all_single(recv_x,   sorted_x,   output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())
    dist.all_to_all_single(recv_eid, sorted_eid,  output_split_sizes=recv_counts.tolist(),
                           input_split_sizes=send_counts.tolist())

    ev.record(5)
    expert_out = torch.zeros(total_recv, HIDDEN, dtype=torch.bfloat16, device=dev)
    for eid in range(E_per):
        mask = (recv_eid == eid)
        cnt  = mask.sum().item()
        if cnt == 0: continue
        xi = recv_x[mask]
        gu = xi @ w1[eid].T
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        expert_out[mask] = h @ w2[eid].T

    ev.record(6)
    recv_buf = torch.empty(T_loc * TOP_K, HIDDEN, dtype=expert_out.dtype, device=dev)
    dist.all_to_all_single(recv_buf, expert_out,
                           output_split_sizes=send_counts.tolist(),
                           input_split_sizes=recv_counts.tolist())
    unsorted = torch.empty_like(recv_buf)
    unsorted[order] = recv_buf
    moe_out = (unsorted.view(T_loc, TOP_K, HIDDEN) * topk_w.unsqueeze(-1)).sum(dim=1).bfloat16()

    ev.record(7)
    hs_local = hs_local + moe_out
    return hs_local, ev


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["4tp", "4dp", "4cp"], required=True)
    args = parser.parse_args()

    local_rank = setup()
    ws   = dist.get_world_size()
    rank = dist.get_rank()
    dev  = torch.device(f"cuda:{local_rank}")

    cap = torch.cuda.get_device_capability()
    r0(f"GPU: SM{cap[0]}{cap[1]} {torch.cuda.get_device_name(local_rank)} × {ws}")
    r0(f"Mode: {args.mode}  SEQ_LEN={SEQ_LEN}  experts={NUM_EXPERTS} ({NUM_EXPERTS//ws}/rank)")

    # ── Expert weights (local EP slice) ───────────────────────────────────────
    E_loc = NUM_EXPERTS // ws
    e_start = rank * E_loc
    r0(f"\nLoading 397B layer-1 FP8 weights ({E_loc} experts/rank) ...")
    t0 = time.perf_counter()
    w1, w2 = load_layer1_experts(e_start, e_start + E_loc, dev)  # [E_loc, 2I, H], [E_loc, H, I]
    dist.barrier()
    r0(f"  done {time.perf_counter()-t0:.1f}s  w1={tuple(w1.shape)} w2={tuple(w2.shape)}")

    # ── Random attention weights ───────────────────────────────────────────────
    H_per = HIDDEN // ws  # 1024 for TP/CP modes
    KV_DIM = NUM_KV_HEADS * HEAD_DIM

    if args.mode == "4tp":
        Wq = torch.randn(HIDDEN, H_per,      dtype=torch.bfloat16, device=dev) * 0.02
        Wk = torch.randn(HIDDEN, KV_DIM,     dtype=torch.bfloat16, device=dev) * 0.02
        Wv = torch.randn(HIDDEN, KV_DIM,     dtype=torch.bfloat16, device=dev) * 0.02
        Wo = torch.randn(H_per,  HIDDEN,     dtype=torch.bfloat16, device=dev) * 0.02
    elif args.mode == "4dp":
        Wq = torch.randn(HIDDEN, HIDDEN,     dtype=torch.bfloat16, device=dev) * 0.02
        Wk = torch.randn(HIDDEN, KV_DIM,     dtype=torch.bfloat16, device=dev) * 0.02
        Wv = torch.randn(HIDDEN, KV_DIM,     dtype=torch.bfloat16, device=dev) * 0.02
        Wo = torch.randn(HIDDEN, HIDDEN,     dtype=torch.bfloat16, device=dev) * 0.02
    else:  # 4cp
        Wq = torch.randn(HIDDEN, H_per,      dtype=torch.bfloat16, device=dev) * 0.02
        Wk = torch.randn(HIDDEN, KV_DIM,     dtype=torch.bfloat16, device=dev) * 0.02
        Wv = torch.randn(HIDDEN, KV_DIM,     dtype=torch.bfloat16, device=dev) * 0.02
        Wo = torch.randn(H_per,  HIDDEN,     dtype=torch.bfloat16, device=dev) * 0.02

    Wg = torch.randn(HIDDEN, NUM_EXPERTS, dtype=torch.bfloat16, device=dev) * 0.02
    Wn = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)

    # ── Input tensors ─────────────────────────────────────────────────────────
    T_local = SEQ_LEN if args.mode == "4tp" else SEQ_LEN // ws

    def make_input():
        return torch.randn(T_local, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.1

    # ── Layer-dispatch helper ──────────────────────────────────────────────────
    def run_layer(hs):
        if args.mode == "4tp":
            return forward_4tp(hs, Wq, Wk, Wv, Wo, Wg, Wn, w1, w2, ws, rank, dev)
        elif args.mode == "4dp":
            return forward_4dp(hs, Wq, Wk, Wv, Wo, Wg, Wn, w1, w2, ws, rank, dev)
        else:
            return forward_4cp(hs, Wq, Wk, Wv, Wo, Wg, Wn, w1, w2, ws, rank, dev)

    # ── Full forward pass (60 layers) ─────────────────────────────────────────
    def forward_pass():
        hs = make_input()
        all_evs = []
        for _ in range(NUM_LAYERS):
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

        def avg(lst): return sum(lst)/len(lst)
        def sm(lst): return sum(lst)

        mode_desc = {
            "4tp": f"TP=4, EP=4  (T={SEQ_LEN}, {HIDDEN//ws} heads/GPU, MoE EP all-to-all from {SEQ_LEN} tokens)",
            "4dp": f"DP=4, EP=4  (T_local={T_local}, {NUM_HEADS} heads/GPU, MoE EP all-to-all from {T_local} tokens)",
            "4cp": f"CP=4+TP=4, EP=4  (T_local={T_local}, {HIDDEN//ws} heads/GPU, ring-attn {ws} steps, MoE EP from {T_local} tokens)",
        }

        print("\n" + "=" * 90)
        print(f"Qwen3.5-397B-A17B  ·  Baseline (NO mega_moe)  ·  {args.mode.upper()}")
        print(f"  Config: {mode_desc[args.mode]}")
        print(f"  Weights: real FP8 layer-1 → BF16 (128 experts/rank × 4 ranks)")
        print(f"  Attn: random weights, same GEMM shapes as real model")
        print("=" * 90)

        print(f"\n  {'Layer':>5}  {'QKV':>7}  {'SDPA':>8}  {'Oproj':>7}  "
              f"{'Gate':>6}  {'Dispatch':>9}  {'ExpertGEMM':>11}  {'Combine':>8}  "
              f"{'MoE':>7}  {'Total':>8}  ms")
        print("  " + "-" * 88)
        for l in range(NUM_LAYERS):
            if l == 0 or l == NUM_LAYERS-1 or (l+1) % 10 == 0:
                print(f"  {l:>5}  {t_qkv[l]:>7.2f}  {t_sdpa[l]:>8.2f}  {t_oproj[l]:>7.2f}  "
                      f"{t_gate[l]:>6.2f}  {t_disp[l]:>9.2f}  {t_gemm[l]:>11.2f}  "
                      f"{t_comb[l]:>8.2f}  {t_moe[l]:>7.2f}  {t_total[l]:>8.2f}")
            elif l == 1:
                print("  " + " " * 5 + "   ...")

        print("  " + "-" * 88)
        print(f"  {'AVG':>5}  {avg(t_qkv):>7.2f}  {avg(t_sdpa):>8.2f}  {avg(t_oproj):>7.2f}  "
              f"{avg(t_gate):>6.2f}  {avg(t_disp):>9.2f}  {avg(t_gemm):>11.2f}  "
              f"{avg(t_comb):>8.2f}  {avg(t_moe):>7.2f}  {avg(t_total):>8.2f}  (per layer)")
        print(f"  {'SUM':>5}  {sm(t_qkv):>7.1f}  {sm(t_sdpa):>8.1f}  {sm(t_oproj):>7.1f}  "
              f"{sm(t_gate):>6.1f}  {sm(t_disp):>9.1f}  {sm(t_gemm):>11.1f}  "
              f"{sm(t_comb):>8.1f}  {sm(t_moe):>7.1f}  {sm(t_total):>8.1f}  (60-layer total)")

        print(f"\n  Wall-clock (60-layer forward pass): {t_wall:.1f} ms")

        attn_total = sm(t_qkv) + sm(t_sdpa) + sm(t_oproj)
        moe_total  = sm(t_gate) + sm(t_moe)
        disp_total = sm(t_disp); gemm_total = sm(t_gemm); comb_total = sm(t_comb)
        print(f"\n  Compute breakdown (60-layer totals):")
        print(f"    Attention (QKV+SDPA+Oproj):  {attn_total:>8.1f} ms  ({100*attn_total/sm(t_total):.1f}%)")
        print(f"    MoE total (Gate+Disp+GEMM+Combine): {moe_total:>6.1f} ms  ({100*moe_total/sm(t_total):.1f}%)")
        print(f"      ├─ EP Dispatch:    {disp_total:>6.1f} ms  ({100*disp_total/sm(t_total):.1f}%)")
        print(f"      ├─ Expert GEMMs:   {gemm_total:>6.1f} ms  ({100*gemm_total/sm(t_total):.1f}%)")
        print(f"      └─ EP Combine:     {comb_total:>6.1f} ms  ({100*comb_total/sm(t_total):.1f}%)")

        # Cumulative timeline
        print(f"\n  Cumulative timeline (every 10 layers):")
        cumul = 0.0
        for l in range(NUM_LAYERS):
            cumul += t_total[l]
            if l == 0 or (l+1) % 10 == 0:
                print(f"    Layer {l:>2}: +{t_total[l]:.2f} ms  →  cumulative {cumul:.1f} ms")

        print("=" * 90)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
