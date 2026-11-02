"""V4 Indexer score Triton kernel — fused einsum + ReLU + weighted-sum.

Replaces the chunked materialization in ``indexer.py`` (final_plan.md P0):

    index_score = einsum("bshd,btd->bsht", q_f, kv_f)        # [B,S,H,T]
    index_score = (index_score.relu_() * w_f.unsqueeze(-1)).sum(dim=2)  # [B,S,T]

with a single Triton kernel that streams the head dimension and never
materializes ``[B,S,H,T]``.  Output ``[B,S,T] fp32``, identical math.

Memory cost drops from ``S*H*T*4`` bytes (the ``relu`` intermediate) to
zero — ``index_score`` itself is the only live tensor.

Optional causal mask for prefill (``apply_causal_mask=True``):
    out[b, s, t] += -inf  if  t >= (q_pos[s] + 1) // compress_ratio

Folds the mask into the kernel so we don't allocate a separate ``[B,S,T]``
mask tensor for the long-context case.

Caller passes:
  q       [B, S, H, D] bf16
  kv      [B, T, D]    bf16
  weights [B, S, H]    bf16/fp32 (cast inside kernel)
  q_pos   [B, S] int32 — global Q position for each row, OR None when not
           applying causal mask (decode / start_pos > 0)
  compress_ratio: int — divisor for the causal kv-column threshold

Returns:
  out     [B, S, T] fp32

Launch configurations:
  H32/D128 prefill uses a dynamic head loop and S32/T128 tiles; other
  shapes retain the V4 S16/T256 configuration. Both tile S × T without
  materializing the head dimension.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "q_b",
        "kv_b",
        "w_b",
        "qpos_b",
        "out_b",
        "out_s",
        "S",
        "T",
    ]
)
def _v4_indexer_score_fwd(
    q_ptr,  # [B, S, H, D] bf16
    kv_ptr,  # [B, T, D]    bf16
    w_ptr,  # [B, S, H]    fp32 (caller casts)
    q_pos_ptr,  # [B, S] int32, or 0 if APPLY_MASK==False
    k_pos_ptr,
    q_request_ptr,
    k_request_ptr,
    out_ptr,  # [B, S, T] fp32
    # strides
    q_b,
    q_s: tl.constexpr,
    q_h: tl.constexpr,  # q strides; d-stride==1 assumed (contiguous)
    kv_b,
    kv_t: tl.constexpr,  # kv strides; d-stride==1 assumed
    w_b,
    w_s: tl.constexpr,  # w strides; h-stride==1 assumed
    qpos_b,  # q_pos stride for B; s-stride==1
    kpos_b,
    qrequest_b,
    krequest_b,
    out_b,
    out_s,  # out strides; t-stride==1 assumed
    # geometry
    S,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    # tile sizes
    BLOCK_S: tl.constexpr,
    BLOCK_T: tl.constexpr,
    # toggles
    APPLY_MASK: tl.constexpr,
    GLOBAL_KEY_POSITIONS: tl.constexpr,
    REQUEST_MASK: tl.constexpr,
    DYNAMIC_HEAD_LOOP: tl.constexpr = False,
):
    """One program covers one (b, s_block, t_block) tile.

    Inner loop streams over H heads, computing
      score[s, t] = sum_h relu(sum_d q[s,h,d] * k[t,d]) * w[s,h]
    using fp32 accumulators. H32 prefill keeps the head loop dynamic to
    avoid spilling the fully unrolled sequence of dot/reduce operations.
    """
    pid_b = tl.program_id(0).to(tl.int64)
    pid_s = tl.program_id(1).to(tl.int64)
    pid_t = tl.program_id(2).to(tl.int64)

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    s_mask = s_off < S
    t_mask = t_off < T

    # K tile [BLOCK_T, D] — shared across all S rows in this program.
    # Kept in BF16 so the inner tl.dot uses BF16 tensor cores with an
    # fp32 accumulator (out_dtype below).  Casting to fp32 here would
    # force the dot through FP32 SIMT / TF32 and lose ~30× throughput.
    d_idx = tl.arange(0, D)
    k_ptrs = kv_ptr + pid_b * kv_b + t_off[:, None] * kv_t + d_idx[None, :]
    k_tile = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0)  # [BLOCK_T, D] bf16
    # Pre-transpose K once.  Inside the H loop, we'd otherwise tl.trans
    # the same tile 64 times — the compiler can sometimes hoist that
    # CSE, but doing it explicitly keeps the dot's RHS as a loop-
    # invariant register tile and saves issue slots.
    k_t = tl.trans(k_tile)  # [D, BLOCK_T] bf16

    # Causal mask (apply once per program; per-tile).  q_pos[s] is the global
    # position of S row s; the row may attend to compressed-KV columns
    # [0, (q_pos+1) // ratio).
    if APPLY_MASK:
        qp_ptrs = q_pos_ptr + pid_b * qpos_b + s_off
        q_pos = tl.load(qp_ptrs, mask=s_mask, other=0).to(tl.int32)  # [BLOCK_S]
        # threshold[s] = (q_pos + 1) // ratio
        thr = (q_pos + 1) // COMPRESS_RATIO  # [BLOCK_S]
        # mask[s, t] = (t < thr[s])
        # broadcast t_off [BLOCK_T] vs thr [BLOCK_S]
        if GLOBAL_KEY_POSITIONS:
            key_pos = tl.load(k_pos_ptr + pid_b * kpos_b + t_off, mask=t_mask, other=0)
        else:
            key_pos = t_off
        causal = key_pos[None, :] < thr[:, None]  # [BLOCK_S, BLOCK_T]
    else:
        causal = tl.full((BLOCK_S, BLOCK_T), True, dtype=tl.int1)
    if REQUEST_MASK:
        q_request = tl.load(
            q_request_ptr + pid_b * qrequest_b + s_off, mask=s_mask, other=-1
        )
        k_request = tl.load(
            k_request_ptr + pid_b * krequest_b + t_off, mask=t_mask, other=-2
        )
        causal = causal & (q_request[:, None] == k_request[None, :])

    acc = tl.zeros((BLOCK_S, BLOCK_T), dtype=tl.float32)

    if DYNAMIC_HEAD_LOOP:
        # Use the actual positions/request IDs for interleaved CP rows/keys.
        # Padded lanes can only keep an empty tile active; all accesses and
        # output stores remain tail-masked, so they cannot hide a live pair.
        if tl.sum(causal.to(tl.int32)) > 0:
            for h in range(H):
                q_ptrs = (
                    q_ptr
                    + pid_b * q_b
                    + s_off[:, None] * q_s
                    + h * q_h
                    + d_idx[None, :]
                )
                q_tile = tl.load(q_ptrs, mask=s_mask[:, None], other=0.0)
                score = tl.dot(q_tile, k_t, out_dtype=tl.float32)
                score = tl.where(score > 0.0, score, 0.0)
                w_ptrs = w_ptr + pid_b * w_b + s_off * w_s + h
                w = tl.load(w_ptrs, mask=s_mask, other=0.0).to(tl.float32)
                acc += score * w[:, None]
    else:
        # Preserve the established V4/decode configuration for other shapes.
        for h in tl.static_range(H):
            q_ptrs = (
                q_ptr + pid_b * q_b + s_off[:, None] * q_s + h * q_h + d_idx[None, :]
            )
            q_tile = tl.load(q_ptrs, mask=s_mask[:, None], other=0.0)
            score = tl.dot(q_tile, k_t, out_dtype=tl.float32)
            score = tl.where(score > 0.0, score, 0.0)
            w_ptrs = w_ptr + pid_b * w_b + s_off * w_s + h
            w = tl.load(w_ptrs, mask=s_mask, other=0.0).to(tl.float32)
            acc += score * w[:, None]

    # Write masked output: invalid s/t rows stay as 0 (won't be read by
    # topk).  Causal-masked entries get -inf so the downstream topk
    # naturally drops them.
    final = tl.where(causal, acc, float("-inf"))
    out_ptrs = out_ptr + pid_b * out_b + s_off[:, None] * out_s + t_off[None, :]
    write_mask = s_mask[:, None] & t_mask[None, :]
    tl.store(out_ptrs, final, mask=write_mask)


def v4_indexer_score(
    q: torch.Tensor,  # [B, S, H, D] bf16, contiguous in last dim
    kv: torch.Tensor,  # [B, T, D]    bf16
    weights: torch.Tensor,  # [B, S, H]    bf16 or fp32
    q_pos: Optional[
        torch.Tensor
    ] = None,  # [B, S] int32 — global Q position; None for no mask
    compress_ratio: int = 1,
    *,
    kv_positions: Optional[torch.Tensor] = None,
    query_requests: Optional[torch.Tensor] = None,
    key_requests: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """V4 indexer fused score.  See module docstring for the kernel math.

    When ``q_pos`` is None (decode / continuation), the causal mask is
    skipped — the caller is responsible for any masking afterwards.
    """
    assert q.dtype == torch.bfloat16, f"q dtype={q.dtype}"
    assert kv.dtype == torch.bfloat16, f"kv dtype={kv.dtype}"
    assert q.is_contiguous(), "q must be contiguous"
    assert kv.is_contiguous(), "kv must be contiguous"
    assert q.dim() == 4 and kv.dim() == 3
    B, S, H, D = q.shape
    Bk, T, Dk = kv.shape
    assert B == Bk and D == Dk, f"q/kv batch/D mismatch: q={q.shape} kv={kv.shape}"
    assert weights.shape == (
        B,
        S,
        H,
    ), f"weights shape={weights.shape} expected {(B,S,H)}"

    weights = weights.contiguous()
    if weights.dtype != torch.float32:
        weights = weights.float()

    apply_mask = q_pos is not None
    if apply_mask:
        q_pos = q_pos.contiguous()
        assert q_pos.shape == (B, S), f"q_pos shape={q_pos.shape} expected {(B,S)}"
        assert q_pos.dtype in (torch.int32, torch.int64), f"q_pos dtype={q_pos.dtype}"
        if q_pos.dtype != torch.int32:
            q_pos = q_pos.to(torch.int32)
    else:
        # passing a tiny dummy avoids a Triton constexpr branch on a real ptr
        q_pos = torch.empty(1, dtype=torch.int32, device=q.device)

    if kv_positions is not None:
        if (
            not apply_mask
            or kv_positions.shape != (B, T)
            or kv_positions.dtype != torch.int32
        ):
            raise ValueError("global Indexer key positions require causal Q positions")
        kv_positions = kv_positions.contiguous()
    request_mask = query_requests is not None
    if request_mask != (key_requests is not None):
        raise ValueError("query/key request masks must be supplied together")
    if request_mask:
        if (
            query_requests.shape != (B, S)
            or key_requests.shape != (B, T)
            or query_requests.dtype != torch.int32
            or key_requests.dtype != torch.int32
        ):
            raise ValueError("Indexer request IDs must match complete Q/K rows")
        query_requests, key_requests = (
            query_requests.contiguous(),
            key_requests.contiguous(),
        )
    if out is None:
        out = torch.empty((B, S, T), dtype=torch.float32, device=q.device)
    elif (
        out.shape != (B, S, T)
        or out.dtype != torch.float32
        or out.device != q.device
        or out.stride(-1) != 1
    ):
        raise ValueError("Indexer score output buffer does not match the score matrix")
    if S == 0 or T == 0:
        return out

    # SM103 H32 prefill avoids register spills from full head unrolling.
    # Query-CP cache hits can have fewer than 32 local rows. Their full
    # head unrolling spills to the CUDA thread stack even for tiny Q.
    # Reuse the existing loop for these masked Prefill rows; Decode keeps
    # its established dispatch. Complete Q/K inputs remain unchanged.
    dynamic_heads = (
        apply_mask
        and H == 32
        and D == 128
        and T >= 128
        and (S >= 32 or (request_mask and kv_positions is not None))
    )
    if dynamic_heads:
        BLOCK_S, BLOCK_T = 32, 128
    else:
        # Existing V4 H64 configuration, also used for decode and tiny KV.
        # Keep MMA dimensions >=16 and mask short/tail tiles.
        BLOCK_S = 16
        BLOCK_T = 256 if T >= 256 else max(16, triton.next_power_of_2(T))

    grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(T, BLOCK_T))

    _v4_indexer_score_fwd[grid](
        q,
        kv,
        weights,
        q_pos,
        kv_positions if kv_positions is not None else q_pos,
        query_requests if request_mask else q_pos,
        key_requests if request_mask else q_pos,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        weights.stride(0),
        weights.stride(1),
        q_pos.stride(0) if apply_mask else 0,
        kv_positions.stride(0) if kv_positions is not None else 0,
        query_requests.stride(0) if request_mask else 0,
        key_requests.stride(0) if request_mask else 0,
        out.stride(0),
        out.stride(1),
        S=S,
        T=T,
        H=H,
        D=D,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_S=BLOCK_S,
        BLOCK_T=BLOCK_T,
        APPLY_MASK=apply_mask,
        GLOBAL_KEY_POSITIONS=kv_positions is not None,
        REQUEST_MASK=request_mask,
        DYNAMIC_HEAD_LOOP=dynamic_heads,
        num_warps=4,
        num_stages=2,
    )
    return out
