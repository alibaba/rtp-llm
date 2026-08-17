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

Shapes constraints (V4-Flash defaults):
  H = 64, D = 128 — both fit in registers per program; we tile S × T.
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
):
    """One program covers one (b, s_block, t_block) tile.

    Inner loop streams over H heads, computing
      score[s, t] = sum_h relu(sum_d q[s,h,d] * k[t,d]) * w[s,h]
    using fp32 accumulators.  D and H are kept as compile-time constants
    so each per-head dot is unrolled into a 1-shot tile load + reduce.
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
        causal = t_off[None, :] < thr[:, None]  # [BLOCK_S, BLOCK_T]
    else:
        causal = tl.full((BLOCK_S, BLOCK_T), True, dtype=tl.int1)

    acc = tl.zeros((BLOCK_S, BLOCK_T), dtype=tl.float32)

    for h in tl.static_range(H):
        # q[s, h, d] tile [BLOCK_S, D] — keep bf16 for tensor-core mma.
        q_ptrs = q_ptr + pid_b * q_b + s_off[:, None] * q_s + h * q_h + d_idx[None, :]
        q_tile = tl.load(q_ptrs, mask=s_mask[:, None], other=0.0)  # [BLOCK_S, D] bf16

        # dot: [BLOCK_S, D] x [D, BLOCK_T] = [BLOCK_S, BLOCK_T]
        # bf16 inputs + out_dtype=fp32 accumulator = BF16 tensor core path.
        # k_t was pre-transposed above the H loop.
        score = tl.dot(q_tile, k_t, out_dtype=tl.float32)

        # ReLU (V4 indexer math), then per-head weighted accumulate
        score = tl.where(score > 0.0, score, 0.0)
        w_ptrs = w_ptr + pid_b * w_b + s_off * w_s + h
        w = tl.load(w_ptrs, mask=s_mask, other=0.0).to(tl.float32)  # [BLOCK_S]
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

    out = torch.empty((B, S, T), dtype=torch.float32, device=q.device)
    if S == 0 or T == 0:
        return out

    # Tile sizes tuned on SM100 (GB200) for V4-Flash 64k+CP=4 (S=T=16384,
    # H=64, D=128).  Best config across the 4-of-4 shape sweep was
    # BLOCK_S=16 / BLOCK_T=256 / num_warps=4 / num_stages=2, with the
    # BF16 tensor-core mma path enabled in the kernel above.  Smaller
    # BLOCK_S beats the previous BLOCK_S=32 default because each program
    # streams Q across 64 heads — fewer rows × wider T columns gives
    # better Q reuse and more T-axis parallelism per SM.
    # Triton 3.4 rejects tl.dot tiles with M or N below 16.  Short prompts
    # can produce S/T < 16, so keep the MMA tile at the legal minimum and
    # rely on the masks above to discard padded rows/columns.
    BLOCK_S = 16
    BLOCK_T = 256 if T >= 256 else max(16, triton.next_power_of_2(T))

    grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(T, BLOCK_T))

    _v4_indexer_score_fwd[grid](
        q,
        kv,
        weights,
        q_pos,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        weights.stride(0),
        weights.stride(1),
        q_pos.stride(0) if apply_mask else 0,
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
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def _v4_fp8_indexer_score_fwd(
    q_ptr,  # [M, H, D] fp8 e4m3
    kv_ptr,  # [N, D] fp8 e4m3
    kv_scale_ptr,  # [N] fp32
    w_ptr,  # [M, H] fp32
    out_ptr,  # [M, N] fp32
    M,
    N,
    q_m: tl.constexpr,
    q_h: tl.constexpr,
    kv_n: tl.constexpr,
    w_m: tl.constexpr,
    out_m: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """SM120 FP8 indexer score without the [M,H,N] FP32 intermediate."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    m_mask = m_off < M
    n_mask = n_off < N
    d_off = tl.arange(0, D)

    k_ptrs = kv_ptr + n_off[:, None] * kv_n + d_off[None, :]
    k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
    k_t = tl.trans(k_tile)
    k_scale = tl.load(kv_scale_ptr + n_off, mask=n_mask, other=0.0).to(
        tl.float32
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for h in tl.static_range(H):
        q_ptrs = q_ptr + m_off[:, None] * q_m + h * q_h + d_off[None, :]
        q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        score = tl.dot(q_tile, k_t, out_dtype=tl.float32)
        score *= k_scale[None, :]
        score = tl.maximum(score, 0.0)
        weight = tl.load(w_ptr + m_off * w_m + h, mask=m_mask, other=0.0)
        acc += score * weight[:, None].to(tl.float32)

    out_ptrs = out_ptr + m_off[:, None] * out_m + n_off[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def v4_fp8_indexer_score(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Fused SM120 FP8 indexer score, returning ``[M, N]`` FP32 logits."""
    assert q.dtype == torch.float8_e4m3fn and q.dim() == 3
    assert kv.dtype == torch.float8_e4m3fn and kv.dim() == 2
    assert kv_scale.dtype == torch.float32 and kv_scale.dim() == 1
    assert weights.dtype == torch.float32 and weights.dim() == 2
    M, H, D = q.shape
    N, kv_d = kv.shape
    assert kv_d == D and kv_scale.shape[0] == N
    assert weights.shape == (M, H)
    assert q.is_contiguous() and kv.is_contiguous()
    weights = weights.contiguous()
    kv_scale = kv_scale.contiguous()
    out = torch.empty((M, N), dtype=torch.float32, device=q.device)
    if M == 0 or N == 0:
        return out
    block_m = 16
    block_n = 256 if N >= 256 else max(16, triton.next_power_of_2(N))
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _v4_fp8_indexer_score_fwd[grid](
        q,
        kv,
        kv_scale,
        weights,
        out,
        M,
        N,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        weights.stride(0),
        out.stride(0),
        H=H,
        D=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        # SM120 needs eight warps for the 16x256 FP8 tile. Four warps makes
        # this shape spill/serialize and is more than 40x slower at M=2048,
        # N=8192 on RTX PRO 5000.
        num_warps=8,
        num_stages=2,
    )
    return out
