"""DSv4 Indexer Q per-(token, head) FP8 quant + weight fold.

Companion to :func:`deep_gemm.fp8_paged_mqa_logits` — that kernel takes
``q`` already in FP8 e4m3fn (no per-token Q scale tensor; the per-token
scale must be folded into ``weights`` upstream).

Math:

    For each (b, s, h):
        absmax_h = max_d(|q_bf16[b,s,h,d]|)
        scale_h  = absmax_h / 448.0           (clamped to avoid /0)
        q_fp8_h  = round_e4m3(q_bf16 / scale_h)
        w_fold_h = w_bf16[b,s,h] * scale_h

Then DeepGEMM does ``score = q_fp8 @ k_dequant`` and the caller multiplies
by ``w_fold`` head-wise to get the correct ``ReLU(real_q · k) * w``
weighted sum (positive scale commutes with ReLU).

Public API:

  * :func:`indexer_q_fp8_quant_fold` — bf16 q [B,S,H,D] + bf16 weights [B,S,H]
    → (fp8 q [B,S,H,D], folded weights [B,S,H] fp32). One kernel pass.
  * :func:`indexer_q_rope_fp8_quant_fold` — same output contract, but applies
    RoPE to the q tail in registers before quantization. This replaces
    ``rope_only_inplace(q[..., -rope_dim:])`` followed by
    :func:`indexer_q_fp8_quant_fold` on the hot indexer path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

INDEXER_HEAD_DIM = 128
FP8_E4M3_MAX = 448.0


@triton.jit(do_not_specialize=["BSH"])
def _indexer_q_fp8_fold_kernel(
    q_ptr,  # [B*S, H, D] bf16
    w_ptr,  # [B*S, H]    bf16/fp32
    q_fp8_ptr,  # [B*S, H, D] fp8e4nv (uint8 view)
    w_fold_ptr,  # [B*S, H]    fp32
    # geometry
    BSH,  # B*S*H — rows
    D: tl.constexpr,  # head_dim = 128
    fp8_max: tl.constexpr,
):
    """One program per (b, s, h) row. Each row is one 128-elem vector;
    fits in registers, no inner-D tiling needed."""
    pid = tl.program_id(0).to(tl.int64)
    if pid >= BSH:
        return

    d_off = tl.arange(0, D)
    q = tl.load(q_ptr + pid * D + d_off).to(tl.float32)

    absmax = tl.max(tl.abs(q), axis=0)
    scale = tl.maximum(absmax / fp8_max, 1e-12)

    q_fp8 = (q / scale).to(tl.float8e4nv)
    tl.store(
        (q_fp8_ptr + pid * D + d_off).to(tl.pointer_type(tl.uint8)),
        q_fp8.to(tl.uint8, bitcast=True),
    )

    w = tl.load(w_ptr + pid).to(tl.float32)
    tl.store(w_fold_ptr + pid, w * scale)


def indexer_q_fp8_quant_fold(
    q_bf16: torch.Tensor,  # [B, S, H, D] bf16, D=128
    weights: torch.Tensor,  # [B, S, H]    bf16 or fp32
):
    """Per-(token, head) FP8 quant of indexer Q + scale-fold into weights.

    Returns ``(q_fp8, w_folded)`` where ``q_fp8`` is contiguous
    ``[B, S, H, D] float8_e4m3fn`` and ``w_folded`` is contiguous
    ``[B, S, H] float32``.
    """
    assert q_bf16.dim() == 4 and q_bf16.shape[-1] == INDEXER_HEAD_DIM
    assert q_bf16.dtype == torch.bfloat16
    assert q_bf16.is_contiguous()
    assert weights.shape == q_bf16.shape[:3]
    if weights.dtype != torch.bfloat16 and weights.dtype != torch.float32:
        weights = weights.float()
    weights = weights.contiguous()

    B, S, H, D = q_bf16.shape
    BSH = B * S * H
    q_fp8 = torch.empty(B, S, H, D, dtype=torch.float8_e4m3fn, device=q_bf16.device)
    w_fold = torch.empty(B, S, H, dtype=torch.float32, device=q_bf16.device)
    if BSH == 0:
        return q_fp8, w_fold

    _indexer_q_fp8_fold_kernel[(BSH,)](
        q_bf16,
        weights,
        q_fp8,
        w_fold,
        BSH=BSH,
        D=D,
        fp8_max=FP8_E4M3_MAX,
        num_warps=4,
    )
    return q_fp8, w_fold


@triton.jit(do_not_specialize=["BSH"])
def _indexer_q_rope_fp8_fold_kernel(
    q_ptr,  # [B*S, H, D] bf16, pre-RoPE
    w_ptr,  # [B*S, H]    bf16/fp32
    freqs_ri_ptr,  # [N_freq, RD/2, 2] float32 view_as_real(complex freqs)
    q_fp8_ptr,  # [B*S, H, D] fp8e4nv (uint8 view)
    w_fold_ptr,  # [B*S, H]    fp32
    BSH,  # B*S*H — rows
    freq_stride_n: tl.constexpr,
    freqs_stride_b,
    freqs_stride_k,
    D: tl.constexpr,
    RD: tl.constexpr,
    RD_HALF: tl.constexpr,
    ROPE_START: tl.constexpr,
    fp8_max: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= BSH:
        return

    d_off = tl.arange(0, D)
    q = tl.load(q_ptr + pid * D + d_off).to(tl.float32)

    is_rope = d_off >= ROPE_START
    rope_local = d_off - ROPE_START
    partner_off = ROPE_START + (rope_local ^ 1)
    partner = tl.load(q_ptr + pid * D + partner_off, mask=is_rope, other=0.0).to(
        tl.float32
    )

    pair_idx = tl.maximum(rope_local >> 1, 0)
    freq_idx = pid // freq_stride_n
    freq_base = freqs_ri_ptr + freq_idx * freqs_stride_b + pair_idx * freqs_stride_k
    cos = tl.load(freq_base, mask=is_rope, other=1.0)
    sin = tl.load(freq_base + 1, mask=is_rope, other=0.0)

    is_even = (rope_local & 1) == 0
    q_rot_even = q * cos - partner * sin
    q_rot_odd = q * cos + partner * sin
    q_rot = tl.where(is_even, q_rot_even, q_rot_odd)
    q = tl.where(is_rope, q_rot, q)
    # Old path stores the RoPE result back to a BF16 q tensor, then the
    # quant-fold kernel reloads it.  Preserve that boundary rounding so the
    # fused path keeps the same FP8/scale contract as closely as possible.
    q = q.to(tl.bfloat16).to(tl.float32)

    absmax = tl.max(tl.abs(q), axis=0)
    scale = tl.maximum(absmax / fp8_max, 1e-12)

    q_fp8 = (q / scale).to(tl.float8e4nv)
    tl.store(
        (q_fp8_ptr + pid * D + d_off).to(tl.pointer_type(tl.uint8)),
        q_fp8.to(tl.uint8, bitcast=True),
    )

    w = tl.load(w_ptr + pid).to(tl.float32)
    tl.store(w_fold_ptr + pid, w * scale)


def indexer_q_rope_fp8_quant_fold(
    q_bf16: torch.Tensor,  # [B, S, H, D] bf16, D=128, pre-RoPE
    weights: torch.Tensor,  # [B, S, H] bf16 or fp32
    freqs_cis: torch.Tensor,  # complex [N_freq, RD/2]
    rope_head_dim: int,
):
    """Fused RoPE + per-(token, head) FP8 quant + scale fold.

    Output contract is identical to :func:`indexer_q_fp8_quant_fold`.
    ``freqs_cis`` follows the same flattened-frequency mapping as
    ``rope_only_inplace``: each frequency row covers
    ``(B*S*H) // N_freq`` consecutive q rows.
    """
    assert q_bf16.dim() == 4 and q_bf16.shape[-1] == INDEXER_HEAD_DIM
    assert q_bf16.dtype == torch.bfloat16
    assert q_bf16.is_cuda
    assert weights.shape == q_bf16.shape[:3]
    assert rope_head_dim > 0 and rope_head_dim <= q_bf16.shape[-1]
    assert rope_head_dim % 2 == 0
    if not q_bf16.is_contiguous():
        q_bf16 = q_bf16.contiguous()
    if weights.dtype != torch.bfloat16 and weights.dtype != torch.float32:
        weights = weights.float()
    weights = weights.contiguous()
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()

    B, S, H, D = q_bf16.shape
    BSH = B * S * H
    q_fp8 = torch.empty(B, S, H, D, dtype=torch.float8_e4m3fn, device=q_bf16.device)
    w_fold = torch.empty(B, S, H, dtype=torch.float32, device=q_bf16.device)
    if BSH == 0:
        return q_fp8, w_fold

    freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
    n_freq = freqs_flat.shape[0]
    assert n_freq > 0
    assert BSH % n_freq == 0, f"N_rows={BSH} not divisible by N_freq={n_freq}"
    assert freqs_flat.shape[-1] == rope_head_dim // 2
    freqs_ri = torch.view_as_real(freqs_flat)
    freq_stride_n = BSH // n_freq

    _indexer_q_rope_fp8_fold_kernel[(BSH,)](
        q_bf16,
        weights,
        freqs_ri,
        q_fp8,
        w_fold,
        BSH=BSH,
        freq_stride_n=freq_stride_n,
        freqs_stride_b=freqs_ri.stride(0),
        freqs_stride_k=freqs_ri.stride(1),
        D=D,
        RD=rope_head_dim,
        RD_HALF=rope_head_dim // 2,
        ROPE_START=D - rope_head_dim,
        fp8_max=FP8_E4M3_MAX,
        num_warps=4,
    )
    return q_fp8, w_fold
