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
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

INDEXER_HEAD_DIM = 128
FP8_E4M3_MAX = 448.0


@triton.jit
def _indexer_q_fp8_fold_kernel(
    q_ptr,  # [B*S, H, D] bf16
    w_ptr,  # [B*S, H]    bf16/fp32
    q_fp8_ptr,  # [B*S, H, D] fp8e4nv (uint8 view)
    w_fold_ptr,  # [B*S, H]    fp32
    # geometry
    BSH: tl.constexpr,  # B*S*H — rows
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
