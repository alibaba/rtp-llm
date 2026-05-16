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

import os

import torch
import triton
import triton.language as tl

INDEXER_HEAD_DIM = 128
FP8_E4M3_MAX = 448.0
DEFAULT_GROUP_HEADS = 8


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


@triton.jit
def _indexer_q_fp8_fold_group_heads_kernel(
    q_ptr,  # [M, H, D] bf16
    w_ptr,  # [M, H]    bf16/fp32
    q_fp8_ptr,  # [M, H, D] fp8e4nv (uint8 view)
    w_fold_ptr,  # [M, H]    fp32
    H: tl.constexpr,
    D: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_HEADS: tl.constexpr,
):
    """One program handles several contiguous indexer heads for one token."""
    pid_m = tl.program_id(0).to(tl.int64)
    head_tile = tl.program_id(1).to(tl.int64)

    h_off = head_tile * GROUP_HEADS + tl.arange(0, GROUP_HEADS)
    d_off = tl.arange(0, D)
    head_mask = h_off < H
    q_offsets = (pid_m * H + h_off[:, None]) * D + d_off[None, :]

    q = tl.load(q_ptr + q_offsets, mask=head_mask[:, None], other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(q), axis=1)
    scale = tl.maximum(absmax / fp8_max, 1e-12)

    q_fp8 = (q / scale[:, None]).to(tl.float8e4nv)
    tl.store(
        (q_fp8_ptr + q_offsets).to(tl.pointer_type(tl.uint8)),
        q_fp8.to(tl.uint8, bitcast=True),
        mask=head_mask[:, None],
    )

    w_offsets = pid_m * H + h_off
    w = tl.load(w_ptr + w_offsets, mask=head_mask, other=0.0).to(tl.float32)
    tl.store(w_fold_ptr + w_offsets, w * scale, mask=head_mask)


def _selected_group_heads(group_heads: int | None) -> int:
    if group_heads is None:
        raw = os.environ.get("DSV4_INDEXER_Q_FP8_GROUP_HEADS")
        group_heads = int(raw) if raw else DEFAULT_GROUP_HEADS
    if group_heads not in (1, 2, 4, 8):
        raise ValueError(
            f"invalid DSV4_INDEXER_Q_FP8_GROUP_HEADS={group_heads}; "
            "expected one of 1, 2, 4, 8"
        )
    return group_heads


def indexer_q_fp8_quant_fold(
    q_bf16: torch.Tensor,  # [B, S, H, D] bf16, D=128
    weights: torch.Tensor,  # [B, S, H]    bf16 or fp32
    *,
    group_heads: int | None = None,
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

    selected_group_heads = _selected_group_heads(group_heads)
    if selected_group_heads > 1 and D == INDEXER_HEAD_DIM:
        grid = (B * S, triton.cdiv(H, selected_group_heads))
        _indexer_q_fp8_fold_group_heads_kernel[grid](
            q_bf16,
            weights,
            q_fp8,
            w_fold,
            H=H,
            D=D,
            fp8_max=FP8_E4M3_MAX,
            GROUP_HEADS=selected_group_heads,
            num_warps=4,
        )
    else:
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
