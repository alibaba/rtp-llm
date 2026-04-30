# -*- coding: utf-8 -*-
# Context-parallel (CP) helpers shared across variants.
#
# This module collects the pure-Python functions used by the CP gated-delta-rule
# implementations: conv1d ctx exchange (NCCL all_gather + per-batch slicing),
# zigzag <-> causal layout maps, and segment cu_seqlens construction. None of
# these contain Triton kernels — kernels live in the variant-specific files.

from typing import Optional, Tuple

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Conv1d context exchange (zigzag variant)
# ---------------------------------------------------------------------------


def exchange_conv_context(
    local_qkv: torch.Tensor,
    cu_seqlens_cpu: list,
    half_lengths_cpu: list,
    rank: int,
    cp_size: int,
    cp_group: dist.ProcessGroup,
    ctx_len: int = 3,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Exchange conv1d context tokens between ranks via all_gather.

    Each rank takes the last `ctx_len` tokens of its seg0 and seg1 (for every
    sequence in the batch), all-gathers them, and returns the neighbouring
    rank's tail as the ctx that needs to be prepended before our own segments.

    Returns `(seg0_ctx, seg1_ctx)` channel-first tensors of shape
    `[dim, ctx_len * batch]`. `seg0_ctx` is `None` on rank 0 (rank 0 reads its
    seg0 conv prefix from the block cache via `prefix_lengths` instead).
    """
    assert cp_size == 2, "Only cp_size=2 supported for now"
    batch_size = len(half_lengths_cpu)
    dim = local_qkv.shape[0]
    device = local_qkv.device

    seg0_tail = torch.empty(
        dim, ctx_len * batch_size, device=device, dtype=local_qkv.dtype
    )
    seg1_tail = torch.empty(
        dim, ctx_len * batch_size, device=device, dtype=local_qkv.dtype
    )
    for b in range(batch_size):
        s = cu_seqlens_cpu[b]
        h = half_lengths_cpu[b]
        seg0_tail[:, b * ctx_len : (b + 1) * ctx_len] = local_qkv[
            :, s + h - ctx_len : s + h
        ]
        seg1_tail[:, b * ctx_len : (b + 1) * ctx_len] = local_qkv[
            :, s + 2 * h - ctx_len : s + 2 * h
        ]

    send_buf = seg0_tail if rank == 0 else seg1_tail
    all_tails = [torch.empty_like(send_buf) for _ in range(cp_size)]
    dist.all_gather(all_tails, send_buf, group=cp_group)

    return (None, all_tails[1]) if rank == 0 else (all_tails[0], seg0_tail)


def prepend_conv_context(
    local_qkv: torch.Tensor,
    cu_seqlens_cpu: list,
    half_lengths_cpu: list,
    seg0_ctx: Optional[torch.Tensor],
    seg1_ctx: torch.Tensor,
    ctx_len: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Prepend ctx tokens before each segment so a single `causal_conv1d_fn`
    call gives the same output as a global conv1d on the full sequence.

    Layout per sequence (inside `padded`):
        [seg0_ctx?] [seg0_data] [seg1_ctx] [seg1_data]

    `seg0_ctx` is omitted on rank 0 (the kernel reads that prefix from the
    block cache instead). Returns `(padded_qkv, padded_seg_cu, padded_seg_cu_cpu)`.
    """
    has_seg0_ctx = seg0_ctx is not None
    batch_size = len(half_lengths_cpu)
    dim = local_qkv.shape[0]
    device = local_qkv.device

    extra_per_seq = ctx_len * (1 + int(has_seg0_ctx))
    new_total = local_qkv.shape[1] + extra_per_seq * batch_size

    padded = torch.zeros(new_total, dim, device=device, dtype=local_qkv.dtype).T
    padded_seg_cu_cpu = [0]

    offset_src = 0
    offset_dst = 0
    for b in range(batch_size):
        half = half_lengths_cpu[b]

        if has_seg0_ctx:
            padded[:, offset_dst : offset_dst + ctx_len] = seg0_ctx[
                :, b * ctx_len : (b + 1) * ctx_len
            ]
            offset_dst += ctx_len
        padded[:, offset_dst : offset_dst + half] = local_qkv[
            :, offset_src : offset_src + half
        ]
        seg0_len = half + (ctx_len if has_seg0_ctx else 0)
        padded_seg_cu_cpu.append(padded_seg_cu_cpu[-1] + seg0_len)
        offset_dst += half
        offset_src += half

        padded[:, offset_dst : offset_dst + ctx_len] = seg1_ctx[
            :, b * ctx_len : (b + 1) * ctx_len
        ]
        offset_dst += ctx_len
        padded[:, offset_dst : offset_dst + half] = local_qkv[
            :, offset_src : offset_src + half
        ]
        seg1_len = half + ctx_len
        padded_seg_cu_cpu.append(padded_seg_cu_cpu[-1] + seg1_len)
        offset_dst += half
        offset_src += half

    padded_seg_cu = torch.tensor(padded_seg_cu_cpu, dtype=torch.long, device=device)
    return padded, padded_seg_cu, padded_seg_cu_cpu


def strip_conv_context(
    conv_output: torch.Tensor,
    cu_seqlens_cpu: list,
    half_lengths_cpu: list,
    padded_seg_cu_cpu: list,
    seg0_has_ctx: bool,
    local_total: int,
    ctx_len: int = 3,
) -> torch.Tensor:
    """Strip the prepended ctx tokens out of `conv_output`.

    Returns `[dim, local_total]` containing only the data positions —
    inverse of `prepend_conv_context`'s layout.
    """
    batch_size = len(half_lengths_cpu)
    dim = conv_output.shape[0]
    device = conv_output.device

    stripped = torch.empty(local_total, dim, device=device, dtype=conv_output.dtype).T
    offset_dst = 0
    for b in range(batch_size):
        half = half_lengths_cpu[b]

        seg0_start = padded_seg_cu_cpu[2 * b] + (ctx_len if seg0_has_ctx else 0)
        stripped[:, offset_dst : offset_dst + half] = conv_output[
            :, seg0_start : seg0_start + half
        ]
        offset_dst += half

        seg1_start = padded_seg_cu_cpu[2 * b + 1] + ctx_len
        stripped[:, offset_dst : offset_dst + half] = conv_output[
            :, seg1_start : seg1_start + half
        ]
        offset_dst += half

    return stripped


# ---------------------------------------------------------------------------
# Zigzag <-> causal layout maps
# ---------------------------------------------------------------------------


def zigzag_causal_order(cp_size: int) -> list:
    """Map all-gather layout to causal order.

    All-gather layout: [rank0_seg0, rank0_seg1, rank1_seg0, rank1_seg1, ...]
    Causal order (zigzag): rank0_seg0, rank1_seg0, ..., rankN_seg0,
                           rankN_seg1, ..., rank1_seg1, rank0_seg1

    Returns indices into the all-gather layout that produce causal order.
    """
    num_segs = 2 * cp_size
    order = []
    for pos in range(num_segs):
        if pos < cp_size:
            rank = pos
            seg = 0
        else:
            rank = num_segs - 1 - pos
            seg = 1
        order.append(rank * 2 + seg)
    return order


def causal_positions(rank: int, cp_size: int) -> Tuple[int, int]:
    """Return the causal chain positions of this rank's seg0 and seg1."""
    seg0_pos = rank
    seg1_pos = 2 * cp_size - 1 - rank
    return seg0_pos, seg1_pos


# ---------------------------------------------------------------------------
# Segment cu_seqlens
# ---------------------------------------------------------------------------


def build_segment_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Build cu_seqlens that treats each sequence's two halves as separate sequences.

    Input cu_seqlens: [0, L0, L0+L1, ...]  (batch+1 entries)
    Output: [0, L0/2, L0, L0+L1/2, L0+L1, ...]  (2*batch+1 entries)

    This lets all kernels process seg0 and seg1 in a single pass.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    half_lengths = lengths // 2
    batch_size = lengths.shape[0]
    seg_cu = torch.zeros(
        2 * batch_size + 1, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    for b in range(batch_size):
        seg_cu[2 * b + 1] = seg_cu[2 * b] + half_lengths[b]
        seg_cu[2 * b + 2] = seg_cu[2 * b + 1] + half_lengths[b]
    return seg_cu
