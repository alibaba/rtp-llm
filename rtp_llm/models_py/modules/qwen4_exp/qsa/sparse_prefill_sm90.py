# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang's Apache-2.0 sparse GQA prefill implementation:
# https://github.com/sgl-project/sglang/blob/5007cbd6a57515bf5e008ff2fd2bdb834356b53c/python/sglang/srt/layers/attention/qsa/sparse_attn.py
"""Bounded SM90 BF16 sparse prefill using the SGLang attention algorithm.

The online exp2 softmax, BF16 probability product and launch configuration
follow the pinned upstream implementation. QK scaling stays in FP32 after the
BF16 dot product, preserving RTP's scaling precision rather than rounding
pre-scaled Q back to BF16. RTP adaptations are
independent BHSD strides, stable selection compaction with a device count, range
masking, and zero output for an empty row. Counts replace the upstream causal
row-length assumption, preserving duplicates and arbitrary selected token order.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_sm90_selection(
    selected,
    packed,
    stride_b,
    stride_s,
    S,
    T,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // S
    query = row % S
    slot = tl.arange(0, BLOCK)
    token = tl.load(
        selected + batch * stride_b + query * stride_s + slot,
        slot < WIDTH,
        other=-1,
    )
    valid = (slot < WIDTH) & (token >= 0) & (token < T)
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    base = packed + row * (WIDTH + 1)
    tl.store(base + rank, token, valid)
    tl.store(base + slot, -1, (slot < WIDTH) & (slot >= count))
    tl.store(base + WIDTH, count)


@triton.jit
def _sparse_gqa_prefill_sm90(
    q,
    k,
    v,
    out,
    packed,
    scale,
    sq_b,
    sq_h,
    sq_s,
    sk_b,
    sk_s,
    sv_b,
    sv_s,
    so_b,
    so_h,
    so_s,
    S,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    idx_row = packed + (batch * S + query) * (WIDTH + 1)
    count = tl.load(idx_row + WIDTH)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q + batch * sq_b + query * sq_s + offs_h[:, None] * sq_h + offs_d[None, :],
        mask=(offs_h < 12)[:, None],
        other=0.0,
    )
    k_base = k + batch * sk_b
    v_base = v + batch * sv_b
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, count, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current, mask=current < count, other=-1)
        valid = (current < count) & (token >= 0) & (token < S)
        keys = tl.load(
            k_base + token[None, :] * sk_s + offs_d[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_s + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        # Keep scaling in FP32: rounding pre-scaled Q to BF16 causes
        # avoidable attention errors when the Q/K magnitude grows.
        scores = tl.where(
            valid[None, :], tl.dot(q_values, keys) * scale * 1.4426950408, -float("inf")
        )
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = tl.where(normalizer[:, None] > 0, accumulator / normalizer[:, None], 0.0)
    tl.store(
        out + batch * so_b + query * so_s + offs_h[:, None] * so_h + offs_d[None, :],
        output,
        mask=(offs_h < 12)[:, None],
    )


def supports_sm90_sparse_prefill(q, k, v, selected, *, blk_k=64):
    """Shape-only support check; no selected data read or CUDA synchronization."""
    if not (
        q.is_cuda
        and torch.version.hip is None
        and q.ndim == k.ndim == v.ndim == 4
        and selected.ndim == 3
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and selected.dtype == torch.int32
        and q.device == k.device == v.device == selected.device
        and blk_k == 64
    ):
        return False
    batch, heads, length, dim = q.shape
    if not (
        1 <= batch <= 2
        and heads == 12
        and dim == 256
        and 128 <= length <= 20480
        and k.shape == v.shape == (batch, 1, length, 256)
        and selected.shape == (batch, length, 2051)
        and all(t.stride(-1) == 1 for t in (q, k, v, selected))
        and all(stride > 0 for t in (q, k, v, selected) for stride in t.stride())
    ):
        return False
    return torch.cuda.get_device_capability(q.device)[0] == 9


def try_sm90_sparse_prefill(q, k, v, selected, *, blk_k=64):
    """Return the adopted result, or None for the existing generic fallback."""
    if not supports_sm90_sparse_prefill(q, k, v, selected, blk_k=blk_k):
        return None
    batch, _, length, _ = q.shape
    packed = torch.empty((batch, length, 2052), device=q.device, dtype=torch.int32)
    _compact_sm90_selection[(batch * length,)](
        selected,
        packed,
        selected.stride(0),
        selected.stride(1),
        S=length,
        T=length,
        WIDTH=2051,
        BLOCK=4096,
        num_warps=8,
    )
    # Preserve upstream launch choices. Long target shapes share one config.
    if "H20" in torch.cuda.get_device_name(q.device):
        block_n, warps = (32, 4) if batch * length <= 1024 else (16, 1)
    elif batch * length <= 128:
        block_n, warps = 64, 4
    elif batch * length <= 512:
        block_n, warps = 32, 4
    else:
        block_n, warps = 16, 1
    # Return the public BHSD shape backed by NHD storage, avoiding a copy in
    # the packed-token consumer when it transposes the result back to NHD.
    out = torch.empty((batch, length, 12, 256), device=q.device, dtype=q.dtype).permute(
        0, 2, 1, 3
    )
    _sparse_gqa_prefill_sm90[(length, batch)](
        q,
        k,
        v,
        out,
        packed,
        256**-0.5,
        *q.stride()[:3],
        k.stride(0),
        k.stride(2),
        v.stride(0),
        v.stride(2),
        *out.stride()[:3],
        S=length,
        WIDTH=2051,
        BLOCK_M=16,
        BLOCK_N=block_n,
        HEAD_DIM=256,
        num_warps=warps,
        num_stages=2,
    )
    return out
