# SPDX-License-Identifier: Apache-2.0
"""Experimental SM120 grouped QSA with compensated BF16 probability MMA."""

import torch
import triton
import triton.language as tl


def is_supported(
    q, cache, table, lengths, selected, partial, *, page_size, splits, block_k
):
    tensors = (q, cache, table, lengths, selected, partial)
    if torch.version.hip is not None or not all(
        t.is_cuda and t.device == q.device for t in tensors
    ):
        return False
    return (
        q.dtype == cache.dtype == torch.bfloat16
        and q.ndim == 4
        and q.shape[0] == 8
        and q.shape[1] == 3
        and 1 <= q.shape[2] <= 4
        and q.shape[3] == 256
        and cache.ndim == 5
        and tuple(cache.shape[1:]) == (2, 1, 128, 256)
        and cache.stride(-1) == 1
        and table.ndim == 2
        and table.shape[0] == 8
        and table.dtype == torch.int32
        and lengths.shape == (8, q.shape[2])
        and lengths.dtype == torch.int32
        and selected.shape == (8, q.shape[2], 2051)
        and selected.dtype == torch.int32
        and partial.shape == (8, 3, q.shape[2], 8, 258)
        and partial.dtype == torch.float32
        and all(t.is_contiguous() for t in (q, table, lengths, selected, partial))
        and page_size == 128
        and splits == 8
        and block_k == 64
        and torch.cuda.get_device_capability(q.device) == (12, 0)
    )


@triton.jit
def _grouped_sparse_paged_split_kernel(
    q,
    cache,
    table,
    lengths,
    selected,
    partial,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QS: tl.constexpr,
    CP: tl.constexpr,
    CKV: tl.constexpr,
    CH: tl.constexpr,
    CT: tl.constexpr,
    BTW: tl.constexpr,
    S: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    WIDTH: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    NB: tl.constexpr,
):
    batch = tl.program_id(0)
    query = tl.program_id(1)
    split = tl.program_id(2)
    head = tl.arange(0, 16)
    dim = tl.arange(0, 256)
    cols = tl.arange(0, BK)
    query_values = tl.load(
        q + batch * QB + head[:, None] * QH + query * QS + dim[None, :],
        mask=head[:, None] < 3,
        other=0,
    )
    length = tl.load(lengths + batch * S + query)
    row = selected + (batch * S + query) * WIDTH
    maximum = tl.full((16,), float("-inf"), tl.float32)
    normalizer = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, 256), tl.float32)
    for step in range(NB):
        offset = (split * NB + step) * BK + cols
        token = tl.load(row + offset, mask=offset < WIDTH, other=-1)
        logical = token // PAGE
        physical = tl.load(
            table + batch * BTW + logical,
            mask=(token >= 0) & (token < length) & (logical < BTW),
            other=0,
        )
        valid = (
            (offset < WIDTH)
            & (token >= 0)
            & (token < length)
            & (logical < BTW)
            & (physical > 0)
            & (physical < PAGES)
        )
        kp = (
            cache + physical[:, None] * CP + (token % PAGE)[:, None] * CT + dim[None, :]
        )
        key = tl.load(kp, mask=valid[:, None], other=0)
        score = tl.dot(query_values, tl.trans(key)) * (1.0 / 16.0)
        score = tl.where(valid[None, :], score, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.where(normalizer > 0, tl.exp(maximum - new_max), 0.0)
        prob = tl.where(valid[None, :], tl.exp(score - new_max[:, None]), 0.0)
        value = tl.load(kp + CKV, mask=valid[:, None], other=0).to(tl.float32)
        prob_hi = prob.to(tl.bfloat16)
        prob_lo = (prob - prob_hi.to(tl.float32)).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(prob_hi, value.to(tl.bfloat16))
        acc += tl.dot(prob_lo, value.to(tl.bfloat16))
        normalizer = normalizer * alpha + tl.sum(prob, 1)
        maximum = new_max
    base = ((batch * 3 + head) * S + query) * NS * 258 + split * 258
    tl.store(partial + base[:, None] + dim[None, :], acc, mask=head[:, None] < 3)
    tl.store(partial + base + 256, maximum, mask=head < 3)
    tl.store(partial + base + 257, normalizer, mask=head < 3)


def grouped_sparse_paged_split(
    q, cache, table, lengths, selected, partial, *, page_size, splits
):
    """Write FP32 split states using the original merge layout.

    K/V are shared across the three query heads. Softmax and accumulation stay
    FP32; two BF16 probability components compensate Tensor Core rounding.
    The caller validates metadata and gates the experimental geometry.
    """
    batch, heads, query_len, dim = q.shape
    width = selected.shape[-1]
    _grouped_sparse_paged_split_kernel[(batch, query_len, splits)](
        q,
        cache,
        table,
        lengths,
        selected,
        partial,
        QB=q.stride(0),
        QH=q.stride(1),
        QS=q.stride(2),
        CP=cache.stride(0),
        CKV=cache.stride(1),
        CH=cache.stride(2),
        CT=cache.stride(3),
        BTW=table.shape[1],
        S=query_len,
        PAGE=page_size,
        PAGES=cache.shape[0],
        WIDTH=width,
        BK=64,
        NS=splits,
        NB=triton.cdiv(width, 64 * splits),
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
