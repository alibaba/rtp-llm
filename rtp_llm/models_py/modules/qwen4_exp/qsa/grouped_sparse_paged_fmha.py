# SPDX-License-Identifier: Apache-2.0
"""Experimental SM120 grouped QSA with compensated BF16 probability MMA."""

import torch
import triton
import triton.language as tl


def preferred_splits(q, cache, selected, *, page_size, block_k):
    """Measured FP8 TP2 shapes; keep the original split policy elsewhere."""
    if not (
        q.ndim == 4
        and cache.ndim == 5
        and q.dtype == torch.bfloat16
        and cache.dtype == torch.float8_e4m3fn
        and q.shape[1] == 12
        and q.shape[3] == 256
        and cache.shape[2] == 1
        and q.shape[0] in (1, 4)
        and q.shape[2] in (1, 4)
        and selected.shape[:2] == (q.shape[0], q.shape[2])
        and selected.shape[-1] in (2051, 2052)
        and page_size == 128
        and block_k == 64
        and q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (12, 0)
    ):
        return None
    return 32


def is_supported(
    q,
    cache,
    table,
    lengths,
    selected,
    partial,
    *,
    page_size,
    splits,
    block_k,
    scale=None,
    selected_has_count=False,
):
    if cache.ndim != 5 or q.ndim != 4:
        return False
    heads, kv_heads = q.shape[1], cache.shape[2]
    if kv_heads not in (1, 2) or heads % kv_heads:
        return False
    group_heads = heads // kv_heads
    # Smaller decode batches benefit only for the larger GQA head group.
    # Single-query batches below four keep the lower-latency online path.
    if q.shape[0] < 8 and (
        group_heads != 12 or (q.shape[0] < 4 and q.shape[2] == 1 and splits != 32)
    ):
        return False
    fp8 = cache.dtype == torch.float8_e4m3fn
    if fp8 and (
        scale is None
        or scale.dtype != torch.float32
        or scale.device != q.device
        or tuple(scale.shape) != (cache.shape[0], 2, kv_heads, 128)
    ):
        return False
    tensors = (q, cache, table, lengths, selected, partial) + ((scale,) if fp8 else ())
    if torch.version.hip is not None or not all(
        t.is_cuda and t.device == q.device for t in tensors
    ):
        return False
    return (
        q.dtype == torch.bfloat16
        and cache.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and group_heads in (3, 6, 12)
        and q.ndim == 4
        and 1 <= q.shape[0] <= 8
        and heads in (3, 6, 12, 24)
        and 1 <= q.shape[2] <= 4
        and q.shape[3] == 256
        and cache.ndim == 5
        and tuple(cache.shape[1:]) == (2, kv_heads, 128, 256)
        and cache.stride(-1) == 1
        and table.ndim == 2
        and table.shape[0] == q.shape[0]
        and table.dtype == torch.int32
        and lengths.shape == (q.shape[0], q.shape[2])
        and lengths.dtype == torch.int32
        and selected.shape
        == (q.shape[0], q.shape[2], 2052 if selected_has_count else 2051)
        and selected.dtype == torch.int32
        and partial.shape == (q.shape[0], heads, q.shape[2], splits, 258)
        and partial.dtype == torch.float32
        and all(t.is_contiguous() for t in (q, table, lengths, selected, partial))
        and page_size == 128
        and (
            splits == 8
            or splits
            == preferred_splits(
                q, cache, selected, page_size=page_size, block_k=block_k
            )
        )
        and block_k == 64
        and torch.cuda.get_device_capability(q.device) == (12, 0)
    )


@triton.jit
def _grouped_sparse_paged_split_kernel(
    q,
    cache,
    scale,
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
    ROW_WIDTH: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    NB: tl.constexpr,
    HQ: tl.constexpr,
    HG: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    SP: tl.constexpr,
    SKV: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    ROTATE_TILES: tl.constexpr,
    FP16_PV: tl.constexpr,
):
    batch = tl.program_id(0)
    group_query = tl.program_id(1)
    query = group_query % S
    group = group_query // S
    split = tl.program_id(2)
    head = tl.arange(0, 16)
    global_head = group * HG + head
    dim = tl.arange(0, 256)
    cols = tl.arange(0, BK)
    query_values = tl.load(
        q + batch * QB + global_head[:, None] * QH + query * QS + dim[None, :],
        mask=head[:, None] < HG,
        other=0,
    )
    length = tl.load(lengths + batch * S + query)
    row = selected + (batch * S + query) * ROW_WIDTH
    count = (
        tl.minimum(tl.maximum(tl.load(row + WIDTH), 0), WIDTH) if HAS_COUNT else WIDTH
    )
    maximum = tl.full((16,), float("-inf"), tl.float32)
    normalizer = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, 256), tl.float32)
    # Round-robin tiles keep all useful splits active when 2051 slots need
    # 33 tiles. Contiguous ceil-sized partitions used only 17 of 32 splits.
    if ROTATE_TILES:
        limit = tl.maximum(0, tl.cdiv(count - split * BK, NS * BK))
    else:
        limit = tl.maximum(0, tl.minimum(NB, tl.cdiv(count - split * NB * BK, BK)))
    limit = tl.where(length > 0, limit, 0)
    for step in range(limit):
        if ROTATE_TILES:
            offset = (split + step * NS) * BK + cols
        else:
            offset = (split * NB + step) * BK + cols
        token = tl.load(row + offset, mask=offset < count, other=-1)
        logical = token // PAGE
        physical = tl.load(
            table + batch * BTW + logical,
            mask=(token >= 0) & (token < length) & (logical < BTW),
            other=0,
        )
        valid = (
            (offset < count)
            & (token >= 0)
            & (token < length)
            & (logical < BTW)
            & (physical > 0)
            & (physical < PAGES)
        )
        kp = (
            cache
            + physical[:, None] * CP
            + group * CH
            + (token % PAGE)[:, None] * CT
            + dim[None, :]
        )
        key = tl.load(kp, mask=valid[:, None], other=0.0)
        if FP8_CACHE:
            sp = scale + physical * SP + group * SH + (token % PAGE) * ST
            kscale = tl.load(sp, mask=valid, other=1.0)
            key = (key.to(tl.float32) * kscale[:, None]).to(tl.bfloat16)
        score = tl.dot(query_values, tl.trans(key)) * (1.0 / 16.0)
        score = tl.where(valid[None, :], score, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.where(normalizer > 0, tl.exp(maximum - new_max), 0.0)
        prob = tl.where(valid[None, :], tl.exp(score - new_max[:, None]), 0.0)
        value = tl.load(kp + CKV, mask=valid[:, None], other=0.0).to(tl.float32)
        if FP8_CACHE:
            vscale = tl.load(sp + SKV, mask=valid, other=1.0)
            value *= vscale[:, None]
        # Preserve the BF16 dequant boundary. A bounded alternative uses FP16
        # probabilities; keep compensated BF16 for values outside FP16 range.
        value_bf16 = value.to(tl.bfloat16)
        if FP16_PV:
            safe_fp16 = (
                tl.max(tl.max(tl.abs(value_bf16.to(tl.float32)), 1), 0) <= 65504.0
            )
            if safe_fp16:
                acc = tl.dot(
                    prob.to(tl.float16), value_bf16.to(tl.float16), acc * alpha[:, None]
                )
            else:
                prob_hi = prob.to(tl.bfloat16)
                prob_lo = (prob - prob_hi.to(tl.float32)).to(tl.bfloat16)
                acc = tl.dot(prob_hi, value_bf16, acc * alpha[:, None])
                acc = tl.dot(prob_lo, value_bf16, acc)
        else:
            prob_hi = prob.to(tl.bfloat16)
            prob_lo = (prob - prob_hi.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(prob_hi, value_bf16, acc * alpha[:, None])
            acc = tl.dot(prob_lo, value_bf16, acc)
        normalizer = normalizer * alpha + tl.sum(prob, 1)
        maximum = new_max
    base = ((batch * HQ + global_head) * S + query) * NS * 258 + split * 258
    tl.store(partial + base[:, None] + dim[None, :], acc, mask=head[:, None] < HG)
    tl.store(partial + base + 256, maximum, mask=head < HG)
    tl.store(partial + base + 257, normalizer, mask=head < HG)


def grouped_sparse_paged_split(
    q,
    cache,
    table,
    lengths,
    selected,
    partial,
    *,
    page_size,
    splits,
    scale=None,
    selected_has_count=False,
    fp16_pv=False,
):
    """Write FP32 split states using the original merge layout.

    K/V are shared across the three query heads. Softmax and accumulation stay
    FP32; two BF16 probability components compensate Tensor Core rounding.
    The caller validates metadata and selects the verified shape domain.
    """
    batch, heads, query_len, dim = q.shape
    row_width = selected.shape[-1]
    width = row_width - int(selected_has_count)
    kv_heads = cache.shape[2]
    fp8 = cache.dtype == torch.float8_e4m3fn
    strides = scale.stride() if fp8 else (0, 0, 0, 0)
    tuned = fp8 and splits == preferred_splits(
        q, cache, selected, page_size=page_size, block_k=64
    )
    # The measured M4 target/draft domains benefit from wider KV tiles.
    # Keep per-token scales, BF16 dequant and compensated PV unchanged.
    block_k = (
        128 if tuned and not fp16_pv and (batch, query_len) in ((1, 4), (4, 1)) else 64
    )
    _grouped_sparse_paged_split_kernel[(batch, kv_heads * query_len, splits)](
        q,
        cache,
        cache if scale is None else scale,
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
        ROW_WIDTH=row_width,
        HAS_COUNT=selected_has_count,
        BK=block_k,
        NS=splits,
        NB=triton.cdiv(width, block_k * splits),
        HQ=heads,
        HG=heads // kv_heads,
        FP8_CACHE=fp8,
        SP=strides[0],
        SKV=strides[1],
        SH=strides[2],
        ST=strides[3],
        ROTATE_TILES=tuned,
        FP16_PV=fp16_pv,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
