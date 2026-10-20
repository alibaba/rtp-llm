"""Opt-in SM120 draft attention over packed inputs and HND main cache."""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .draft_layout_triton import _enabled
from .grouped_sparse_paged_fmha import _grouped_sparse_paged_split_kernel
from .sparse_paged_fmha import main_kv_scale_view


@triton.jit
def _merge_draft_packed_splits(
    PARTIAL,
    INVERSE,
    VALID,
    GATE,
    OUT,
    SPLITS: tl.constexpr,
    WITH_GATE: tl.constexpr,
):
    packed, head = tl.program_id(0), tl.program_id(1)
    inverse = tl.load(INVERSE + packed)
    valid = tl.load(VALID + packed)
    row = ((inverse // 4) * 12 + head) * 4 + inverse % 4
    splits = tl.arange(0, SPLITS)
    dim = tl.arange(0, 256)
    base = PARTIAL + row * SPLITS * 258
    maxima = tl.load(base + splits * 258 + 256)
    sums = tl.load(base + splits * 258 + 257)
    maximum = tl.max(maxima, 0)
    scale = tl.where(sums > 0, tl.exp(maxima - maximum), 0.0)
    partial = tl.load(base + splits[:, None] * 258 + dim[None, :])
    numerator = tl.sum(partial * scale[:, None], 0)
    denominator = tl.sum(sums * scale, 0)
    value = tl.where(denominator > 0, numerator / denominator, 0.0).to(tl.bfloat16)
    value = tl.where(valid, value, 0).to(tl.float32)
    offset = packed * 3072 + head * 256 + dim
    if WITH_GATE:
        gate = tl.load(GATE + offset).to(tl.float32)
        sigmoid = (1.0 / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        value *= sigmoid
    tl.store(OUT + offset, value)


def try_draft_paged_attention(
    qkv,
    selected,
    sources,
    inverse,
    valid,
    visible,
    cache,
    table,
    *,
    heads,
    kv_heads,
    head_dim,
    page_size,
    kv_scale=None,
    gate=None,
):
    """Return None outside the measured domain, before launching any work.

    Ownership maps come from draft_graph_rows. Address fusion keeps the same
    grouped split policy, compensated BF16 PV and BF16 output/gate boundaries.
    """
    if not (
        _enabled(qkv)
        and os.environ.get("QWEN4_DRAFT_QSA_DIRECT", "0") == "1"
        and (heads, kv_heads, head_dim, page_size) == (12, 1, 256, 128)
        and qkv.ndim == 2
        and qkv.shape[1] >= 3072
        and qkv.stride(1) == 1
        and qkv.stride(0) >= qkv.shape[1]
        and visible.ndim == 2
        and visible.shape[1] == 4
        and 1 <= visible.shape[0] <= 8
        and visible.numel() == qkv.shape[0]
        and visible.dtype == torch.int32
        and selected.shape == (qkv.shape[0], 2051)
        and selected.dtype == torch.int32
        and selected.stride(1) == 1
        and selected.stride(0) >= 2051
        and sources.shape == inverse.shape == valid.shape == (qkv.shape[0],)
        and sources.dtype == inverse.dtype == torch.int64
        and valid.dtype == torch.bool
        and table.ndim == 2
        and table.shape[0] == visible.shape[0]
        and table.shape[1] > 0
        and table.dtype == torch.int32
        and cache.dtype == torch.float8_e4m3fn
        and cache.ndim in (2, 5)
        and cache.shape[0] > 0
        and not cache.requires_grad
        and kv_scale is not None
        and not kv_scale.requires_grad
        and all(t.is_contiguous() for t in (sources, inverse, valid, visible, table))
        and all(
            t.device == qkv.device
            for t in (
                selected,
                sources,
                inverse,
                valid,
                visible,
                cache,
                table,
                kv_scale,
            )
        )
        and (
            gate is None
            or (
                gate.shape == (qkv.shape[0], 3072)
                and gate.dtype == torch.bfloat16
                and gate.device == qkv.device
                and gate.is_contiguous()
                and not gate.requires_grad
            )
        )
    ):
        return None
    if cache.ndim == 2:
        if (
            cache.shape[1] < 65536
            or cache.stride(1) != 1
            or cache.stride(0) < cache.shape[1]
        ):
            return None
        cache = torch.as_strided(
            cache,
            (cache.shape[0], 2, 1, 128, 256),
            (cache.stride(0), 32768, 32768, 256, 1),
        )
    if cache.shape[1:] != (2, 1, 128, 256) or cache.stride()[1:] != (
        32768,
        32768,
        256,
        1,
    ):
        return None
    if cache.stride(0) < 65536:
        return None
    try:
        scales = main_kv_scale_view(cache, kv_scale, kv_heads=1, page_size=128)
    except ValueError:
        return None
    if scales.stride(0) < 256:
        return None
    batch = visible.shape[0]
    rows = batch * 4
    splits = 32 if batch in (1, 4) else 8
    block_k = 128 if batch == 1 else 64
    partial = torch.empty(
        (batch, 12, 4, splits, 258), device=qkv.device, dtype=torch.float32
    )
    output = torch.empty((rows, 3072), device=qkv.device, dtype=torch.bfloat16)
    _grouped_sparse_paged_split_kernel[(batch, 4, splits)](
        qkv,
        cache,
        scales,
        table,
        visible,
        selected,
        partial,
        QB=0,
        QH=256,
        QS=0,
        CP=cache.stride(0),
        CKV=cache.stride(1),
        CH=cache.stride(2),
        CT=cache.stride(3),
        BTW=table.shape[1],
        S=4,
        PAGE=128,
        PAGES=cache.shape[0],
        WIDTH=2051,
        ROW_WIDTH=selected.stride(0),
        HAS_COUNT=False,
        BK=block_k,
        NS=splits,
        NB=triton.cdiv(2051, block_k * splits),
        HQ=12,
        HG=12,
        FP8_CACHE=True,
        SP=scales.stride(0),
        SKV=scales.stride(1),
        SH=scales.stride(2),
        ST=scales.stride(3),
        ROTATE_TILES=batch in (1, 4),
        FP16_PV=False,
        DRAFT_SOURCES=sources,
        DRAFT_INPUT=True,
        DRAFT_QS=qkv.stride(0),
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    _merge_draft_packed_splits[(rows, 12)](
        partial,
        inverse,
        valid,
        gate if gate is not None else output,
        output,
        SPLITS=splits,
        WITH_GATE=gate is not None,
    )
    return output
