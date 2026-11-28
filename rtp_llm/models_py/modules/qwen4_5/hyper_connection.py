"""Qwen 4.5-Flash 4-stream Hyper-Connections with hc_prev_mix and GemmaRMSNorm+bias.

Implements:
  - Per-branch GemmaRMSNorm (1.0 + weight) with affine bias (Triton + FP32 PyTorch ref)
  - Fused down + block-inject projection and activation epilogue
  - hc_prev_mix gate routing (head / normal / non-emitting / final mixer modes)
  - Fused branch mix-reduce and residual stream combine (+ combine-and-norm fusion)
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_5.short_conv import _triton_int_spec, triton_fast_launch

from rtp_llm.models_py.modules.qwen4_5.short_conv import _causal_dwconv1d_decode_opaque_k4_kernel

_HC_TRITON_CACHE: dict = {}

# Eager-only stream-keyed static scratch for strictly internal 'normed' tensors
# (never returned to callers) plus caller-owned HCStaticWorkspace for intermediate layer slots.
_HC_STATIC_WS_MAX_BYTES = 16 * 1024 * 1024
_HC_INTERNAL_NORMED_POOL: dict = {}


def _hc_alloc_from_pool(
    pool: dict,
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    tag: str,
    n_slots: int,
) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= int(s)
    if (
        numel == 0
        or device.type != "cuda"
        or torch.is_grad_enabled()
        or torch.cuda.is_current_stream_capturing()
    ):
        return torch.empty(shape, dtype=dtype, device=device)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    if nbytes > _HC_STATIC_WS_MAX_BYTES:
        return torch.empty(shape, dtype=dtype, device=device)
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(device).cuda_stream)
    key = (dev_idx, stream_id, tag, dtype)
    entry = pool.get(key)
    if entry is None:
        bufs = [torch.empty((numel,), dtype=dtype, device=device) for _ in range(n_slots)]
        entry = [bufs, 0]
        pool[key] = entry
    bufs, idx = entry
    buf = bufs[idx]
    if int(buf.numel()) < numel:
        buf = torch.empty((numel,), dtype=dtype, device=device)
        bufs[idx] = buf
    entry[1] = (idx + 1) % n_slots
    return buf[:numel].view(*shape)


class HCStaticWorkspace:
    """Per-model/per-caller static workspace for intermediate HC state tensors (2x2 ping-pong safe)."""

    _SLOTS = {"combined": 4, "out": 4, "latent": 4, "inj": 4}

    def __init__(self) -> None:
        self._pool: dict = {}

    def empty(self, shape: tuple, dtype: torch.dtype, device: torch.device, tag: str) -> torch.Tensor:
        return _hc_alloc_from_pool(self._pool, shape, dtype, device, tag, self._SLOTS.get(tag, 4))


def _hc_static_empty(
    shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    tag: str,
    ws: Optional[HCStaticWorkspace] = None,
) -> torch.Tensor:
    """Allocate from ws when provided, from internal 1-slot pool for strictly dead 'normed', or fresh."""
    if ws is not None:
        return ws.empty(shape, dtype, device, tag)
    if tag in ("normed", "sk_part"):
        return _hc_alloc_from_pool(_HC_INTERNAL_NORMED_POOL, shape, dtype, device, tag, 1)
    return torch.empty(shape, dtype=dtype, device=device)



# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _grouped_gemma_rmsnorm_bias_kernel(
    X_PTR,
    W_PTR,
    B_PTR,
    OUT_PTR,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    X_STRIDE: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP_SIZE
    col = group * GROUP_SIZE + offs
    x_offs = row * X_STRIDE + (offs if SINGLE_STREAM else col)
    x = tl.load(X_PTR + x_offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_PTR + col, mask=mask, other=0.0).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / GROUP_SIZE + EPS)
    out = x * inv_rms * (1.0 + w)
    if HAS_BIAS:
        b = tl.load(B_PTR + col, mask=mask, other=0.0).to(tl.float32)
        out = out + b
    tl.store(OUT_PTR + row * WIDTH + col, out, mask=mask)


@triton.jit
def _grouped_gemma_rmsnorm_bias_chunked_kernel(
    X_PTR,
    W_PTR,
    B_PTR,
    OUT_PTR,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    X_STRIDE: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    out_base = row * WIDTH + group * GROUP_SIZE
    x_base = row * X_STRIDE if SINGLE_STREAM else out_base
    w_base = group * GROUP_SIZE
    sum_sq = tl.zeros((), dtype=tl.float32)
    for start in range(0, GROUP_SIZE, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < GROUP_SIZE
        x = tl.load(X_PTR + x_base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / GROUP_SIZE + EPS)
    for start in range(0, GROUP_SIZE, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < GROUP_SIZE
        x = tl.load(X_PTR + x_base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_PTR + w_base + offs, mask=mask, other=0.0).to(tl.float32)
        out = x * inv_rms * (1.0 + w)
        if HAS_BIAS:
            b = tl.load(B_PTR + w_base + offs, mask=mask, other=0.0).to(tl.float32)
            out = out + b
        tl.store(OUT_PTR + out_base + offs, out, mask=mask)


@triton.jit
def _grouped_gemma_rmsnorm_bias_split5120_kernel(
    X_PTR,
    W_PTR,
    B_PTR,
    OUT_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Exact 4096+1024 split per-branch GemmaRMSNorm+bias with L2-friendly (HC, row) grid."""
    br = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    o0 = tl.arange(0, 4096)
    o1 = 4096 + tl.arange(0, 1024)
    c0 = br * HIDDEN + o0
    c1 = br * HIDDEN + o1
    out_base = row * (HC_COUNT * HIDDEN)
    in0 = (row * HIDDEN + o0) if SINGLE_STREAM else (out_base + c0)
    in1 = (row * HIDDEN + o1) if SINGLE_STREAM else (out_base + c1)
    x0 = tl.load(X_PTR + in0).to(tl.float32)
    x1 = tl.load(X_PTR + in1).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt((tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)) / HIDDEN + EPS)
    y0 = x0 * inv_rms * (1.0 + tl.load(W_PTR + c0).to(tl.float32))
    y1 = x1 * inv_rms * (1.0 + tl.load(W_PTR + c1).to(tl.float32))
    if HAS_BIAS:
        y0 = y0 + tl.load(B_PTR + c0).to(tl.float32)
        y1 = y1 + tl.load(B_PTR + c1).to(tl.float32)
    tl.store(OUT_PTR + out_base + c0, y0)
    tl.store(OUT_PTR + out_base + c1, y1)


@triton.jit
def _hc_combine_kernel(
    RES_PTR,
    OUT_PTR,
    INJ_PTR,
    DST_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN
    sub = tl.load(OUT_PTR + row * HIDDEN + col, mask=mask, other=0.0).to(tl.float32)
    row_base = row * (HC_COUNT * HIDDEN)
    res_single = (
        tl.load(RES_PTR + row * HIDDEN + col, mask=mask, other=0.0).to(tl.float32)
        if SINGLE_STREAM
        else 0.0
    )
    for branch in tl.static_range(HC_COUNT):
        offset = row_base + branch * HIDDEN + col
        res = res_single if SINGLE_STREAM else tl.load(RES_PTR + offset, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(INJ_PTR + row * HC_COUNT + branch).to(tl.float32)
        comb = res + sub * w
        tl.store(DST_PTR + offset, comb, mask=mask)


@triton.jit
def _hc_combine_and_norm_kernel(
    RES_PTR,
    SUB_PTR,
    INJ_PTR,
    W_PTR,
    B_PTR,
    COMB_PTR,
    NORM_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < HIDDEN
    col = branch * HIDDEN + offs
    res_offset = row * (HC_COUNT * HIDDEN) + col
    res_in_offset = (row * HIDDEN + offs) if SINGLE_STREAM else res_offset

    res = tl.load(RES_PTR + res_in_offset, mask=mask, other=0.0).to(tl.float32)
    sub = tl.load(SUB_PTR + row * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
    inj = tl.load(INJ_PTR + row * HC_COUNT + branch).to(tl.float32)
    comb = (res + sub * inj).to(RES_PTR.dtype.element_ty)
    tl.store(COMB_PTR + res_offset, comb, mask=mask)

    comb_f = tl.where(mask, comb.to(tl.float32), 0.0)
    inv_rms = 1.0 / tl.sqrt(tl.sum(comb_f * comb_f, axis=0) / HIDDEN + EPS)
    w = tl.load(W_PTR + col, mask=mask, other=0.0).to(tl.float32)
    normed = comb_f * inv_rms * (1.0 + w)
    if HAS_BIAS:
        b = tl.load(B_PTR + col, mask=mask, other=0.0).to(tl.float32)
        normed = normed + b
    tl.store(NORM_PTR + res_offset, normed, mask=mask)


@triton.jit
def _hc_combine_and_norm_split5120_shared_sub_kernel(
    RES_PTR,
    SUB_PTR,
    INJ_PTR,
    W_PTR,
    B_PTR,
    COMB_PTR,
    NORM_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Exact 4096+1024 split HC combine+norm loading sub once per row across all 4 branches."""
    row = tl.program_id(0).to(tl.int64)
    o0 = tl.arange(0, 4096)
    o1 = 4096 + tl.arange(0, 1024)
    sub0 = tl.load(SUB_PTR + row * HIDDEN + o0)
    sub1 = tl.load(SUB_PTR + row * HIDDEN + o1)
    res_base = row * (HC_COUNT * HIDDEN)
    for br in tl.static_range(HC_COUNT):
        c0 = br * HIDDEN + o0
        c1 = br * HIDDEN + o1
        in0 = (row * HIDDEN + o0) if SINGLE_STREAM else (res_base + c0)
        in1 = (row * HIDDEN + o1) if SINGLE_STREAM else (res_base + c1)
        inj = tl.load(INJ_PTR + row * HC_COUNT + br).to(tl.float32)
        r0 = tl.load(RES_PTR + in0).to(tl.float32)
        r1 = tl.load(RES_PTR + in1).to(tl.float32)
        cb0 = (r0 + sub0.to(tl.float32) * inj).to(RES_PTR.dtype.element_ty)
        cb1 = (r1 + sub1.to(tl.float32) * inj).to(RES_PTR.dtype.element_ty)
        tl.store(COMB_PTR + res_base + c0, cb0)
        tl.store(COMB_PTR + res_base + c1, cb1)
        f0 = cb0.to(tl.float32)
        f1 = cb1.to(tl.float32)
        inv_rms = 1.0 / tl.sqrt(
            (tl.sum(f0 * f0, axis=0) + tl.sum(f1 * f1, axis=0)) / HIDDEN + EPS
        )
        w0 = tl.load(W_PTR + c0).to(tl.float32)
        w1 = tl.load(W_PTR + c1).to(tl.float32)
        n0 = f0 * inv_rms * (1.0 + w0)
        n1 = f1 * inv_rms * (1.0 + w1)
        if HAS_BIAS:
            n0 = n0 + tl.load(B_PTR + c0).to(tl.float32)
            n1 = n1 + tl.load(B_PTR + c1).to(tl.float32)
        tl.store(NORM_PTR + res_base + c0, n0)
        tl.store(NORM_PTR + res_base + c1, n1)


@triton.jit
def _hc_combine_and_norm_branch5120_kernel(
    RES_PTR,
    SUB_PTR,
    INJ_PTR,
    W_PTR,
    B_PTR,
    COMB_PTR,
    NORM_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """4096+1024 branch-parallel HC combine+norm with L2-friendly (HC, row) grid."""
    br = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    o0 = tl.arange(0, 4096)
    o1 = 4096 + tl.arange(0, 1024)
    c0 = br * HIDDEN + o0
    c1 = br * HIDDEN + o1
    res_base = row * (HC_COUNT * HIDDEN)
    in0 = (row * HIDDEN + o0) if SINGLE_STREAM else (res_base + c0)
    in1 = (row * HIDDEN + o1) if SINGLE_STREAM else (res_base + c1)
    inj = tl.load(INJ_PTR + row * HC_COUNT + br).to(tl.float32)
    r0 = tl.load(RES_PTR + in0).to(tl.float32)
    r1 = tl.load(RES_PTR + in1).to(tl.float32)
    s0 = tl.load(SUB_PTR + row * HIDDEN + o0).to(tl.float32)
    s1 = tl.load(SUB_PTR + row * HIDDEN + o1).to(tl.float32)
    res_ty = RES_PTR.dtype.element_ty
    cb0 = (r0 + s0 * inj).to(res_ty)
    cb1 = (r1 + s1 * inj).to(res_ty)
    tl.store(COMB_PTR + res_base + c0, cb0)
    tl.store(COMB_PTR + res_base + c1, cb1)
    f0 = cb0.to(tl.float32)
    f1 = cb1.to(tl.float32)
    inv_rms = 1.0 / tl.sqrt((tl.sum(f0 * f0, axis=0) + tl.sum(f1 * f1, axis=0)) / HIDDEN + EPS)
    n0 = f0 * inv_rms * (1.0 + tl.load(W_PTR + c0).to(tl.float32))
    n1 = f1 * inv_rms * (1.0 + tl.load(W_PTR + c1).to(tl.float32))
    if HAS_BIAS:
        n0 = n0 + tl.load(B_PTR + c0).to(tl.float32)
        n1 = n1 + tl.load(B_PTR + c1).to(tl.float32)
    tl.store(NORM_PTR + res_base + c0, n0)
    tl.store(NORM_PTR + res_base + c1, n1)


@triton.jit(do_not_specialize=["TPB", "MAX_BLOCKS", "NUM_BLOCKS"])
def _hc_sconv_decode_opaque_combine_norm_kernel(
    RES_PTR,
    RAW_SUB_PTR,
    INJ_PTR,
    W_HC_PTR,
    B_HC_PTR,
    CONV_W_PTR,
    POOL_PTR,
    BT_PTR,
    SEQ_BEFORE_PTR,
    COMB_PTR,
    NORM_PTR,
    stride_xb: tl.int64,
    stride_pb: tl.int64,
    stride_bt_b: tl.int64,
    stride_bt_p: tl.int64,
    TPB,
    MAX_BLOCKS,
    NUM_BLOCKS,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CONV_RESIDUAL: tl.constexpr,
    CONV_USE_SILU: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    CLAMP_VAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused paged T=1 K=4 ShortConv(+clamp+pool roll) + HC 4-stream combine+norm (H <= 4096)."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < HIDDEN

    seq_b = tl.load(SEQ_BEFORE_PTR + row).to(tl.int32)
    has_pfx = seq_b > 0
    in_blk = tl.where(has_pfx, (seq_b - 1) // TPB, 0)
    ok_in_blk = has_pfx & (in_blk < MAX_BLOCKS)
    in_phys = tl.load(
        BT_PTR + row * stride_bt_b + tl.minimum(in_blk, MAX_BLOCKS - 1) * stride_bt_p,
        mask=ok_in_blk,
        other=0,
    ).to(tl.int64)
    ok_in = ok_in_blk & (in_phys > 0) & (in_phys < NUM_BLOCKS)
    in_base = POOL_PTR + in_phys * stride_pb + offs * 3
    m_in = ok_in & mask
    s0_raw = tl.load(in_base + 0, mask=m_in, other=0.0)
    s1_raw = tl.load(in_base + 1, mask=m_in, other=0.0)
    s2_raw = tl.load(in_base + 2, mask=m_in, other=0.0)

    x0_raw = tl.load(RAW_SUB_PTR + row * stride_xb + offs, mask=mask, other=0.0)
    cw_base = offs * 4
    cw0 = tl.load(CONV_W_PTR + cw_base + 0, mask=mask, other=0.0).to(tl.float32)
    cw1 = tl.load(CONV_W_PTR + cw_base + 1, mask=mask, other=0.0).to(tl.float32)
    cw2 = tl.load(CONV_W_PTR + cw_base + 2, mask=mask, other=0.0).to(tl.float32)
    cw3 = tl.load(CONV_W_PTR + cw_base + 3, mask=mask, other=0.0).to(tl.float32)
    x0_f = x0_raw.to(tl.float32)
    sub = s0_raw.to(tl.float32) * cw0 + s1_raw.to(tl.float32) * cw1 + s2_raw.to(tl.float32) * cw2 + x0_f * cw3
    if CONV_USE_SILU:
        sub = sub * tl.sigmoid(sub)
    if CONV_RESIDUAL:
        sub = x0_f + sub
    res_ty = RES_PTR.dtype.element_ty
    if HAS_CLAMP:
        sub = sub.to(res_ty).to(tl.float32)
        sub = tl.minimum(tl.maximum(sub, -CLAMP_VAL), CLAMP_VAL)
    sub_bf = sub.to(res_ty)

    out_blk = seq_b // TPB
    ok_out_blk = (seq_b >= 0) & (out_blk < MAX_BLOCKS)
    out_phys = tl.load(
        BT_PTR + row * stride_bt_b + tl.minimum(tl.maximum(out_blk, 0), MAX_BLOCKS - 1) * stride_bt_p,
        mask=ok_out_blk,
        other=0,
    ).to(tl.int64)
    ok_out = ok_out_blk & (out_phys > 0) & (out_phys < NUM_BLOCKS)
    out_base = POOL_PTR + out_phys * stride_pb + offs * 3
    m_out = ok_out & mask
    tl.store(out_base + 0, s1_raw, mask=m_out)
    tl.store(out_base + 1, s2_raw, mask=m_out)
    tl.store(out_base + 2, x0_raw, mask=m_out)

    res_base = row * (HC_COUNT * HIDDEN)
    sub_f = sub_bf.to(tl.float32)
    for br in tl.static_range(HC_COUNT):
        col = br * HIDDEN + offs
        res_off = res_base + col
        res_in_off = (row * HIDDEN + offs) if SINGLE_STREAM else res_off
        inj = tl.load(INJ_PTR + row * HC_COUNT + br).to(tl.float32)
        res = tl.load(RES_PTR + res_in_off, mask=mask, other=0.0).to(tl.float32)
        comb = (res + sub_f * inj).to(res_ty)
        tl.store(COMB_PTR + res_off, comb, mask=mask)
        comb_f = tl.where(mask, comb.to(tl.float32), 0.0)
        inv_rms = 1.0 / tl.sqrt(tl.sum(comb_f * comb_f, axis=0) / HIDDEN + EPS)
        w = tl.load(W_HC_PTR + col, mask=mask, other=0.0).to(tl.float32)
        normed = comb_f * inv_rms * (1.0 + w)
        if HAS_BIAS:
            normed = normed + tl.load(B_HC_PTR + col, mask=mask, other=0.0).to(tl.float32)
        tl.store(NORM_PTR + res_off, normed, mask=mask)


@triton.jit(do_not_specialize=["TPB", "MAX_BLOCKS", "NUM_BLOCKS"])
def _hc_sconv_decode_opaque_combine_norm_split5120_kernel(
    RES_PTR,
    RAW_SUB_PTR,
    INJ_PTR,
    W_HC_PTR,
    B_HC_PTR,
    CONV_W_PTR,
    POOL_PTR,
    BT_PTR,
    SEQ_BEFORE_PTR,
    COMB_PTR,
    NORM_PTR,
    stride_xb: tl.int64,
    stride_pb: tl.int64,
    stride_bt_b: tl.int64,
    stride_bt_p: tl.int64,
    TPB,
    MAX_BLOCKS,
    NUM_BLOCKS,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CONV_RESIDUAL: tl.constexpr,
    CONV_USE_SILU: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    CLAMP_VAL: tl.constexpr,
):
    """Exact 4096+1024 split fused paged T=1 K=4 ShortConv(+clamp+pool roll) + HC combine+norm."""
    row = tl.program_id(0).to(tl.int64)
    o0 = tl.arange(0, 4096).to(tl.int64)
    o1 = 4096 + tl.arange(0, 1024).to(tl.int64)

    seq_b = tl.load(SEQ_BEFORE_PTR + row).to(tl.int32)
    has_pfx = seq_b > 0
    in_blk = tl.where(has_pfx, (seq_b - 1) // TPB, 0)
    ok_in_blk = has_pfx & (in_blk < MAX_BLOCKS)
    in_phys = tl.load(
        BT_PTR + row * stride_bt_b + tl.minimum(in_blk, MAX_BLOCKS - 1) * stride_bt_p,
        mask=ok_in_blk,
        other=0,
    ).to(tl.int64)
    ok_in = ok_in_blk & (in_phys > 0) & (in_phys < NUM_BLOCKS)

    in_b0 = POOL_PTR + in_phys * stride_pb + o0 * 3
    in_b1 = POOL_PTR + in_phys * stride_pb + o1 * 3
    s0_0 = tl.load(in_b0 + 0, mask=ok_in, other=0.0)
    s1_0 = tl.load(in_b0 + 1, mask=ok_in, other=0.0)
    s2_0 = tl.load(in_b0 + 2, mask=ok_in, other=0.0)
    s0_1 = tl.load(in_b1 + 0, mask=ok_in, other=0.0)
    s1_1 = tl.load(in_b1 + 1, mask=ok_in, other=0.0)
    s2_1 = tl.load(in_b1 + 2, mask=ok_in, other=0.0)

    x0_0 = tl.load(RAW_SUB_PTR + row * stride_xb + o0)
    x0_1 = tl.load(RAW_SUB_PTR + row * stride_xb + o1)
    cw_b0 = o0 * 4
    cw_b1 = o1 * 4
    sub0 = (
        s0_0.to(tl.float32) * tl.load(CONV_W_PTR + cw_b0 + 0).to(tl.float32)
        + s1_0.to(tl.float32) * tl.load(CONV_W_PTR + cw_b0 + 1).to(tl.float32)
        + s2_0.to(tl.float32) * tl.load(CONV_W_PTR + cw_b0 + 2).to(tl.float32)
        + x0_0.to(tl.float32) * tl.load(CONV_W_PTR + cw_b0 + 3).to(tl.float32)
    )
    sub1 = (
        s0_1.to(tl.float32) * tl.load(CONV_W_PTR + cw_b1 + 0).to(tl.float32)
        + s1_1.to(tl.float32) * tl.load(CONV_W_PTR + cw_b1 + 1).to(tl.float32)
        + s2_1.to(tl.float32) * tl.load(CONV_W_PTR + cw_b1 + 2).to(tl.float32)
        + x0_1.to(tl.float32) * tl.load(CONV_W_PTR + cw_b1 + 3).to(tl.float32)
    )
    if CONV_USE_SILU:
        sub0 = sub0 * tl.sigmoid(sub0)
        sub1 = sub1 * tl.sigmoid(sub1)
    if CONV_RESIDUAL:
        sub0 = x0_0.to(tl.float32) + sub0
        sub1 = x0_1.to(tl.float32) + sub1
    res_ty = RES_PTR.dtype.element_ty
    if HAS_CLAMP:
        sub0 = tl.minimum(tl.maximum(sub0.to(res_ty).to(tl.float32), -CLAMP_VAL), CLAMP_VAL)
        sub1 = tl.minimum(tl.maximum(sub1.to(res_ty).to(tl.float32), -CLAMP_VAL), CLAMP_VAL)
    sub0_f = sub0.to(res_ty).to(tl.float32)
    sub1_f = sub1.to(res_ty).to(tl.float32)

    out_blk = seq_b // TPB
    ok_out_blk = (seq_b >= 0) & (out_blk < MAX_BLOCKS)
    out_phys = tl.load(
        BT_PTR + row * stride_bt_b + tl.minimum(tl.maximum(out_blk, 0), MAX_BLOCKS - 1) * stride_bt_p,
        mask=ok_out_blk,
        other=0,
    ).to(tl.int64)
    ok_out = ok_out_blk & (out_phys > 0) & (out_phys < NUM_BLOCKS)
    out_b0 = POOL_PTR + out_phys * stride_pb + o0 * 3
    out_b1 = POOL_PTR + out_phys * stride_pb + o1 * 3
    tl.store(out_b0 + 0, s1_0, mask=ok_out)
    tl.store(out_b0 + 1, s2_0, mask=ok_out)
    tl.store(out_b0 + 2, x0_0, mask=ok_out)
    tl.store(out_b1 + 0, s1_1, mask=ok_out)
    tl.store(out_b1 + 1, s2_1, mask=ok_out)
    tl.store(out_b1 + 2, x0_1, mask=ok_out)

    res_base = row * (HC_COUNT * HIDDEN)
    for br in tl.static_range(HC_COUNT):
        c0 = br * HIDDEN + o0
        c1 = br * HIDDEN + o1
        in0 = (row * HIDDEN + o0) if SINGLE_STREAM else (res_base + c0)
        in1 = (row * HIDDEN + o1) if SINGLE_STREAM else (res_base + c1)
        inj = tl.load(INJ_PTR + row * HC_COUNT + br).to(tl.float32)
        r0 = tl.load(RES_PTR + in0).to(tl.float32)
        r1 = tl.load(RES_PTR + in1).to(tl.float32)
        cb0 = (r0 + sub0_f * inj).to(res_ty)
        cb1 = (r1 + sub1_f * inj).to(res_ty)
        tl.store(COMB_PTR + res_base + c0, cb0)
        tl.store(COMB_PTR + res_base + c1, cb1)
        f0 = cb0.to(tl.float32)
        f1 = cb1.to(tl.float32)
        inv_rms = 1.0 / tl.sqrt((tl.sum(f0 * f0, axis=0) + tl.sum(f1 * f1, axis=0)) / HIDDEN + EPS)
        w0 = tl.load(W_HC_PTR + c0).to(tl.float32)
        w1 = tl.load(W_HC_PTR + c1).to(tl.float32)
        n0 = f0 * inv_rms * (1.0 + w0)
        n1 = f1 * inv_rms * (1.0 + w1)
        if HAS_BIAS:
            n0 = n0 + tl.load(B_HC_PTR + c0).to(tl.float32)
            n1 = n1 + tl.load(B_HC_PTR + c1).to(tl.float32)
        tl.store(NORM_PTR + res_base + c0, n0)
        tl.store(NORM_PTR + res_base + c1, n1)


@triton.jit
def _hc_dflash_gconv_combine_norm_kernel(
    RES_PTR,
    RAW_SUB_PTR,
    GCONV_DELTA_PTR,
    GCONV_BASE_PTR,
    INJ_PTR,
    W_PTR,
    B_PTR,
    COMB_PTR,
    NORM_PTR,
    X_STRIDE_ROW: tl.int64,
    DELTA_STRIDE_ROW: tl.int64,
    DELTA_STRIDE_TAP: tl.int64,
    BASE_STRIDE_TAP: tl.int64,
    CLIP_VAL: tl.float32,
    HAS_CLIP: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-pass DFlashGroupedConv.finish + HC combine + per-branch GemmaRMSNorm+bias."""
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < HIDDEN
    groups = offs // GROUP_SIZE
    pos_in_blk = row % BLOCK_SIZE

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for tap in tl.static_range(0, TAPS):
        src_row = row - tap
        valid = pos_in_blk >= tap
        safe_row = tl.where(valid, src_row, 0)
        x_val = tl.load(
            RAW_SUB_PTR + safe_row * X_STRIDE_ROW + offs,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)
        base_val = tl.load(
            GCONV_BASE_PTR + tap * BASE_STRIDE_TAP + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        delta_val = tl.load(
            GCONV_DELTA_PTR + row * DELTA_STRIDE_ROW + tap * DELTA_STRIDE_TAP + groups,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += (base_val + delta_val) * x_val

    if HAS_CLIP:
        acc = tl.minimum(tl.maximum(acc, -CLIP_VAL), CLIP_VAL)
    sub_f = acc.to(RES_PTR.dtype.element_ty).to(tl.float32)

    col = branch * HIDDEN + offs
    res_offset = row * (HC_COUNT * HIDDEN) + col
    res_in_offset = (row * HIDDEN + offs) if SINGLE_STREAM else res_offset
    res = tl.load(RES_PTR + res_in_offset, mask=mask, other=0.0).to(tl.float32)
    inj = tl.load(INJ_PTR + row * HC_COUNT + branch).to(tl.float32)
    comb = (res + sub_f * inj).to(RES_PTR.dtype.element_ty)
    tl.store(COMB_PTR + res_offset, comb, mask=mask)

    comb_f = tl.where(mask, comb.to(tl.float32), 0.0)
    inv_rms = 1.0 / tl.sqrt(tl.sum(comb_f * comb_f, axis=0) / HIDDEN + EPS)
    w = tl.load(W_PTR + col, mask=mask, other=0.0).to(tl.float32)
    normed = comb_f * inv_rms * (1.0 + w)
    if HAS_BIAS:
        b = tl.load(B_PTR + col, mask=mask, other=0.0).to(tl.float32)
        normed = normed + b
    tl.store(NORM_PTR + res_offset, normed, mask=mask)


@triton.jit
def _hc_dflash_gconv_combine_norm_split5120_kernel(
    RES_PTR,
    RAW_SUB_PTR,
    GCONV_DELTA_PTR,
    GCONV_BASE_PTR,
    INJ_PTR,
    W_PTR,
    B_PTR,
    COMB_PTR,
    NORM_PTR,
    X_STRIDE_ROW: tl.int64,
    DELTA_STRIDE_ROW: tl.int64,
    DELTA_STRIDE_TAP: tl.int64,
    BASE_STRIDE_TAP: tl.int64,
    CLIP_VAL: tl.float32,
    HAS_CLIP: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """4096+1024 split DFlashGroupedConv.finish + HC 4-stream combine+norm."""
    row = tl.program_id(0).to(tl.int64)
    o0 = tl.arange(0, 4096).to(tl.int64)
    o1 = 4096 + tl.arange(0, 1024).to(tl.int64)
    g0 = o0 // GROUP_SIZE
    g1 = o1 // GROUP_SIZE
    pos_in_blk = row % BLOCK_SIZE

    acc0 = tl.zeros((4096,), dtype=tl.float32)
    acc1 = tl.zeros((1024,), dtype=tl.float32)
    for tap in tl.static_range(0, TAPS):
        src_row = row - tap
        valid = pos_in_blk >= tap
        safe_row = tl.where(valid, src_row, 0)
        x0 = tl.load(RAW_SUB_PTR + safe_row * X_STRIDE_ROW + o0, mask=valid, other=0.0).to(tl.float32)
        x1 = tl.load(RAW_SUB_PTR + safe_row * X_STRIDE_ROW + o1, mask=valid, other=0.0).to(tl.float32)
        b0 = tl.load(GCONV_BASE_PTR + tap * BASE_STRIDE_TAP + o0).to(tl.float32)
        b1 = tl.load(GCONV_BASE_PTR + tap * BASE_STRIDE_TAP + o1).to(tl.float32)
        d0 = tl.load(GCONV_DELTA_PTR + row * DELTA_STRIDE_ROW + tap * DELTA_STRIDE_TAP + g0).to(tl.float32)
        d1 = tl.load(GCONV_DELTA_PTR + row * DELTA_STRIDE_ROW + tap * DELTA_STRIDE_TAP + g1).to(tl.float32)
        acc0 += (b0 + d0) * x0
        acc1 += (b1 + d1) * x1

    if HAS_CLIP:
        acc0 = tl.minimum(tl.maximum(acc0, -CLIP_VAL), CLIP_VAL)
        acc1 = tl.minimum(tl.maximum(acc1, -CLIP_VAL), CLIP_VAL)
    res_ty = RES_PTR.dtype.element_ty
    sub0_f = acc0.to(res_ty).to(tl.float32)
    sub1_f = acc1.to(res_ty).to(tl.float32)

    res_base = row * (HC_COUNT * HIDDEN)
    for br in tl.static_range(HC_COUNT):
        c0 = br * HIDDEN + o0
        c1 = br * HIDDEN + o1
        in0 = (row * HIDDEN + o0) if SINGLE_STREAM else (res_base + c0)
        in1 = (row * HIDDEN + o1) if SINGLE_STREAM else (res_base + c1)
        inj = tl.load(INJ_PTR + row * HC_COUNT + br).to(tl.float32)
        r0 = tl.load(RES_PTR + in0).to(tl.float32)
        r1 = tl.load(RES_PTR + in1).to(tl.float32)
        cb0 = (r0 + sub0_f * inj).to(res_ty)
        cb1 = (r1 + sub1_f * inj).to(res_ty)
        tl.store(COMB_PTR + res_base + c0, cb0)
        tl.store(COMB_PTR + res_base + c1, cb1)
        f0 = cb0.to(tl.float32)
        f1 = cb1.to(tl.float32)
        inv_rms = 1.0 / tl.sqrt((tl.sum(f0 * f0, axis=0) + tl.sum(f1 * f1, axis=0)) / HIDDEN + EPS)
        w0 = tl.load(W_PTR + c0).to(tl.float32)
        w1 = tl.load(W_PTR + c1).to(tl.float32)
        n0 = f0 * inv_rms * (1.0 + w0)
        n1 = f1 * inv_rms * (1.0 + w1)
        if HAS_BIAS:
            n0 = n0 + tl.load(B_PTR + c0).to(tl.float32)
            n1 = n1 + tl.load(B_PTR + c1).to(tl.float32)
        tl.store(NORM_PTR + res_base + c0, n0)
        tl.store(NORM_PTR + res_base + c1, n1)


@triton.jit
def _hc_dflash_gconv_combine_norm_branch5120_kernel(
    RES_PTR,
    RAW_SUB_PTR,
    GCONV_DELTA_PTR,
    GCONV_BASE_PTR,
    INJ_PTR,
    W_PTR,
    B_PTR,
    COMB_PTR,
    NORM_PTR,
    X_STRIDE_ROW: tl.int64,
    DELTA_STRIDE_ROW: tl.int64,
    DELTA_STRIDE_TAP: tl.int64,
    BASE_STRIDE_TAP: tl.int64,
    CLIP_VAL: tl.float32,
    HAS_CLIP: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    SINGLE_STREAM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SWAP_GRID: tl.constexpr = False,
):
    """Per-(row, branch) 4096+1024 exact DFlashGroupedConv.finish + HC combine+norm."""
    row = tl.program_id(1).to(tl.int64) if SWAP_GRID else tl.program_id(0).to(tl.int64)
    branch = tl.program_id(0).to(tl.int64) if SWAP_GRID else tl.program_id(1).to(tl.int64)
    o0 = tl.arange(0, 4096).to(tl.int64)
    o1 = 4096 + tl.arange(0, 1024).to(tl.int64)
    g0 = o0 // GROUP_SIZE
    g1 = o1 // GROUP_SIZE
    pos_in_blk = row % BLOCK_SIZE

    acc0 = tl.zeros((4096,), dtype=tl.float32)
    acc1 = tl.zeros((1024,), dtype=tl.float32)
    for tap in tl.static_range(0, TAPS):
        src_row = row - tap
        valid = pos_in_blk >= tap
        safe_row = tl.where(valid, src_row, 0)
        x0 = tl.load(RAW_SUB_PTR + safe_row * X_STRIDE_ROW + o0, mask=valid, other=0.0).to(tl.float32)
        x1 = tl.load(RAW_SUB_PTR + safe_row * X_STRIDE_ROW + o1, mask=valid, other=0.0).to(tl.float32)
        b0 = tl.load(GCONV_BASE_PTR + tap * BASE_STRIDE_TAP + o0).to(tl.float32)
        b1 = tl.load(GCONV_BASE_PTR + tap * BASE_STRIDE_TAP + o1).to(tl.float32)
        d0 = tl.load(GCONV_DELTA_PTR + row * DELTA_STRIDE_ROW + tap * DELTA_STRIDE_TAP + g0).to(tl.float32)
        d1 = tl.load(GCONV_DELTA_PTR + row * DELTA_STRIDE_ROW + tap * DELTA_STRIDE_TAP + g1).to(tl.float32)
        acc0 += (b0 + d0) * x0
        acc1 += (b1 + d1) * x1

    if HAS_CLIP:
        acc0 = tl.minimum(tl.maximum(acc0, -CLIP_VAL), CLIP_VAL)
        acc1 = tl.minimum(tl.maximum(acc1, -CLIP_VAL), CLIP_VAL)
    sub0_f = acc0.to(RES_PTR.dtype.element_ty).to(tl.float32)
    sub1_f = acc1.to(RES_PTR.dtype.element_ty).to(tl.float32)

    c0 = branch * HIDDEN + o0
    c1 = branch * HIDDEN + o1
    res_base = row * (HC_COUNT * HIDDEN)
    in0 = (row * HIDDEN + o0) if SINGLE_STREAM else (res_base + c0)
    in1 = (row * HIDDEN + o1) if SINGLE_STREAM else (res_base + c1)
    inj = tl.load(INJ_PTR + row * HC_COUNT + branch).to(tl.float32)
    r0 = tl.load(RES_PTR + in0).to(tl.float32)
    r1 = tl.load(RES_PTR + in1).to(tl.float32)
    cb0 = (r0 + sub0_f * inj).to(RES_PTR.dtype.element_ty)
    cb1 = (r1 + sub1_f * inj).to(RES_PTR.dtype.element_ty)
    tl.store(COMB_PTR + res_base + c0, cb0)
    tl.store(COMB_PTR + res_base + c1, cb1)

    f0 = cb0.to(tl.float32)
    f1 = cb1.to(tl.float32)
    inv_rms = 1.0 / tl.sqrt((tl.sum(f0 * f0, axis=0) + tl.sum(f1 * f1, axis=0)) / HIDDEN + EPS)
    w0 = tl.load(W_PTR + c0).to(tl.float32)
    w1 = tl.load(W_PTR + c1).to(tl.float32)
    n0 = f0 * inv_rms * (1.0 + w0)
    n1 = f1 * inv_rms * (1.0 + w1)
    if HAS_BIAS:
        n0 = n0 + tl.load(B_PTR + c0).to(tl.float32)
        n1 = n1 + tl.load(B_PTR + c1).to(tl.float32)
    tl.store(NORM_PTR + res_base + c0, n0)
    tl.store(NORM_PTR + res_base + c1, n1)


@triton.jit(do_not_specialize=["M"])
def _hc_down_inject_sk_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """High-K Split-K GEMM writing FP32 partials [SPLIT_K, M, N] without splitKreduce."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    rk = k_start + tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = B_ptr + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = rm[:, None] < M
    n_mask = rn[None, :] < N
    for _ in range(k_start, k_end, BLOCK_K):
        km = rk < k_end
        a = tl.load(a_ptrs, mask=m_mask & km[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=km[:, None] & n_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        rk += BLOCK_K
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    c_ptrs = C_ptr + pid_k * (M * N) + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc, mask=m_mask & n_mask)


@triton.jit(do_not_specialize=["M"])
def _hc_mix_reduce_with_down_inject_sk_kernel(
    PART_PTR,
    LOGITS_PTR,
    NORMED_PTR,
    OUT_PTR,
    LATENT_PTR,
    INJ_PTR,
    M,
    SPLIT_K: tl.constexpr,
    N_OUT: tl.constexpr,
    LOWRANK: tl.constexpr,
    HC_COUNT: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused Split-K partial reduction + down/inject epilogue + weighted mix_reduce."""
    row = tl.program_id(0).to(tl.int64)
    col_blk = tl.program_id(1).to(tl.int64)
    if col_blk == 0:
        offs_n = tl.arange(0, BLOCK_N)
        offs_sk = tl.arange(0, BLOCK_SK)
        vals = tl.load(
            PART_PTR + offs_sk[:, None] * (M * N_OUT) + row * N_OUT + offs_n[None, :],
            mask=(offs_sk[:, None] < SPLIT_K) & (offs_n[None, :] < N_OUT),
            other=0.0,
        )
        raw = tl.sum(vals, axis=0)
        scaled = raw * (1.0 / HC_COUNT)
        sig_s = tl.sigmoid(scaled)
        tl.store(
            LATENT_PTR + row * LOWRANK + offs_n,
            (scaled * sig_s).to(LATENT_PTR.dtype.element_ty),
            mask=offs_n < LOWRANK,
        )
        tl.store(
            INJ_PTR + row * HC_COUNT + (offs_n - LOWRANK),
            (2.0 * sig_s).to(INJ_PTR.dtype.element_ty),
            mask=(offs_n >= LOWRANK) & (offs_n < N_OUT),
        )

    offs_h = col_blk * BLOCK_H + tl.arange(0, BLOCK_H)
    mh = offs_h < HIDDEN
    acc_h = tl.zeros((BLOCK_H,), dtype=tl.float32)
    base = row * (HC_COUNT * HIDDEN) + offs_h
    for b in tl.static_range(HC_COUNT):
        gw = tl.load(LOGITS_PTR + base + b * HIDDEN, mask=mh, other=0.0).to(tl.float32)
        xv = tl.load(NORMED_PTR + base + b * HIDDEN, mask=mh, other=0.0).to(tl.float32)
        acc_h += tl.sigmoid(gw) * xv
    tl.store(
        OUT_PTR + row * HIDDEN + offs_h,
        (acc_h * (1.0 / HC_COUNT)).to(OUT_PTR.dtype.element_ty),
        mask=mh,
    )


@triton.jit(do_not_specialize=["M"])
def _hc_up_mix_sk_1d_kernel(
    PART_PTR,
    UP_PTR,
    PREV_LATENT_PTR,
    NORMED_PTR,
    OUT_PTR,
    LATENT_PTR,
    INJ_PTR,
    M,
    SPLIT_K: tl.constexpr,
    N_OUT: tl.constexpr,
    LOWRANK: tl.constexpr,
    HC_COUNT: tl.constexpr,
    HIDDEN: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """1D-scheduled fused up-proj + sigmoid*normed mix_reduce + per-row Split-K epilogue."""
    pid = tl.program_id(0).to(tl.int64)
    m_i64 = M.to(tl.int64)
    if pid < m_i64:
        row = pid
        offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
        offs_sk = tl.arange(0, BLOCK_SK).to(tl.int64)
        n_mask = offs_n < N_OUT
        sk_mask = offs_sk < SPLIT_K
        vals = tl.load(
            PART_PTR + offs_sk[:, None] * (m_i64 * N_OUT) + row * N_OUT + offs_n[None, :],
            mask=sk_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        raw = tl.sum(vals, axis=0)
        scaled = raw * (1.0 / HC_COUNT)
        sig_s = tl.sigmoid(scaled)
        tl.store(
            LATENT_PTR + row * LOWRANK + offs_n,
            (scaled * sig_s).to(LATENT_PTR.dtype.element_ty),
            mask=offs_n < LOWRANK,
        )
        tl.store(
            INJ_PTR + row * HC_COUNT + (offs_n - LOWRANK),
            (2.0 * sig_s).to(INJ_PTR.dtype.element_ty),
            mask=(offs_n >= LOWRANK) & n_mask,
        )
        return

    tile = pid - m_i64
    pid_m = tile // NUM_H_BLOCKS
    pid_h = tile % NUM_H_BLOCKS
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    m_mask = offs_m < m_i64
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    mh = offs_h < HIDDEN

    acc_h = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    if BLOCK_K >= LOWRANK:
        offs_k = tl.arange(0, LOWRANK).to(tl.int64)
        pl = tl.load(
            PREV_LATENT_PTR + offs_m[:, None] * LOWRANK + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        for b in tl.static_range(HC_COUNT):
            bh_offs = b * HIDDEN + offs_h
            up_w = tl.load(
                UP_PTR + bh_offs[:, None] * LOWRANK + offs_k[None, :],
                mask=mh[:, None],
                other=0.0,
            )
            logits = tl.dot(pl, tl.trans(up_w))
            xv = tl.load(
                NORMED_PTR + offs_m[:, None] * (HC_COUNT * HIDDEN) + bh_offs[None, :],
                mask=m_mask[:, None] & mh[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_h += tl.sigmoid(logits) * xv
    else:
        for b in tl.static_range(HC_COUNT):
            bh_offs = b * HIDDEN + offs_h
            logits = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
            for ks in tl.static_range(0, LOWRANK, BLOCK_K):
                offs_k = (ks + tl.arange(0, BLOCK_K)).to(tl.int64)
                pl = tl.load(
                    PREV_LATENT_PTR + offs_m[:, None] * LOWRANK + offs_k[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                )
                up_w = tl.load(
                    UP_PTR + bh_offs[:, None] * LOWRANK + offs_k[None, :],
                    mask=mh[:, None],
                    other=0.0,
                )
                logits = tl.dot(pl, tl.trans(up_w), logits)
            xv = tl.load(
                NORMED_PTR + offs_m[:, None] * (HC_COUNT * HIDDEN) + bh_offs[None, :],
                mask=m_mask[:, None] & mh[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_h += tl.sigmoid(logits) * xv

    tl.store(
        OUT_PTR + offs_m[:, None] * HIDDEN + offs_h[None, :],
        (acc_h * (1.0 / HC_COUNT)).to(OUT_PTR.dtype.element_ty),
        mask=m_mask[:, None] & mh[None, :],
    )


@triton.jit
def _hc_mix_reduce_with_down_inject_kernel(
    LOGITS_PTR,
    NORMED_PTR,
    OUT_PTR,
    RAW_PTR,
    LATENT_PTR,
    INJ_PTR,
    RAW_STRIDE: tl.constexpr,
    LOWRANK: tl.constexpr,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    HAS_LATENT: tl.constexpr,
    HAS_INJ: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SWAP_GRID: tl.constexpr = False,
):
    """Fused hc_mix_reduce + (pid_col == 0) down/inject epilogue when prev_mix and not is_head."""
    pid_col = tl.program_id(0) if SWAP_GRID else tl.program_id(1)
    row = tl.program_id(1).to(tl.int64) if SWAP_GRID else tl.program_id(0).to(tl.int64)
    col = pid_col.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    row_base = row * (HC_COUNT * HIDDEN)
    for branch in tl.static_range(HC_COUNT):
        offset = row_base + branch * HIDDEN + col
        logits = tl.load(LOGITS_PTR + offset, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(NORMED_PTR + offset, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sigmoid(logits) * normed
    out = acc * (1.0 / HC_COUNT)
    tl.store(OUT_PTR + row * HIDDEN + col, out, mask=mask)

    if pid_col == 0:
        inv_hc: tl.constexpr = 1.0 / HC_COUNT
        row_raw = RAW_PTR + row * RAW_STRIDE
        if HAS_LATENT:
            l_offs = tl.arange(0, BLOCK_L)
            l_mask = l_offs < LOWRANK
            raw_l = tl.load(row_raw + l_offs, mask=l_mask, other=0.0).to(tl.float32)
            scaled_l = raw_l * inv_hc
            latent = scaled_l * tl.sigmoid(scaled_l)
            tl.store(LATENT_PTR + row * LOWRANK + l_offs, latent, mask=l_mask)
        if HAS_INJ:
            h_offs = tl.arange(0, BLOCK_H)
            h_mask = h_offs < HC_COUNT
            h_base = LOWRANK if HAS_LATENT else 0
            raw_h = tl.load(row_raw + h_base + h_offs, mask=h_mask, other=0.0).to(tl.float32)
            inj = 2.0 * tl.sigmoid(raw_h * inv_hc)
            tl.store(INJ_PTR + row * HC_COUNT + h_offs, inj, mask=h_mask)


@triton.jit(do_not_specialize=["ROWS"])
def _hc_down_silu_epilogue_kernel(
    RAW_PTR,
    LATENT_PTR,
    ROWS,
    LOWRANK: tl.constexpr,
    HC_COUNT: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Single-launch SiLU(raw[..., :LOWRANK] / HC_COUNT) for emits_latent-only final mixer."""
    r_offs = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offs < ROWS
    l_offs = tl.arange(0, BLOCK_L)
    l_mask = r_mask[:, None] & (l_offs[None, :] < LOWRANK)
    raw_l = tl.load(
        RAW_PTR + r_offs[:, None] * RAW_STRIDE + l_offs[None, :],
        mask=l_mask,
        other=0.0,
    ).to(tl.float32)
    scaled = raw_l * (1.0 / HC_COUNT)
    latent = scaled * tl.sigmoid(scaled)
    tl.store(
        LATENT_PTR + r_offs[:, None] * LOWRANK + l_offs[None, :],
        latent,
        mask=l_mask,
    )


@triton.jit
def _hc_mix_reduce_kernel(
    LOGITS_PTR,
    NORMED_PTR,
    OUT_PTR,
    HIDDEN: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    row_base = row * (HC_COUNT * HIDDEN)
    for branch in tl.static_range(HC_COUNT):
        offset = row_base + branch * HIDDEN + col
        logits = tl.load(LOGITS_PTR + offset, mask=mask, other=0.0).to(tl.float32)
        normed = tl.load(NORMED_PTR + offset, mask=mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(logits)
        acc += gate * normed
    out = acc * (1.0 / HC_COUNT)
    tl.store(OUT_PTR + row * HIDDEN + col, out, mask=mask)


@triton.jit
def _hc_head_mix_kernel(
    LOGITS_PTR,
    NORMED_PTR,
    OUT_PTR,
    HIDDEN: tl.constexpr,
    HC_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HIDDEN
    logits = tl.load(LOGITS_PTR + row * HIDDEN + col, mask=mask, other=0.0).to(tl.float32)
    normed0 = tl.load(NORMED_PTR + row * HC_WIDTH + col, mask=mask, other=0.0).to(tl.float32)
    out = tl.sigmoid(logits) * normed0
    tl.store(OUT_PTR + row * HIDDEN + col, out, mask=mask)


@triton.jit(do_not_specialize=["ROWS"])
def _hc_down_inject_epilogue_kernel(
    RAW_PTR,
    LATENT_PTR,
    INJ_PTR,
    ROWS,
    LOWRANK: tl.constexpr,
    HC_COUNT: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r_offs = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offs < ROWS
    inv_hc: tl.constexpr = 1.0 / HC_COUNT

    l_offs = tl.arange(0, BLOCK_L)
    l_mask = r_mask[:, None] & (l_offs[None, :] < LOWRANK)
    raw_l = tl.load(
        RAW_PTR + r_offs[:, None] * RAW_STRIDE + l_offs[None, :],
        mask=l_mask,
        other=0.0,
    ).to(tl.float32)
    scaled_l = raw_l * inv_hc
    latent = scaled_l * tl.sigmoid(scaled_l)
    tl.store(
        LATENT_PTR + r_offs[:, None] * LOWRANK + l_offs[None, :],
        latent,
        mask=l_mask,
    )

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = r_mask[:, None] & (h_offs[None, :] < HC_COUNT)
    raw_h = tl.load(
        RAW_PTR + r_offs[:, None] * RAW_STRIDE + LOWRANK + h_offs[None, :],
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)
    inj = 2.0 * tl.sigmoid(raw_h * inv_hc)
    tl.store(
        INJ_PTR + r_offs[:, None] * HC_COUNT + h_offs[None, :],
        inj,
        mask=h_mask,
    )


@triton.jit(do_not_specialize=["ROWS"])
def _hc_inject_epilogue_kernel(
    RAW_PTR,
    INJ_PTR,
    ROWS,
    HC_COUNT: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r_offs = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offs < ROWS
    inv_hc: tl.constexpr = 1.0 / HC_COUNT
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = r_mask[:, None] & (h_offs[None, :] < HC_COUNT)
    raw_h = tl.load(
        RAW_PTR + r_offs[:, None] * RAW_STRIDE + h_offs[None, :],
        mask=h_mask,
        other=0.0,
    ).to(tl.float32)
    inj = 2.0 * tl.sigmoid(raw_h * inv_hc)
    tl.store(
        INJ_PTR + r_offs[:, None] * HC_COUNT + h_offs[None, :],
        inj,
        mask=h_mask,
    )


# ---------------------------------------------------------------------------
# Functional Triton & Reference Wrappers
# ---------------------------------------------------------------------------


def grouped_gemma_rmsnorm_bias_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    hc_count: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact FP32 PyTorch reference for per-branch GemmaRMSNorm (1 + w) + bias."""
    w_width = int(weight.shape[0])
    x_width = int(x.shape[-1])
    if hc_count > 1 and x_width * hc_count == w_width:
        x = x.unsqueeze(-2).expand(*x.shape[:-1], hc_count, x_width).flatten(-2)
        width = w_width
        hidden_size = x_width
    else:
        width = x_width
        if width % hc_count != 0:
            raise ValueError(f"Last dim {width} is not divisible by hc_count={hc_count}")
        hidden_size = width // hc_count
    streams = x.unflatten(-1, (hc_count, hidden_size)).float()
    normed = streams * torch.rsqrt(streams.square().mean(dim=-1, keepdim=True) + eps)
    out = normed.flatten(-2) * (1.0 + weight.float())
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def _launch_grouped_gemma_rmsnorm_bias(
    x_c: torch.Tensor,
    w_c: torch.Tensor,
    b_c: torch.Tensor,
    out: torch.Tensor,
    width: int,
    group_size: int,
    x_width: int,
    hc_count: int,
    single_stream: bool,
    eps: float,
    has_bias: bool,
    rows: int,
) -> None:
    """Dispatch grouped GemmaRMSNorm+bias, using L2-friendly (HC, row) 4096+1024 split for large H=5120."""
    if group_size == 5120 and hc_count == 4 and rows >= 128:
        num_warps = 4 if rows >= 256 else 8
        ckey = (
            x_c.device.index, 36, x_c.dtype, w_c.dtype, b_c.dtype, out.dtype,
            single_stream, float(eps), has_bias, num_warps,
        )
        triton_fast_launch(
            _HC_TRITON_CACHE, ckey, _grouped_gemma_rmsnorm_bias_split5120_kernel,
            hc_count, rows, 1, x_c.data_ptr() | w_c.data_ptr() | b_c.data_ptr() | out.data_ptr(),
            (x_c, w_c, b_c, out),
            {"HIDDEN": group_size, "HC_COUNT": hc_count, "SINGLE_STREAM": single_stream, "EPS": eps, "HAS_BIAS": has_bias},
            num_warps=num_warps,
        )
    elif group_size <= 8192:
        block = max(128, triton.next_power_of_2(group_size))
        num_warps = 8 if block >= 4096 else 4
        ckey = (x_c.device.index, 0, x_c.dtype, w_c.dtype, b_c.dtype, out.dtype, width, group_size, x_width, single_stream, float(eps), has_bias, block, num_warps)
        triton_fast_launch(
            _HC_TRITON_CACHE, ckey, _grouped_gemma_rmsnorm_bias_kernel,
            rows, hc_count, 1, x_c.data_ptr() | w_c.data_ptr() | b_c.data_ptr() | out.data_ptr(),
            (x_c, w_c, b_c, out),
            {"WIDTH": width, "GROUP_SIZE": group_size, "X_STRIDE": x_width, "SINGLE_STREAM": single_stream, "EPS": eps, "HAS_BIAS": has_bias, "BLOCK": block},
            num_warps=num_warps,
        )
    else:
        _grouped_gemma_rmsnorm_bias_chunked_kernel[(rows, hc_count)](
            x_c,
            w_c,
            b_c,
            out,
            WIDTH=width,
            GROUP_SIZE=group_size,
            X_STRIDE=x_width,
            SINGLE_STREAM=single_stream,
            EPS=eps,
            HAS_BIAS=has_bias,
            BLOCK=4096,
            num_warps=8,
        )


def grouped_gemma_rmsnorm_bias_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    hc_count: int,
    eps: float = 1e-6,
    *,
    normed_tag: str = "normed_ext",
    ws: Optional[HCStaticWorkspace] = None,
) -> torch.Tensor:
    """Fused Triton per-branch GemmaRMSNorm (1.0 + weight) + optional affine bias."""
    if not x.is_cuda:
        return grouped_gemma_rmsnorm_bias_ref(x, weight, bias, hc_count, eps)
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else w_c
    width = int(w_c.shape[0])
    x_width = int(x_c.shape[-1])
    if width % hc_count != 0:
        raise ValueError(f"Weight dim {width} is not divisible by hc_count={hc_count}")
    group_size = width // hc_count
    if x_width == width:
        single_stream = False
    elif hc_count > 1 and x_width == group_size:
        single_stream = True
    else:
        raise ValueError(f"Last dim {x_width} is not divisible by hc_count={hc_count}")
    rows = x_c.numel() // x_width
    out_shape = (*x_c.shape[:-1], width) if single_stream else tuple(x_c.shape)
    out = _hc_static_empty(out_shape, x_c.dtype, x_c.device, normed_tag, ws=ws)
    if rows == 0:
        return out
    has_bias = bias is not None
    _launch_grouped_gemma_rmsnorm_bias(
        x_c, w_c, b_c, out, width, group_size, x_width, hc_count, single_stream, float(eps), has_bias, rows,
    )
    return out


def hc_combine_ref(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_w: torch.Tensor,
) -> torch.Tensor:
    """Exact FP32 PyTorch reference for injecting block_output into all HC streams."""
    hc_count = inject_w.shape[-1]
    hidden_size = block_output.shape[-1]
    if hc_count > 1 and int(residual.shape[-1]) == hidden_size:
        res_f = residual.unsqueeze(-2).expand(*residual.shape[:-1], hc_count, hidden_size).float()
    else:
        res_f = residual.unflatten(-1, (hc_count, hidden_size)).float()
    out_f = block_output.unsqueeze(-2).float()
    inj_f = inject_w.unsqueeze(-1).float()
    return (res_f + out_f * inj_f).flatten(-2).to(residual.dtype)


def hc_combine_triton(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_w: torch.Tensor,
    *,
    ws: Optional[HCStaticWorkspace] = None,
) -> torch.Tensor:
    """Fused Triton injection of block_output into hc_count residual streams."""
    if not residual.is_cuda:
        return hc_combine_ref(block_output, residual, inject_w)
    res_c = residual.contiguous()
    out_c = block_output.contiguous()
    inj_c = inject_w.contiguous()
    hc_count = int(inj_c.shape[-1])
    hidden_size = int(out_c.shape[-1])
    res_width = int(res_c.shape[-1])
    width = hc_count * hidden_size
    if res_width == width:
        single_stream = False
    elif hc_count > 1 and res_width == hidden_size:
        single_stream = True
    else:
        raise ValueError(
            f"residual width {res_width} != hc_count ({hc_count}) * hidden_size ({hidden_size})"
        )
    rows = res_c.numel() // res_width
    dst_shape = (*res_c.shape[:-1], width) if single_stream else tuple(res_c.shape)
    dst = _hc_static_empty(dst_shape, res_c.dtype, res_c.device, "combined", ws=ws)
    if rows == 0:
        return dst
    block = 512 if hidden_size >= 512 else max(128, triton.next_power_of_2(hidden_size))
    ckey = (res_c.device.index, 1, res_c.dtype, out_c.dtype, inj_c.dtype, hidden_size, hc_count, single_stream, block)
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_combine_kernel,
        rows, triton.cdiv(hidden_size, block), 1, res_c.data_ptr() | out_c.data_ptr() | inj_c.data_ptr() | dst.data_ptr(),
        (res_c, out_c, inj_c, dst),
        {"HIDDEN": hidden_size, "HC_COUNT": hc_count, "SINGLE_STREAM": single_stream, "BLOCK": block},
        num_warps=4,
    )
    return dst


def _launch_hc_combine_and_norm_branch5120(
    res_c: torch.Tensor,
    sub_c: torch.Tensor,
    inj_c: torch.Tensor,
    w_c: torch.Tensor,
    b_c: torch.Tensor,
    combined: torch.Tensor,
    normed: torch.Tensor,
    rows: int,
    hidden_size: int,
    hc_count: int,
    single_stream: bool,
    eps: float,
    has_bias: bool,
) -> None:
    """Launch L2-friendly (HC, row) 4096+1024 split combine+norm across all M regimes."""
    num_warps = 4 if rows >= 128 else 8
    ptr_or = (
        res_c.data_ptr() | sub_c.data_ptr() | inj_c.data_ptr() | w_c.data_ptr()
        | b_c.data_ptr() | combined.data_ptr() | normed.data_ptr()
    )
    ckey = (
        res_c.device.index, 34, res_c.dtype, sub_c.dtype, inj_c.dtype,
        w_c.dtype, b_c.dtype, single_stream, float(eps), has_bias, num_warps,
    )
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_combine_and_norm_branch5120_kernel,
        hc_count, rows, 1, ptr_or, (res_c, sub_c, inj_c, w_c, b_c, combined, normed),
        {
            "HIDDEN": hidden_size, "HC_COUNT": hc_count, "SINGLE_STREAM": single_stream,
            "EPS": float(eps), "HAS_BIAS": has_bias,
        },
        num_warps=num_warps,
    )


def hc_combine_and_norm_triton(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    prev_inject_w: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: Optional[torch.Tensor],
    eps: float = 1e-6,
    *,
    normed_tag: str = "normed_ext",
    ws: Optional[HCStaticWorkspace] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-pass fused combine + per-branch GemmaRMSNorm+bias."""
    hc_count = int(prev_inject_w.shape[-1])
    hidden_size = int(block_output.shape[-1])
    if not residual.is_cuda or hidden_size > 8192:
        combined = hc_combine_triton(block_output, residual, prev_inject_w, ws=ws)
        normed = grouped_gemma_rmsnorm_bias_triton(
            combined, norm_weight, norm_bias, hc_count, eps, normed_tag=normed_tag, ws=ws
        )
        return combined, normed

    res_c = residual.contiguous()
    sub_c = block_output.contiguous()
    inj_c = prev_inject_w.contiguous()
    w_c = norm_weight.contiguous()
    b_c = norm_bias.contiguous() if norm_bias is not None else w_c
    res_width = int(res_c.shape[-1])
    width = hc_count * hidden_size
    if res_width == width:
        single_stream = False
    elif hc_count > 1 and res_width == hidden_size:
        single_stream = True
    else:
        raise ValueError(
            f"residual width {res_width} != hc_count ({hc_count}) * hidden_size ({hidden_size})"
        )
    rows = res_c.numel() // res_width
    out_shape = (*res_c.shape[:-1], width)
    combined = _hc_static_empty(out_shape, res_c.dtype, res_c.device, "combined", ws=ws)
    normed = _hc_static_empty(out_shape, res_c.dtype, res_c.device, normed_tag, ws=(None if normed_tag == "normed" else ws))
    if rows == 0:
        return combined, normed

    has_b = norm_bias is not None
    if hidden_size == 5120 and hc_count == 4:
        _launch_hc_combine_and_norm_branch5120(
            res_c, sub_c, inj_c, w_c, b_c, combined, normed,
            rows, hidden_size, hc_count, single_stream, float(eps), has_b,
        )
        return combined, normed

    ptr_or = res_c.data_ptr() | sub_c.data_ptr() | inj_c.data_ptr() | w_c.data_ptr() | b_c.data_ptr() | combined.data_ptr() | normed.data_ptr()

    block = max(128, triton.next_power_of_2(hidden_size))
    num_warps = 4
    ckey = (res_c.device.index, 3, res_c.dtype, sub_c.dtype, inj_c.dtype, w_c.dtype, b_c.dtype, hidden_size, hc_count, single_stream, float(eps), has_b, block)
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_combine_and_norm_kernel,
        rows, hc_count, 1, ptr_or, (res_c, sub_c, inj_c, w_c, b_c, combined, normed),
        {"HIDDEN": hidden_size, "HC_COUNT": hc_count, "SINGLE_STREAM": single_stream, "EPS": eps, "HAS_BIAS": has_b, "BLOCK": block},
        num_warps=num_warps,
    )
    return combined, normed


def hc_sconv_decode_opaque_combine_norm_triton(
    raw_sub: torch.Tensor,
    residual: torch.Tensor,
    prev_inject_w: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: Optional[torch.Tensor],
    conv_weight: torch.Tensor,
    pool_2d: torch.Tensor,
    block_table: torch.Tensor,
    seq_before: torch.Tensor,
    *,
    page_size: int,
    eps: float = 1e-6,
    conv_residual: bool = False,
    conv_use_silu: bool = False,
    clamp_val: Optional[float] = None,
    normed_tag: str = "normed_ext",
    ws: Optional[HCStaticWorkspace] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-launch T=1 K=4 paged opaque ShortConv(+clamp+pool roll) + HC combine+norm."""
    hc_count = int(prev_inject_w.shape[-1])
    hidden_size = int(raw_sub.shape[-1])
    res_c = residual if residual.is_contiguous() else residual.contiguous()
    sub_c = raw_sub if raw_sub.stride(-1) == 1 else raw_sub.contiguous()
    inj_c = prev_inject_w if prev_inject_w.is_contiguous() else prev_inject_w.contiguous()
    w_c = norm_weight if norm_weight.is_contiguous() else norm_weight.contiguous()
    has_bias = norm_bias is not None
    b_c = (norm_bias if norm_bias.is_contiguous() else norm_bias.contiguous()) if has_bias else w_c
    cw_c = conv_weight if conv_weight.is_contiguous() else conv_weight.contiguous()

    res_width = int(res_c.shape[-1])
    width = hc_count * hidden_size
    single_stream = hc_count > 1 and res_width == hidden_size
    rows = res_c.numel() // res_width
    out_shape = (*res_c.shape[:-1], width)
    combined = _hc_static_empty(out_shape, res_c.dtype, res_c.device, "combined", ws=ws)
    normed = _hc_static_empty(out_shape, res_c.dtype, res_c.device, normed_tag, ws=(None if normed_tag == "normed" else ws))
    if rows == 0:
        return combined, normed

    sxb = int(sub_c.reshape(rows, hidden_size).stride(0))
    spb = int(pool_2d.stride(0))
    sbt0 = int(block_table.stride(0))
    sbt1 = int(block_table.stride(1))
    has_c = clamp_val is not None
    cval = float(clamp_val) if has_c else 0.0
    rt_args = (
        res_c, sub_c, inj_c, w_c, b_c, cw_c, pool_2d, block_table, seq_before, combined, normed,
        sxb, spb, sbt0, sbt1, int(page_size), int(block_table.shape[1]), int(pool_2d.shape[0]),
    )
    ptr_or = (
        res_c.data_ptr() | sub_c.data_ptr() | inj_c.data_ptr() | w_c.data_ptr() | b_c.data_ptr()
        | cw_c.data_ptr() | pool_2d.data_ptr() | block_table.data_ptr() | seq_before.data_ptr()
        | combined.data_ptr() | normed.data_ptr()
    )
    if hidden_size == 5120 and hc_count == 4:
        if rows <= 512 and sub_c.dtype == res_c.dtype:
            sc_bc = 256
            sc_nw = 8 if rows <= 32 else 4
            sconv_sub = _hc_alloc_from_pool(
                _HC_INTERNAL_NORMED_POOL, (rows, hidden_size), sub_c.dtype, sub_c.device, "sconv_sub", 1
            )
            sc_rt = (
                sub_c, cw_c, pool_2d, sconv_sub, block_table, seq_before,
                sxb, spb, sbt0, sbt1, int(page_size), int(block_table.shape[1]), int(pool_2d.shape[0]), 0,
            )
            sc_ptr_or = (
                sub_c.data_ptr() | cw_c.data_ptr() | pool_2d.data_ptr() | sconv_sub.data_ptr()
                | block_table.data_ptr() | seq_before.data_ptr()
            )
            sc_ckey = (
                res_c.device.index, 35, sub_c.dtype, cw_c.dtype, pool_2d.dtype, sconv_sub.dtype,
                block_table.dtype, seq_before.dtype,
                _triton_int_spec(sxb), _triton_int_spec(spb), _triton_int_spec(sbt0), _triton_int_spec(sbt1),
                hidden_size, conv_residual, conv_use_silu, False, has_c, cval, sc_bc, sc_nw,
            )
            triton_fast_launch(
                _HC_TRITON_CACHE, sc_ckey, _causal_dwconv1d_decode_opaque_k4_kernel,
                rows, triton.cdiv(hidden_size, sc_bc), 1, sc_ptr_or, sc_rt,
                {
                    "C": hidden_size, "RESIDUAL": conv_residual, "USE_SILU": conv_use_silu,
                    "ROUND_PRE_ACT": False, "HAS_CLAMP": has_c, "CLAMP_VAL": cval, "BLOCK_C": sc_bc,
                },
                num_warps=sc_nw,
            )
            _launch_hc_combine_and_norm_branch5120(
                res_c, sconv_sub, inj_c, w_c, b_c, combined, normed,
                rows, hidden_size, hc_count, single_stream, float(eps), has_bias,
            )
            return combined, normed
        ckey = (
            res_c.device.index, 20, res_c.dtype, sub_c.dtype, inj_c.dtype, w_c.dtype, b_c.dtype, cw_c.dtype, pool_2d.dtype,
            block_table.dtype, seq_before.dtype,
            _triton_int_spec(sxb), _triton_int_spec(spb), _triton_int_spec(sbt0), _triton_int_spec(sbt1),
            single_stream, float(eps), has_bias, conv_residual, conv_use_silu, has_c, cval,
        )
        triton_fast_launch(
            _HC_TRITON_CACHE, ckey, _hc_sconv_decode_opaque_combine_norm_split5120_kernel,
            rows, 1, 1, ptr_or, rt_args,
            {
                "HIDDEN": hidden_size, "HC_COUNT": hc_count, "SINGLE_STREAM": single_stream,
                "EPS": float(eps), "HAS_BIAS": has_bias, "CONV_RESIDUAL": conv_residual,
                "CONV_USE_SILU": conv_use_silu, "HAS_CLAMP": has_c, "CLAMP_VAL": cval,
            },
            num_warps=32,
        )
        return combined, normed

    block = max(128, triton.next_power_of_2(hidden_size))
    ckey = (
        res_c.device.index, 21, res_c.dtype, sub_c.dtype, inj_c.dtype, w_c.dtype, b_c.dtype, cw_c.dtype, pool_2d.dtype,
        block_table.dtype, seq_before.dtype,
        _triton_int_spec(sxb), _triton_int_spec(spb), _triton_int_spec(sbt0), _triton_int_spec(sbt1),
        hidden_size, hc_count, single_stream, float(eps), has_bias, conv_residual, conv_use_silu, has_c, cval, block,
    )
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_sconv_decode_opaque_combine_norm_kernel,
        rows, 1, 1, ptr_or, rt_args,
        {
            "HIDDEN": hidden_size, "HC_COUNT": hc_count, "SINGLE_STREAM": single_stream,
            "EPS": float(eps), "HAS_BIAS": has_bias, "CONV_RESIDUAL": conv_residual,
            "CONV_USE_SILU": conv_use_silu, "HAS_CLAMP": has_c, "CLAMP_VAL": cval, "BLOCK": block,
        },
        num_warps=4,
    )
    return combined, normed


def hc_mix_reduce_triton(
    gate_logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
    *,
    ws: Optional[HCStaticWorkspace] = None,
) -> torch.Tensor:
    """Fused (sigmoid(gate_logits) * normed).mean(dim=-2) across HC branches."""
    if not gate_logits.is_cuda:
        hidden_size = gate_logits.shape[-1] // hc_count
        gate = torch.sigmoid(gate_logits.float()).unflatten(-1, (hc_count, hidden_size))
        n_f = normed.float().unflatten(-1, (hc_count, hidden_size))
        return (gate * n_f).mean(dim=-2).to(normed.dtype)

    logits_c = gate_logits.contiguous()
    normed_c = normed.contiguous()
    width = int(logits_c.shape[-1])
    hidden_size = width // hc_count
    rows = logits_c.numel() // width
    out = _hc_static_empty(
        (*logits_c.shape[:-1], hidden_size),
        normed_c.dtype,
        logits_c.device,
        "out",
        ws=ws,
    )
    if rows == 0:
        return out
    block = 512 if hidden_size >= 512 else max(128, triton.next_power_of_2(hidden_size))
    ckey = (logits_c.device.index, 4, logits_c.dtype, normed_c.dtype, hidden_size, hc_count, block)
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_mix_reduce_kernel,
        rows, triton.cdiv(hidden_size, block), 1, logits_c.data_ptr() | normed_c.data_ptr() | out.data_ptr(),
        (logits_c, normed_c, out),
        {"HIDDEN": hidden_size, "HC_COUNT": hc_count, "BLOCK": block},
        num_warps=4,
    )
    return out


def hc_head_mix_triton(
    gate_logits: torch.Tensor,
    normed: torch.Tensor,
    *,
    ws: Optional[HCStaticWorkspace] = None,
) -> torch.Tensor:
    """Fused Layer-0 head mix: sigmoid(gate_logits) * normed[..., :hidden_size]."""
    hidden_size = int(gate_logits.shape[-1])
    if not gate_logits.is_cuda:
        return (
            torch.sigmoid(gate_logits.float()) * normed[..., :hidden_size].float()
        ).to(normed.dtype)

    logits_c = gate_logits.contiguous()
    normed_c = normed.contiguous()
    hc_width = int(normed_c.shape[-1])
    rows = logits_c.numel() // hidden_size
    out = _hc_static_empty(
        tuple(logits_c.shape),
        logits_c.dtype,
        logits_c.device,
        "out",
        ws=ws,
    )
    if rows == 0:
        return out
    block = 512 if hidden_size >= 512 else max(128, triton.next_power_of_2(hidden_size))
    ckey = (logits_c.device.index, 5, logits_c.dtype, normed_c.dtype, hidden_size, hc_width, block)
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_head_mix_kernel,
        rows, triton.cdiv(hidden_size, block), 1, logits_c.data_ptr() | normed_c.data_ptr() | out.data_ptr(),
        (logits_c, normed_c, out),
        {"HIDDEN": hidden_size, "HC_WIDTH": hc_width, "BLOCK": block},
        num_warps=4,
    )
    return out


def hc_down_inject_epilogue_triton(
    down_inject_raw: torch.Tensor,
    hc_lowrank: int,
    hc_count: int,
    *,
    ws: Optional[HCStaticWorkspace] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fused [..., lowrank + hc_count] GEMM output into (latent, inject_w)."""
    prefix = down_inject_raw.shape[:-1]
    if not down_inject_raw.is_cuda:
        raw_f = down_inject_raw.float()
        latent = F.silu(raw_f[..., :hc_lowrank] / hc_count).to(down_inject_raw.dtype)
        inject_w = (2.0 * torch.sigmoid(raw_f[..., hc_lowrank : hc_lowrank + hc_count] / hc_count)).to(
            down_inject_raw.dtype
        )
        return latent, inject_w

    raw_c = down_inject_raw.contiguous()
    raw_stride = int(raw_c.shape[-1])
    rows = raw_c.numel() // raw_stride
    latent = _hc_static_empty((*prefix, hc_lowrank), raw_c.dtype, raw_c.device, "latent", ws=ws)
    inject_w = _hc_static_empty((*prefix, hc_count), raw_c.dtype, raw_c.device, "inj", ws=ws)
    if rows == 0:
        return latent, inject_w

    block_r = 1 if rows <= 8 else (8 if rows <= 128 else 16)
    block_l = max(16, triton.next_power_of_2(hc_lowrank))
    block_h = max(16, triton.next_power_of_2(hc_count))
    ckey = (raw_c.device.index, 6, raw_c.dtype, hc_lowrank, hc_count, raw_stride, block_r, block_l, block_h)
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_down_inject_epilogue_kernel,
        triton.cdiv(rows, block_r), 1, 1, raw_c.data_ptr() | latent.data_ptr() | inject_w.data_ptr(),
        (raw_c, latent, inject_w, rows),
        {"LOWRANK": hc_lowrank, "HC_COUNT": hc_count, "RAW_STRIDE": raw_stride, "BLOCK_R": block_r, "BLOCK_L": block_l, "BLOCK_H": block_h},
        num_warps=4,
    )
    return latent, inject_w


def hc_inject_epilogue_triton(
    inject_raw: torch.Tensor,
    hc_count: int,
    *,
    ws: Optional[HCStaticWorkspace] = None,
) -> torch.Tensor:
    """Compute 2.0 * sigmoid(raw[..., :hc_count] / hc_count) directly from padded GEMM output."""
    prefix = inject_raw.shape[:-1]
    if not inject_raw.is_cuda:
        return (2.0 * torch.sigmoid(inject_raw[..., :hc_count].float() / hc_count)).to(
            inject_raw.dtype
        )

    raw_c = inject_raw.contiguous()
    raw_stride = int(raw_c.shape[-1])
    rows = raw_c.numel() // raw_stride
    inject_w = _hc_static_empty((*prefix, hc_count), raw_c.dtype, raw_c.device, "inj", ws=ws)
    if rows == 0:
        return inject_w

    block_r = 1 if rows <= 8 else (8 if rows <= 128 else 16)
    block_h = max(16, triton.next_power_of_2(hc_count))
    ckey = (raw_c.device.index, 7, raw_c.dtype, hc_count, raw_stride, block_r, block_h)
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_inject_epilogue_kernel,
        triton.cdiv(rows, block_r), 1, 1, raw_c.data_ptr() | inject_w.data_ptr(),
        (raw_c, inject_w, rows),
        {"HC_COUNT": hc_count, "RAW_STRIDE": raw_stride, "BLOCK_R": block_r, "BLOCK_H": block_h},
        num_warps=4,
    )
    return inject_w


def hc_dflash_gconv_combine_norm_triton(
    raw_sub: torch.Tensor,
    gconv_delta: torch.Tensor,
    gconv_base: torch.Tensor,
    gconv_block_size: int,
    gconv_group_size: int,
    gconv_taps: int,
    residual: torch.Tensor,
    prev_inject_w: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: Optional[torch.Tensor],
    eps: float = 1e-6,
    *,
    clip_value: Optional[float] = None,
    normed_tag: str = "normed_ext",
    ws: Optional[HCStaticWorkspace] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-pass DFlashGroupedConv.finish + HC combine + GemmaRMSNorm+bias."""
    hc_count = int(prev_inject_w.shape[-1])
    hidden_size = int(raw_sub.shape[-1])
    sub_c = raw_sub if raw_sub.stride(-1) == 1 else raw_sub.contiguous()
    delta_c = gconv_delta if gconv_delta.stride(-1) == 1 else gconv_delta.contiguous()
    base_c = gconv_base if gconv_base.is_contiguous() else gconv_base.contiguous()
    res_c = residual if residual.is_contiguous() else residual.contiguous()
    inj_c = prev_inject_w if prev_inject_w.is_contiguous() else prev_inject_w.contiguous()
    w_c = norm_weight if norm_weight.is_contiguous() else norm_weight.contiguous()
    has_bias = norm_bias is not None
    b_c = (norm_bias if norm_bias.is_contiguous() else norm_bias.contiguous()) if has_bias else w_c

    res_width = int(res_c.shape[-1])
    width = hc_count * hidden_size
    if res_width == width:
        single_stream = False
    elif hc_count > 1 and res_width == hidden_size:
        single_stream = True
    else:
        raise ValueError(
            f"residual width {res_width} != hc_count ({hc_count}) * hidden_size ({hidden_size})"
        )
    rows = res_c.numel() // res_width
    out_shape = (*res_c.shape[:-1], width)
    combined = _hc_static_empty(out_shape, res_c.dtype, res_c.device, "combined", ws=ws)
    normed = _hc_static_empty(out_shape, res_c.dtype, res_c.device, normed_tag, ws=(None if normed_tag == "normed" else ws))
    if rows == 0:
        return combined, normed

    if not res_c.is_cuda or hidden_size > 8192:
        raise RuntimeError("hc_dflash_gconv_combine_norm_triton requires CUDA and hidden_size <= 8192")

    sub_2d = sub_c.reshape(rows, hidden_size)
    x_s0 = int(sub_2d.stride(0))
    delta_3d = delta_c.reshape(rows, int(gconv_taps), int(delta_c.shape[-1]))
    d_s0 = int(delta_3d.stride(0))
    d_s1 = int(delta_3d.stride(1))
    b_s0 = int(base_c.stride(0))
    has_clip = clip_value is not None
    cval = float(clip_value) if has_clip else 0.0
    rt_args = (
        res_c, sub_2d, delta_3d, base_c, inj_c, w_c, b_c, combined, normed,
        x_s0, d_s0, d_s1, b_s0, cval,
    )
    ptr_or = (
        res_c.data_ptr() | sub_2d.data_ptr() | delta_3d.data_ptr() | base_c.data_ptr()
        | inj_c.data_ptr() | w_c.data_ptr() | b_c.data_ptr() | combined.data_ptr() | normed.data_ptr()
    )
    if (
        hidden_size == 5120
        and hc_count == 4
        and (4096 % int(gconv_group_size)) == 0
    ):
        swap_grid = rows >= 128
        num_warps = 4 if swap_grid else 8
        grid0, grid1 = (hc_count, rows) if swap_grid else (rows, hc_count)
        ckey = (
            res_c.device.index, 37, res_c.dtype, sub_2d.dtype, delta_3d.dtype, base_c.dtype,
            inj_c.dtype, w_c.dtype, b_c.dtype,
            _triton_int_spec(x_s0), _triton_int_spec(d_s0), _triton_int_spec(d_s1), _triton_int_spec(b_s0),
            single_stream, float(eps), has_bias, has_clip,
            int(gconv_taps), int(gconv_block_size), int(gconv_group_size), swap_grid, num_warps,
        )
        triton_fast_launch(
            _HC_TRITON_CACHE, ckey, _hc_dflash_gconv_combine_norm_branch5120_kernel,
            grid0, grid1, 1, ptr_or, rt_args,
            {
                "HAS_CLIP": has_clip, "TAPS": int(gconv_taps),
                "BLOCK_SIZE": int(gconv_block_size), "GROUP_SIZE": int(gconv_group_size),
                "HIDDEN": hidden_size, "HC_COUNT": hc_count, "SINGLE_STREAM": single_stream,
                "EPS": float(eps), "HAS_BIAS": has_bias, "SWAP_GRID": swap_grid,
            },
            num_warps=num_warps,
        )
        return combined, normed

    block = max(128, triton.next_power_of_2(hidden_size))
    ckey = (
        res_c.device.index, 33, res_c.dtype, sub_2d.dtype, delta_3d.dtype, base_c.dtype,
        inj_c.dtype, w_c.dtype, b_c.dtype,
        _triton_int_spec(x_s0), _triton_int_spec(d_s0), _triton_int_spec(d_s1), _triton_int_spec(b_s0),
        hidden_size, hc_count, single_stream, float(eps), has_bias, has_clip,
        int(gconv_taps), int(gconv_block_size), int(gconv_group_size), block,
    )
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_dflash_gconv_combine_norm_kernel,
        rows, hc_count, 1, ptr_or, rt_args,
        {
            "HAS_CLIP": has_clip, "TAPS": int(gconv_taps),
            "BLOCK_SIZE": int(gconv_block_size), "GROUP_SIZE": int(gconv_group_size),
            "HIDDEN": hidden_size, "HC_COUNT": hc_count, "SINGLE_STREAM": single_stream,
            "EPS": float(eps), "HAS_BIAS": has_bias, "BLOCK": block,
        },
        num_warps=4,
    )
    return combined, normed


def hc_mix_reduce_with_down_inject_triton(
    gate_logits: torch.Tensor,
    normed: torch.Tensor,
    down_inject_raw: torch.Tensor,
    hc_lowrank: int,
    hc_count: int,
    *,
    has_latent: bool = True,
    has_inj: bool = True,
    ws: Optional[HCStaticWorkspace] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused hc_mix_reduce + down/inject epilogue when prev_mix=True and not is_head."""
    logits_c = gate_logits if gate_logits.is_contiguous() else gate_logits.contiguous()
    normed_c = normed if normed.is_contiguous() else normed.contiguous()
    raw_c = down_inject_raw if down_inject_raw.is_contiguous() else down_inject_raw.contiguous()
    prefix = raw_c.shape[:-1]
    width = int(logits_c.shape[-1])
    hidden_size = width // hc_count
    raw_stride = int(raw_c.shape[-1])
    rows = logits_c.numel() // width

    out = _hc_static_empty((*logits_c.shape[:-1], hidden_size), normed_c.dtype, logits_c.device, "out", ws=ws)
    latent = (
        _hc_static_empty((*prefix, hc_lowrank), raw_c.dtype, raw_c.device, "latent", ws=ws)
        if has_latent
        else None
    )
    inject_w = (
        _hc_static_empty((*prefix, hc_count), raw_c.dtype, raw_c.device, "inj", ws=ws)
        if has_inj
        else None
    )
    if rows == 0:
        return out, latent, inject_w

    latent_arg = latent if latent is not None else out
    inj_arg = inject_w if inject_w is not None else out
    block = 512 if hidden_size >= 512 else max(128, triton.next_power_of_2(hidden_size))
    block_l = max(16, triton.next_power_of_2(hc_lowrank)) if has_latent else 16
    block_h = max(16, triton.next_power_of_2(hc_count))
    swap_grid = rows >= 2048
    n_col_blocks = triton.cdiv(hidden_size, block)
    grid0, grid1 = (n_col_blocks, rows) if swap_grid else (rows, n_col_blocks)
    ckey = (
        logits_c.device.index, 30, logits_c.dtype, normed_c.dtype, raw_c.dtype,
        hidden_size, hc_count, hc_lowrank, raw_stride, has_latent, has_inj, block, block_l, block_h, swap_grid,
    )
    ptr_or = (
        logits_c.data_ptr() | normed_c.data_ptr() | out.data_ptr()
        | raw_c.data_ptr() | latent_arg.data_ptr() | inj_arg.data_ptr()
    )
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_mix_reduce_with_down_inject_kernel,
        grid0, grid1, 1, ptr_or,
        (logits_c, normed_c, out, raw_c, latent_arg, inj_arg),
        {
            "RAW_STRIDE": raw_stride, "LOWRANK": hc_lowrank, "HIDDEN": hidden_size,
            "HC_COUNT": hc_count, "HAS_LATENT": has_latent, "HAS_INJ": has_inj,
            "BLOCK": block, "BLOCK_L": block_l, "BLOCK_H": block_h, "SWAP_GRID": swap_grid,
        },
        num_warps=(2 if rows >= 128 else 4),
    )
    return out, latent, inject_w


def _hc_sk_params(rows: int) -> Tuple[int, int, int, int, int]:
    """Select (BLOCK_M, BLOCK_N, SPLIT_K, BLOCK_K, BLOCK_H) for K=20480, N=132."""
    if rows <= 16:
        return 16, 64, 80, 64, 32
    if rows <= 32:
        return 32, 32, 80, 64, 32
    return 64, 64, 40, 64, 32


def hc_down_inject_sk_and_mix_triton(
    normed: torch.Tensor,
    down_inject_weight: torch.Tensor,
    up_weight: torch.Tensor,
    prev_latent: torch.Tensor,
    hc_lowrank: int,
    hc_count: int,
    *,
    ws: Optional[HCStaticWorkspace] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eliminate cublasLt splitKreduce and fuse prev_latent up-proj + mix_reduce into one 1D epilogue kernel."""
    normed_c = normed if normed.is_contiguous() else normed.contiguous()
    w_c = down_inject_weight if down_inject_weight.is_contiguous() else down_inject_weight.contiguous()
    up_c = up_weight if up_weight.is_contiguous() else up_weight.contiguous()
    pl_c = prev_latent if prev_latent.is_contiguous() else prev_latent.contiguous()
    prefix = normed_c.shape[:-1]
    width = int(normed_c.shape[-1])
    hidden_size = width // hc_count
    n_out = hc_lowrank + hc_count
    rows = normed_c.numel() // width

    out = _hc_static_empty((*prefix, hidden_size), normed_c.dtype, normed_c.device, "out", ws=ws)
    latent = _hc_static_empty((*prefix, hc_lowrank), normed_c.dtype, normed_c.device, "latent", ws=ws)
    inject_w = _hc_static_empty((*prefix, hc_count), normed_c.dtype, normed_c.device, "inj", ws=ws)
    if rows == 0:
        return out, latent, inject_w

    bm, bn, sk, bk, bh = _hc_sk_params(rows)
    part = _hc_static_empty((sk, rows, n_out), torch.float32, normed_c.device, "sk_part", ws=ws)
    dev_idx = normed_c.device.index if normed_c.device.index is not None else torch.cuda.current_device()
    n_2d = normed_c.view(rows, width)

    ckey_gemm = (dev_idx, 39, normed_c.dtype, w_c.dtype, n_out, width, sk, bm, bn, bk, 4, 3)
    triton_fast_launch(
        _HC_TRITON_CACHE,
        ckey_gemm,
        _hc_down_inject_sk_gemm_kernel,
        triton.cdiv(rows, bm),
        triton.cdiv(n_out, bn),
        sk,
        n_2d.data_ptr() | w_c.data_ptr() | part.data_ptr(),
        (n_2d, w_c, part, rows),
        {"N": n_out, "K": width, "SPLIT_K": sk, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=4,
        num_stages=3,
    )

    block_sk = max(16, triton.next_power_of_2(sk))
    block_n = max(16, triton.next_power_of_2(n_out))
    can_fuse_up = (
        up_c.dtype == normed_c.dtype
        and pl_c.dtype == normed_c.dtype
        and tuple(up_c.shape) == (width, hc_lowrank)
        and pl_c.numel() == rows * hc_lowrank
        and hidden_size >= 16
        and hc_lowrank >= 16
        and (hc_lowrank & (hc_lowrank - 1)) == 0
    )
    if can_fuse_up:
        pl_2d = pl_c.view(rows, hc_lowrank)
        bm_mix = 16
        bk_mix = hc_lowrank
        ns_mix = 2
        nhb = triton.cdiv(hidden_size, bh)
        grid1d = triton.cdiv(rows, bm_mix) * nhb + rows
        ckey_up_mix = (
            dev_idx, 38, up_c.dtype, pl_c.dtype, normed_c.dtype, out.dtype,
            sk, n_out, hc_lowrank, hc_count, hidden_size, nhb, bm_mix, bk_mix, block_sk, block_n, bh, ns_mix,
        )
        ptr_or = (
            part.data_ptr()
            | up_c.data_ptr()
            | pl_2d.data_ptr()
            | n_2d.data_ptr()
            | out.data_ptr()
            | latent.data_ptr()
            | inject_w.data_ptr()
        )
        triton_fast_launch(
            _HC_TRITON_CACHE,
            ckey_up_mix,
            _hc_up_mix_sk_1d_kernel,
            grid1d,
            1,
            1,
            ptr_or,
            (part, up_c, pl_2d, n_2d, out, latent, inject_w, rows),
            {
                "SPLIT_K": sk,
                "N_OUT": n_out,
                "LOWRANK": hc_lowrank,
                "HC_COUNT": hc_count,
                "HIDDEN": hidden_size,
                "NUM_H_BLOCKS": nhb,
                "BLOCK_M": bm_mix,
                "BLOCK_K": bk_mix,
                "BLOCK_SK": block_sk,
                "BLOCK_N": block_n,
                "BLOCK_H": bh,
            },
            num_warps=4,
            num_stages=ns_mix,
        )
        return out, latent, inject_w

    bh_fallback = 512
    gate_logits = F.linear(prev_latent, up_weight)
    logits_c = gate_logits if gate_logits.is_contiguous() else gate_logits.contiguous()
    ckey_mix = (
        dev_idx, 40, logits_c.dtype, normed_c.dtype, out.dtype,
        sk, n_out, hc_lowrank, hc_count, hidden_size, block_sk, block_n, bh_fallback,
    )
    ptr_or = (
        part.data_ptr()
        | logits_c.data_ptr()
        | n_2d.data_ptr()
        | out.data_ptr()
        | latent.data_ptr()
        | inject_w.data_ptr()
    )
    triton_fast_launch(
        _HC_TRITON_CACHE,
        ckey_mix,
        _hc_mix_reduce_with_down_inject_sk_kernel,
        rows,
        triton.cdiv(hidden_size, bh_fallback),
        1,
        ptr_or,
        (part, logits_c, n_2d, out, latent, inject_w, rows),
        {
            "SPLIT_K": sk,
            "N_OUT": n_out,
            "LOWRANK": hc_lowrank,
            "HC_COUNT": hc_count,
            "HIDDEN": hidden_size,
            "BLOCK_SK": block_sk,
            "BLOCK_N": block_n,
            "BLOCK_H": bh_fallback,
        },
        num_warps=4,
    )
    return out, latent, inject_w


def hc_down_silu_epilogue_triton(
    down_raw: torch.Tensor,
    hc_lowrank: int,
    hc_count: int,
) -> torch.Tensor:
    """Single-launch SiLU(raw[..., :hc_lowrank] / hc_count) directly from padded GEMM output."""
    prefix = down_raw.shape[:-1]
    if not down_raw.is_cuda:
        return F.silu(down_raw[..., :hc_lowrank].float() / hc_count).to(down_raw.dtype)
    raw_c = down_raw if down_raw.is_contiguous() else down_raw.contiguous()
    raw_stride = int(raw_c.shape[-1])
    rows = raw_c.numel() // raw_stride
    latent = _hc_static_empty((*prefix, hc_lowrank), raw_c.dtype, raw_c.device, "latent")
    if rows == 0:
        return latent
    block_r = 1 if rows <= 8 else (8 if rows <= 128 else 16)
    block_l = max(16, triton.next_power_of_2(hc_lowrank))
    ckey = (raw_c.device.index, 31, raw_c.dtype, hc_lowrank, hc_count, raw_stride, block_r, block_l)
    triton_fast_launch(
        _HC_TRITON_CACHE, ckey, _hc_down_silu_epilogue_kernel,
        triton.cdiv(rows, block_r), 1, 1, raw_c.data_ptr() | latent.data_ptr(),
        (raw_c, latent, rows),
        {"LOWRANK": hc_lowrank, "HC_COUNT": hc_count, "RAW_STRIDE": raw_stride, "BLOCK_R": block_r, "BLOCK_L": block_l},
        num_warps=4,
    )
    return latent


def _aligned_hc_out_dim(n: int, *, hidden_size: int = 0) -> int:
    """Align HC linear N dimension to SM100 TensorCore multiples (avoid cutlass_80)."""
    if hidden_size < 1024:
        return 16 if n < 16 else n
    if 32 < n < 64:
        return 64
    return ((n + 15) // 16) * 16


# ---------------------------------------------------------------------------
# Module: Qwen4_5HyperConnection
# ---------------------------------------------------------------------------


class Qwen4_5HyperConnection(nn.Module):
    """4-stream Hyper-Connection block for Qwen 4.5-Flash (Qwen4_5ForCausalLM).

    Supports all four architectural sites:
      - Layer 0 attn_hc:        is_head=True,  prev_mix=True,  emits_latent=True,  use_combine=True
      - Layers 0..39 standard:  is_head=False, prev_mix=True,  emits_latent=True,  use_combine=True
      - Layer 39 mlp_hc:        is_head=False, prev_mix=True,  emits_latent=False, use_combine=True
      - Final mixer:            is_head=False, prev_mix=False, emits_latent=True,  use_combine=False
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        hc_count: int = 4,
        hc_lowrank: int = 128,
        eps: float = 1e-6,
        use_combine: bool = True,
        prev_mix: bool = True,
        is_head: bool = False,
        emits_latent: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        *,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_attn: bool = False,
    ) -> None:
        super().__init__()
        if not emits_latent and (not prev_mix or is_head):
            raise ValueError(
                "emits_latent=False requires prev_mix=True and is_head=False "
                "so gate_latent can come from prev_latent."
            )

        self.sp_size = max(1, int(sp_size))
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group
        self.dp_attn = bool(dp_attn)
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.hc_hidden_size = hc_count * hidden_size
        self.eps = eps
        self.use_combine = use_combine
        self.prev_mix = prev_mix
        self.is_head = is_head
        self.emits_latent = emits_latent

        # Checkpoint-matching parameters:
        # hc_norm_weight is zero-centered (applied as 1.0 + weight in GemmaRMSNorm)
        self.hc_norm_weight = nn.Parameter(
            torch.zeros(self.hc_hidden_size, dtype=dtype, device=device)
        )
        self.hc_norm_bias = nn.Parameter(
            torch.zeros(self.hc_hidden_size, dtype=dtype, device=device)
        )

        if emits_latent:
            self.input_mix_weight_down: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(hc_lowrank, self.hc_hidden_size, dtype=dtype, device=device)
            )
            nn.init.normal_(self.input_mix_weight_down, mean=0.0, std=1.0 / math.sqrt(self.hc_hidden_size))
        else:
            self.register_parameter("input_mix_weight_down", None)

        up_out_dim = hidden_size if is_head else self.hc_hidden_size
        self.input_mix_weight_up = nn.Parameter(
            torch.empty(up_out_dim, hc_lowrank, dtype=dtype, device=device)
        )
        nn.init.normal_(self.input_mix_weight_up, mean=0.0, std=0.02)

        if use_combine:
            self.block_inject_weight: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(hc_count, self.hc_hidden_size, dtype=dtype, device=device)
            )
            nn.init.normal_(self.block_inject_weight, mean=0.0, std=1.0 / math.sqrt(self.hc_hidden_size))
        else:
            self.register_parameter("block_inject_weight", None)

        # Inference caches for padded/fused SM100-aligned [input_mix_weight_down; block_inject_weight]
        self._fused_down_inject_key: Optional[Tuple[int, int, int, int]] = None
        self._fused_down_inject_weight: Optional[torch.Tensor] = None
        self._padded_down_key: Optional[Tuple[int, int]] = None
        self._padded_down_weight: Optional[torch.Tensor] = None
        self._padded_inj_key: Optional[Tuple[int, int]] = None
        self._padded_inj_weight: Optional[torch.Tensor] = None

    def _get_fused_down_inject_weight(self) -> torch.Tensor:
        w_down = self.input_mix_weight_down
        w_inj = self.block_inject_weight
        assert w_down is not None and w_inj is not None
        if torch.is_grad_enabled() and (w_down.requires_grad or w_inj.requires_grad):
            w_cat = torch.cat([w_down, w_inj], dim=0)
            pad_n = _aligned_hc_out_dim(int(w_cat.shape[0]), hidden_size=self.hidden_size) - int(w_cat.shape[0])
            return F.pad(w_cat, (0, 0, 0, pad_n)) if pad_n > 0 else w_cat
        key = (w_down.data_ptr(), 0 if w_down.is_inference() else w_down._version, w_inj.data_ptr(), 0 if w_inj.is_inference() else w_inj._version)
        if self._fused_down_inject_key != key or self._fused_down_inject_weight is None:
            w_cat = torch.cat([w_down, w_inj], dim=0)
            pad_n = _aligned_hc_out_dim(int(w_cat.shape[0]), hidden_size=self.hidden_size) - int(w_cat.shape[0])
            self._fused_down_inject_weight = (
                F.pad(w_cat, (0, 0, 0, pad_n)).contiguous() if pad_n > 0 else w_cat.contiguous()
            )
            self._fused_down_inject_key = key
        return self._fused_down_inject_weight

    def _get_padded_down_weight(self) -> torch.Tensor:
        w_down = self.input_mix_weight_down
        assert w_down is not None
        out_n = int(w_down.shape[0])
        pad_n = _aligned_hc_out_dim(out_n, hidden_size=self.hidden_size) - out_n
        if pad_n <= 0:
            return w_down
        if torch.is_grad_enabled() and w_down.requires_grad:
            return F.pad(w_down, (0, 0, 0, pad_n))
        key = (w_down.data_ptr(), 0 if w_down.is_inference() else w_down._version)
        if self._padded_down_key != key or self._padded_down_weight is None:
            self._padded_down_weight = F.pad(w_down, (0, 0, 0, pad_n)).contiguous()
            self._padded_down_key = key
        return self._padded_down_weight

    def _get_padded_inject_weight(self) -> torch.Tensor:
        w_inj = self.block_inject_weight
        assert w_inj is not None
        out_n = int(w_inj.shape[0])
        pad_n = _aligned_hc_out_dim(out_n, hidden_size=self.hidden_size) - out_n
        if pad_n <= 0:
            return w_inj
        if torch.is_grad_enabled() and w_inj.requires_grad:
            return F.pad(w_inj, (0, 0, 0, pad_n))
        key = (w_inj.data_ptr(), 0 if w_inj.is_inference() else w_inj._version)
        if self._padded_inj_key != key or self._padded_inj_weight is None:
            self._padded_inj_weight = F.pad(w_inj, (0, 0, 0, pad_n)).contiguous()
            self._padded_inj_key = key
        return self._padded_inj_weight

    def _project_and_mix_from_normed(
        self,
        normed: torch.Tensor,
        residual: torch.Tensor,
        prev_latent: Optional[torch.Tensor],
        *,
        ws: Optional[HCStaticWorkspace] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Fast fused path: when prev_mix=True and not is_head, gate_logits depends only on
        # prev_latent, so down/inject epilogue fuses directly into _hc_mix_reduce_with_down_inject.
        if (
            self.prev_mix
            and not self.is_head
            and normed.is_cuda
            and (self.emits_latent or self.use_combine)
        ):
            if prev_latent is None:
                raise ValueError(
                    "prev_latent must be provided when prev_mix=True and is_head=False"
                )
            if self.emits_latent and self.use_combine:
                w_fused = self._get_fused_down_inject_weight()
                n_out = self.hc_lowrank + self.hc_count
                width = self.hc_count * self.hidden_size
                rows = normed.numel() // width
                if (
                    self.hidden_size == 5120
                    and self.hc_count == 4
                    and self.hc_lowrank == 128
                    and 0 < rows <= 48
                    and normed.dtype in (torch.bfloat16, torch.float16)
                    and w_fused.dtype == normed.dtype
                    and not (
                        torch.is_grad_enabled()
                        and (
                            normed.requires_grad
                            or self.input_mix_weight_down.requires_grad
                            or self.block_inject_weight.requires_grad
                        )
                    )
                ):
                    mixed, latent, inject_w = hc_down_inject_sk_and_mix_triton(
                        normed,
                        w_fused[:n_out],
                        self.input_mix_weight_up,
                        prev_latent,
                        self.hc_lowrank,
                        self.hc_count,
                        ws=ws,
                    )
                    return mixed, residual, inject_w, latent
                raw = F.linear(normed, w_fused)
            elif self.emits_latent:
                raw = F.linear(normed, self._get_padded_down_weight())
            else:
                raw = F.linear(normed, self._get_padded_inject_weight())
            gate_logits = F.linear(prev_latent, self.input_mix_weight_up)
            mixed, latent, inject_w = hc_mix_reduce_with_down_inject_triton(
                gate_logits,
                normed,
                raw,
                self.hc_lowrank,
                self.hc_count,
                has_latent=self.emits_latent,
                has_inj=self.use_combine,
                ws=ws,
            )
            return mixed, residual, inject_w, latent

        # 1. Down & inject projections (SM100-aligned fused GEMM when both exist)
        if self.emits_latent and self.use_combine:
            w_fused = self._get_fused_down_inject_weight()
            raw = F.linear(normed, w_fused)
            latent, inject_w = hc_down_inject_epilogue_triton(
                raw, self.hc_lowrank, self.hc_count, ws=ws
            )
        else:
            latent = None
            inject_w = None
            if self.emits_latent:
                raw_down = F.linear(normed, self._get_padded_down_weight())
                latent = hc_down_silu_epilogue_triton(
                    raw_down, self.hc_lowrank, self.hc_count
                )
            if self.use_combine:
                raw_inj = F.linear(normed, self._get_padded_inject_weight())
                inject_w = hc_inject_epilogue_triton(raw_inj, self.hc_count, ws=ws)

        # 2. hc_prev_mix gate latent selection
        if not self.prev_mix or self.is_head:
            gate_latent = latent
        else:
            gate_latent = prev_latent
        if gate_latent is None:
            raise ValueError(
                "prev_latent must be provided when prev_mix=True and is_head=False"
            )

        # 3. Up projection + sigmoid gate mix
        gate_logits = F.linear(gate_latent, self.input_mix_weight_up)
        if self.is_head:
            mixed = hc_head_mix_triton(gate_logits, normed, ws=ws)
        else:
            mixed = hc_mix_reduce_triton(gate_logits, normed, self.hc_count, ws=ws)

        return mixed, residual, inject_w, latent

    def mix_from_normed(
        self,
        normed: torch.Tensor,
        residual: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Read-side hyper-connection mix from precomputed grouped RMSNorm output."""
        return self._project_and_mix_from_normed(normed, residual, prev_latent)

    def mix_with_latent(
        self,
        hyper_input: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
        *,
        ws: Optional[HCStaticWorkspace] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Read-side hyper-connection mix returning (mixed, residual, inject_w, next_latent)."""
        if not hyper_input.is_cuda or self.hidden_size > 8192:
            normed = grouped_gemma_rmsnorm_bias_triton(
                hyper_input,
                self.hc_norm_weight,
                self.hc_norm_bias,
                self.hc_count,
                self.eps,
                normed_tag="normed",
            )
            return self._project_and_mix_from_normed(normed, hyper_input, prev_latent, ws=ws)

        x_c = hyper_input.contiguous()
        w_c = self.hc_norm_weight.contiguous()
        b_c = self.hc_norm_bias.contiguous() if self.hc_norm_bias is not None else w_c
        x_w = int(x_c.shape[-1])
        width = self.hc_count * self.hidden_size
        single_stream = self.hc_count > 1 and x_w == self.hidden_size
        normed = _hc_static_empty((*x_c.shape[:-1], width), x_c.dtype, x_c.device, "normed")
        rows = x_c.numel() // x_w
        if rows > 0:
            _launch_grouped_gemma_rmsnorm_bias(
                x_c, w_c, b_c, normed, width, self.hidden_size, x_w,
                self.hc_count, single_stream, float(self.eps), self.hc_norm_bias is not None, rows,
            )
        return self._project_and_mix_from_normed(normed, hyper_input, prev_latent, ws=ws)

    def combine(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        inject_w: torch.Tensor,
    ) -> torch.Tensor:
        """Write-side hyper-connection injection back into the 4-stream residual."""
        return hc_combine_triton(block_output, residual, inject_w)

    def combine_and_mix_with_latent(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        prev_inject_w: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
        *,
        ws: Optional[HCStaticWorkspace] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Fused previous-block combine + current-block GemmaRMSNorm+bias + mix_with_latent."""
        combined, normed = hc_combine_and_norm_triton(
            block_output,
            residual,
            prev_inject_w,
            self.hc_norm_weight,
            self.hc_norm_bias,
            self.eps,
            normed_tag="normed",
            ws=ws,
        )
        return self._project_and_mix_from_normed(normed, combined, prev_latent, ws=ws)

    def combine_dflash_gconv_and_mix_with_latent(
        self,
        raw_sub: torch.Tensor,
        gconv_delta: torch.Tensor,
        gconv_base: torch.Tensor,
        gconv_block_size: int,
        gconv_group_size: int,
        gconv_taps: int,
        residual: torch.Tensor,
        prev_inject_w: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
        *,
        clip_value: Optional[float] = None,
        ws: Optional[HCStaticWorkspace] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Single-pass DFlashGroupedConv.finish + HC combine+norm + mix_with_latent."""
        combined, normed = hc_dflash_gconv_combine_norm_triton(
            raw_sub,
            gconv_delta,
            gconv_base,
            gconv_block_size,
            gconv_group_size,
            gconv_taps,
            residual,
            prev_inject_w,
            self.hc_norm_weight,
            self.hc_norm_bias,
            self.eps,
            clip_value=clip_value,
            normed_tag="normed",
            ws=ws,
        )
        return self._project_and_mix_from_normed(normed, combined, prev_latent, ws=ws)

    def forward(
        self,
        hyper_input: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.mix_with_latent(hyper_input, prev_latent=prev_latent)

    # ------------------------------------------------------------------
    # Exact FP32 PyTorch Reference Methods (for numerical verification)
    # ------------------------------------------------------------------

    def mix_with_latent_ref(
        self,
        hyper_input: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Exact pure-PyTorch FP32 reference matching the Qwen 4.5-Flash HC specification."""
        out_dtype = hyper_input.dtype
        streams = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size)).float()
        normed_f = streams * torch.rsqrt(
            streams.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normed = (
            normed_f.flatten(-2) * (1.0 + self.hc_norm_weight.float())
            + self.hc_norm_bias.float()
        ).to(out_dtype)
        normed_f = normed.float()

        latent_f: Optional[torch.Tensor] = None
        if self.emits_latent:
            assert self.input_mix_weight_down is not None
            latent_f = F.silu(
                F.linear(normed_f, self.input_mix_weight_down.float()) / self.hc_count
            )

        inject_f: Optional[torch.Tensor] = None
        if self.use_combine:
            assert self.block_inject_weight is not None
            inject_f = 2.0 * torch.sigmoid(
                F.linear(normed_f, self.block_inject_weight.float()) / self.hc_count
            )

        if not self.prev_mix or self.is_head:
            gate_latent_f = latent_f
        else:
            gate_latent_f = prev_latent.float() if prev_latent is not None else None
        if gate_latent_f is None:
            raise ValueError(
                "prev_latent must be provided when prev_mix=True and is_head=False"
            )

        gate_logits_f = F.linear(gate_latent_f, self.input_mix_weight_up.float())
        if self.is_head:
            mixed_f = torch.sigmoid(gate_logits_f) * normed_f[..., : self.hidden_size]
        else:
            gate_f = torch.sigmoid(gate_logits_f).unflatten(
                -1, (self.hc_count, self.hidden_size)
            )
            mixed_f = (
                gate_f * normed_f.unflatten(-1, (self.hc_count, self.hidden_size))
            ).mean(dim=-2)

        return (
            mixed_f.to(out_dtype),
            hyper_input,
            inject_f.to(out_dtype) if inject_f is not None else None,
            latent_f.to(out_dtype) if latent_f is not None else None,
        )

    def combine_ref(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        inject_w: torch.Tensor,
    ) -> torch.Tensor:
        return hc_combine_ref(block_output, residual, inject_w)

    def combine_and_mix_with_latent_ref(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        prev_inject_w: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        combined = self.combine_ref(block_output, residual, prev_inject_w)
        return self.mix_with_latent_ref(combined, prev_latent=prev_latent)
