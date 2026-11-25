"""Qwen 4.5-Flash Causal Depthwise Short-Conv (k=4) with Triton Prefill & Decode kernels.

Supports:
  - KV short-conv (channels=256, residual=True, use_activation=False)
  - Output short-conv (channels=5120, residual=True/False, use_activation=False/True)
  - PLE short-conv (channels=20480, residual=False, use_activation=False)
  - Stateful Prefill, Chunked Prefill, Single-Step Decode (T=1), and Multi-Step Decode (T>1)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .contracts import PLELayerState, ShortConvState

ConvStateLike = Union[ShortConvState, PLELayerState, torch.Tensor]


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _causal_dwconv1d_prefill_kernel(
    X_PTR,
    W_PTR,
    STATE_PTR,
    OUT_PTR,
    T,
    C: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_STATE: tl.constexpr,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    t_start = tl.program_id(1).to(tl.int64) * BLOCK_T
    c_start = tl.program_id(2).to(tl.int64) * BLOCK_C

    t_offs = t_start + tl.arange(0, BLOCK_T).to(tl.int64)
    c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
    t_mask = t_offs < T
    c_mask = c_offs < C
    tc_mask = t_mask[:, None] & c_mask[None, :]

    x_batch_base = batch * T * C
    s_batch_base = batch * C * STATE_LEN
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    x_cur = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for k in tl.static_range(KERNEL_SIZE):
        w_k = tl.load(
            W_PTR + c_offs * KERNEL_SIZE + k, mask=c_mask, other=0.0
        ).to(tl.float32)
        src_t = t_offs + (k - STATE_LEN)
        in_x = tc_mask & (src_t[:, None] >= 0)
        val = tl.load(
            X_PTR + x_batch_base + src_t[:, None] * C + c_offs[None, :],
            mask=in_x,
            other=0.0,
        ).to(tl.float32)
        if HAS_STATE:
            if t_start < STATE_LEN:
                in_s = tc_mask & (src_t[:, None] < 0)
                s_idx = src_t + STATE_LEN
                s_val = tl.load(
                    STATE_PTR
                    + s_batch_base
                    + c_offs[None, :] * STATE_LEN
                    + s_idx[:, None],
                    mask=in_s,
                    other=0.0,
                ).to(tl.float32)
                val = tl.where(src_t[:, None] < 0, s_val, val)
        if RESIDUAL:
            if k == KERNEL_SIZE - 1:
                x_cur = val
        acc += val * w_k[None, :]

    if USE_SILU:
        acc = acc * tl.sigmoid(acc)
    if RESIDUAL:
        acc = x_cur + acc

    tl.store(
        OUT_PTR + x_batch_base + t_offs[:, None] * C + c_offs[None, :],
        acc,
        mask=tc_mask,
    )


@triton.jit(do_not_specialize=["T"])
def _update_conv_state_kernel(
    X_PTR,
    STATE_PTR,
    T,
    C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    c_mask = c_offs < C
    x_batch_base = batch * T * C
    s_base = batch * C * STATE_LEN + c_offs * STATE_LEN

    for j in tl.static_range(STATE_LEN):
        src_t = T - STATE_LEN + j
        if src_t >= 0:
            val = tl.load(X_PTR + x_batch_base + src_t * C + c_offs, mask=c_mask)
        else:
            val = tl.load(STATE_PTR + s_base + (j + T), mask=c_mask)
        tl.store(STATE_PTR + s_base + j, val, mask=c_mask)


@triton.jit
def _causal_dwconv1d_decode_k4_kernel(
    X_PTR,
    W_PTR,
    STATE_PTR,
    OUT_PTR,
    C: tl.constexpr,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    ROUND_PRE_ACT: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    CLAMP_VAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    c_mask = c_offs < C

    x_off = batch * C + c_offs
    s_base = batch * C * 3 + c_offs * 3
    w_base = c_offs * 4

    s0_raw = tl.load(STATE_PTR + s_base + 0, mask=c_mask, other=0.0)
    s1_raw = tl.load(STATE_PTR + s_base + 1, mask=c_mask, other=0.0)
    s2_raw = tl.load(STATE_PTR + s_base + 2, mask=c_mask, other=0.0)
    x0_raw = tl.load(X_PTR + x_off, mask=c_mask, other=0.0)

    w0 = tl.load(W_PTR + w_base + 0, mask=c_mask, other=0.0).to(tl.float32)
    w1 = tl.load(W_PTR + w_base + 1, mask=c_mask, other=0.0).to(tl.float32)
    w2 = tl.load(W_PTR + w_base + 2, mask=c_mask, other=0.0).to(tl.float32)
    w3 = tl.load(W_PTR + w_base + 3, mask=c_mask, other=0.0).to(tl.float32)

    x0_f = x0_raw.to(tl.float32)
    acc = (
        s0_raw.to(tl.float32) * w0
        + s1_raw.to(tl.float32) * w1
        + s2_raw.to(tl.float32) * w2
        + x0_f * w3
    )

    if ROUND_PRE_ACT:
        acc = acc.to(OUT_PTR.dtype.element_ty).to(tl.float32)
    if USE_SILU:
        acc = acc * tl.sigmoid(acc)
    if RESIDUAL:
        acc = x0_f + acc
    if HAS_CLAMP:
        acc = acc.to(OUT_PTR.dtype.element_ty).to(tl.float32)
        acc = tl.minimum(tl.maximum(acc, -CLAMP_VAL), CLAMP_VAL)

    tl.store(OUT_PTR + x_off, acc, mask=c_mask)

    # Roll state in-place: [s0, s1, s2] -> [s1, s2, x0]
    tl.store(STATE_PTR + s_base + 0, s1_raw, mask=c_mask)
    tl.store(STATE_PTR + s_base + 1, s2_raw, mask=c_mask)
    tl.store(STATE_PTR + s_base + 2, x0_raw, mask=c_mask)


@triton.jit
def _causal_dwconv1d_verify_k4_kernel(
    X_PTR,
    W_PTR,
    INIT_STATE_PTR,
    STAGE_STATE_PTR,
    OUT_PTR,
    stride_xb: tl.int64,
    stride_xv: tl.int64,
    stride_xc: tl.int64,
    stride_ob: tl.int64,
    stride_ov: tl.int64,
    stride_oc: tl.int64,
    stride_sb: tl.int64,
    stride_sc: tl.int64,
    stride_sk: tl.int64,
    stride_stb: tl.int64,
    stride_stv: tl.int64,
    stride_stc: tl.int64,
    stride_stk: tl.int64,
    C: tl.constexpr,
    V: tl.constexpr,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    ROUND_PRE_ACT: tl.constexpr,
    HAS_STAGE: tl.constexpr,
    UPDATE_INIT: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    CLAMP_VAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Single-launch K=4 causal dwconv1d across V speculative verify steps in registers."""
    batch = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    c_mask = c_offs < C

    s_base = INIT_STATE_PTR + batch * stride_sb + c_offs * stride_sc
    s0_raw = tl.load(s_base + 0 * stride_sk, mask=c_mask, other=0.0)
    s1_raw = tl.load(s_base + 1 * stride_sk, mask=c_mask, other=0.0)
    s2_raw = tl.load(s_base + 2 * stride_sk, mask=c_mask, other=0.0)

    w_base = c_offs * 4
    w0 = tl.load(W_PTR + w_base + 0, mask=c_mask, other=0.0).to(tl.float32)
    w1 = tl.load(W_PTR + w_base + 1, mask=c_mask, other=0.0).to(tl.float32)
    w2 = tl.load(W_PTR + w_base + 2, mask=c_mask, other=0.0).to(tl.float32)
    w3 = tl.load(W_PTR + w_base + 3, mask=c_mask, other=0.0).to(tl.float32)

    x_batch_base = X_PTR + batch * stride_xb + c_offs * stride_xc
    o_batch_base = OUT_PTR + batch * stride_ob + c_offs * stride_oc
    st_batch_base = STAGE_STATE_PTR + batch * stride_stb + c_offs * stride_stc
    out_ty = OUT_PTR.dtype.element_ty

    for v in tl.static_range(V):
        x_raw = tl.load(x_batch_base + v * stride_xv, mask=c_mask, other=0.0)
        x_f = x_raw.to(tl.float32)
        acc = (
            s0_raw.to(tl.float32) * w0
            + s1_raw.to(tl.float32) * w1
            + s2_raw.to(tl.float32) * w2
            + x_f * w3
        )
        if ROUND_PRE_ACT:
            acc = acc.to(out_ty).to(tl.float32)
        if USE_SILU:
            acc = acc * tl.sigmoid(acc)
        if RESIDUAL:
            acc = x_f + acc
        if HAS_CLAMP:
            acc = acc.to(out_ty).to(tl.float32)
            acc = tl.minimum(tl.maximum(acc, -CLAMP_VAL), CLAMP_VAL)
        tl.store(o_batch_base + v * stride_ov, acc, mask=c_mask)

        s0_raw = s1_raw
        s1_raw = s2_raw
        s2_raw = x_raw
        if HAS_STAGE:
            st_v = st_batch_base + v * stride_stv
            tl.store(st_v + 0 * stride_stk, s0_raw, mask=c_mask)
            tl.store(st_v + 1 * stride_stk, s1_raw, mask=c_mask)
            tl.store(st_v + 2 * stride_stk, s2_raw, mask=c_mask)

    if UPDATE_INIT:
        tl.store(s_base + 0 * stride_sk, s0_raw, mask=c_mask)
        tl.store(s_base + 1 * stride_sk, s1_raw, mask=c_mask)
        tl.store(s_base + 2 * stride_sk, s2_raw, mask=c_mask)


@triton.jit
def _causal_dwconv1d_ragged_prefill_k4_kernel(
    X_PTR,
    W_PTR,
    INIT_STATE_PTR,
    OUT_PTR,
    CU_SEQLENS_PTR,
    TILE_REQ_PTR,
    TILE_T0_PTR,
    stride_xn: tl.int64,
    stride_xc: tl.int64,
    stride_on: tl.int64,
    stride_oc: tl.int64,
    stride_ib: tl.int64,
    stride_ic: tl.int64,
    stride_ik: tl.int64,
    C: tl.constexpr,
    CONTIG_C: tl.constexpr,
    HAS_INIT: tl.constexpr,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    ROUND_PRE_ACT: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    CLAMP_VAL: tl.constexpr,
    SUB_T: tl.constexpr,
    SUB_NUM: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fast 1D register sliding-window K=4 causal dwconv1d ragged prefill kernel."""
    pid0 = tl.program_id(0).to(tl.int64)
    tile_id = pid0 // SUB_NUM
    sub_id = pid0 % SUB_NUM
    c_start = tl.program_id(1).to(tl.int64) * BLOCK_C

    b = tl.load(TILE_REQ_PTR + tile_id).to(tl.int64)
    t_start = tl.load(TILE_T0_PTR + tile_id).to(tl.int64) + sub_id * SUB_T
    req_start = tl.load(CU_SEQLENS_PTR + b).to(tl.int64)
    req_end = tl.load(CU_SEQLENS_PTR + b + 1).to(tl.int64)
    req_len = req_end - req_start
    if t_start >= req_len:
        return
    tile_len = tl.minimum(SUB_T, req_len - t_start)

    c_offs = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    x_col = c_offs if CONTIG_C else c_offs.to(tl.int64) * stride_xc
    o_col = c_offs if CONTIG_C else c_offs.to(tl.int64) * stride_oc

    w_base = c_offs.to(tl.int64) * 4
    w0 = tl.load(W_PTR + w_base + 0, mask=c_mask, other=0.0).to(tl.float32)
    w1 = tl.load(W_PTR + w_base + 1, mask=c_mask, other=0.0).to(tl.float32)
    w2 = tl.load(W_PTR + w_base + 2, mask=c_mask, other=0.0).to(tl.float32)
    w3 = tl.load(W_PTR + w_base + 3, mask=c_mask, other=0.0).to(tl.float32)

    if t_start >= 3:
        s0 = tl.load(X_PTR + (req_start + t_start - 3) * stride_xn + x_col, mask=c_mask, other=0.0).to(tl.float32)
        s1 = tl.load(X_PTR + (req_start + t_start - 2) * stride_xn + x_col, mask=c_mask, other=0.0).to(tl.float32)
        s2 = tl.load(X_PTR + (req_start + t_start - 1) * stride_xn + x_col, mask=c_mask, other=0.0).to(tl.float32)
    else:
        t0 = t_start - 3
        t1 = t_start - 2
        t2 = t_start - 1
        s0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        s1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        s2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        if t0 >= 0:
            s0 = tl.load(X_PTR + (req_start + t0) * stride_xn + x_col, mask=c_mask, other=0.0).to(tl.float32)
        elif HAS_INIT:
            s0 = tl.load(
                INIT_STATE_PTR + b * stride_ib + c_offs.to(tl.int64) * stride_ic + (t0 + 3) * stride_ik,
                mask=c_mask,
                other=0.0,
            ).to(tl.float32)
        if t1 >= 0:
            s1 = tl.load(X_PTR + (req_start + t1) * stride_xn + x_col, mask=c_mask, other=0.0).to(tl.float32)
        elif HAS_INIT:
            s1 = tl.load(
                INIT_STATE_PTR + b * stride_ib + c_offs.to(tl.int64) * stride_ic + (t1 + 3) * stride_ik,
                mask=c_mask,
                other=0.0,
            ).to(tl.float32)
        if t2 >= 0:
            s2 = tl.load(X_PTR + (req_start + t2) * stride_xn + x_col, mask=c_mask, other=0.0).to(tl.float32)
        elif HAS_INIT:
            s2 = tl.load(
                INIT_STATE_PTR + b * stride_ib + c_offs.to(tl.int64) * stride_ic + (t2 + 3) * stride_ik,
                mask=c_mask,
                other=0.0,
            ).to(tl.float32)

    out_ty = OUT_PTR.dtype.element_ty
    x_ptr = X_PTR + (req_start + t_start) * stride_xn + x_col
    o_ptr = OUT_PTR + (req_start + t_start) * stride_on + o_col

    for _dt in range(tile_len):
        xf = tl.load(x_ptr, mask=c_mask, other=0.0).to(tl.float32)
        acc = s0 * w0 + s1 * w1 + s2 * w2 + xf * w3
        if ROUND_PRE_ACT:
            acc = acc.to(out_ty).to(tl.float32)
        if USE_SILU:
            acc = acc * tl.sigmoid(acc)
        if RESIDUAL:
            acc = xf + acc
        if HAS_CLAMP:
            acc = acc.to(out_ty).to(tl.float32)
            acc = tl.minimum(tl.maximum(acc, -CLAMP_VAL), CLAMP_VAL)
        tl.store(o_ptr, acc.to(out_ty), mask=c_mask)
        s0 = s1
        s1 = s2
        s2 = xf
        x_ptr += stride_xn
        o_ptr += stride_on


@triton.jit
def _causal_dwconv1d_ragged_prefill_kernel(
    X_PTR,
    W_PTR,
    INIT_STATE_PTR,
    OUT_PTR,
    CU_SEQLENS_PTR,
    TILE_REQ_PTR,
    TILE_T0_PTR,
    stride_xn: tl.int64,
    stride_xc: tl.int64,
    stride_on: tl.int64,
    stride_oc: tl.int64,
    stride_ib: tl.int64,
    stride_ic: tl.int64,
    stride_ik: tl.int64,
    C: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_INIT: tl.constexpr,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    ROUND_PRE_ACT: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    CLAMP_VAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Whole-batch packed ragged causal depthwise conv1d prefill kernel."""
    tile_id = tl.program_id(0).to(tl.int64)
    c_start = tl.program_id(1).to(tl.int64) * BLOCK_C

    b = tl.load(TILE_REQ_PTR + tile_id).to(tl.int64)
    t_start = tl.load(TILE_T0_PTR + tile_id).to(tl.int64)
    req_start = tl.load(CU_SEQLENS_PTR + b).to(tl.int64)
    req_end = tl.load(CU_SEQLENS_PTR + b + 1).to(tl.int64)
    req_len = req_end - req_start

    t_offs = t_start + tl.arange(0, BLOCK_T).to(tl.int64)
    c_offs = c_start + tl.arange(0, BLOCK_C).to(tl.int64)
    t_mask = t_offs < req_len
    c_mask = c_offs < C
    tc_mask = t_mask[:, None] & c_mask[None, :]

    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    x_cur = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    out_ty = OUT_PTR.dtype.element_ty

    for k in tl.static_range(KERNEL_SIZE):
        w_k = tl.load(W_PTR + c_offs * KERNEL_SIZE + k, mask=c_mask, other=0.0).to(tl.float32)
        src_t = t_offs + (k - STATE_LEN)
        in_x = tc_mask & (src_t[:, None] >= 0)
        safe_src = tl.where(src_t >= 0, req_start + src_t, 0)
        val = tl.load(
            X_PTR + safe_src[:, None] * stride_xn + c_offs[None, :] * stride_xc,
            mask=in_x,
            other=0.0,
        ).to(tl.float32)
        if HAS_INIT:
            if t_start < STATE_LEN:
                in_s = tc_mask & (src_t[:, None] < 0)
                s_idx = tl.where(src_t < 0, src_t + STATE_LEN, 0)
                s_val = tl.load(
                    INIT_STATE_PTR
                    + b * stride_ib
                    + c_offs[None, :] * stride_ic
                    + s_idx[:, None] * stride_ik,
                    mask=in_s,
                    other=0.0,
                ).to(tl.float32)
                val = tl.where(src_t[:, None] < 0, s_val, val)
        if RESIDUAL:
            if k == KERNEL_SIZE - 1:
                x_cur = val
        acc += val * w_k[None, :]

    if ROUND_PRE_ACT:
        acc = acc.to(out_ty).to(tl.float32)
    if USE_SILU:
        acc = acc * tl.sigmoid(acc)
    if RESIDUAL:
        acc = x_cur + acc
    if HAS_CLAMP:
        acc = acc.to(out_ty).to(tl.float32)
        acc = tl.minimum(tl.maximum(acc, -CLAMP_VAL), CLAMP_VAL)

    tl.store(
        OUT_PTR + (req_start + t_offs[:, None]) * stride_on + c_offs[None, :] * stride_oc,
        acc,
        mask=tc_mask,
    )


@triton.jit
def _scatter_conv_snapshots_ragged_kernel(
    X_PTR,
    INIT_STATE_PTR,
    CU_SEQLENS_PTR,
    SNAP_TOK_PTR,
    SNAP_REQ_PTR,
    SNAP_DST_PTR,
    DST_PTR,
    stride_xn: tl.int64,
    stride_xc: tl.int64,
    stride_ib: tl.int64,
    stride_ic: tl.int64,
    stride_ik: tl.int64,
    dst_stride_row: tl.int64,
    dst_col_offset: tl.int64,
    dst_rows: tl.int64,
    C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_INIT: tl.constexpr,
    SKIP_INVALID_PHYS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Scatter boundary 3-token pre-conv states for packed ragged prefill into cache or buffer."""
    s_id = tl.program_id(0).to(tl.int64)
    dst_idx = tl.load(SNAP_DST_PTR + s_id).to(tl.int64)
    if SKIP_INVALID_PHYS:
        if dst_idx <= 0 or dst_idx >= dst_rows:
            return
    g_tok = tl.load(SNAP_TOK_PTR + s_id).to(tl.int64)
    b = tl.load(SNAP_REQ_PTR + s_id).to(tl.int64)
    req_start = tl.load(CU_SEQLENS_PTR + b).to(tl.int64)
    local_t = g_tok - req_start

    c_offs = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    c_mask = c_offs < C
    dst_base = DST_PTR + dst_idx * dst_stride_row + dst_col_offset + c_offs * STATE_LEN
    dst_ty = DST_PTR.dtype.element_ty

    for j in tl.static_range(STATE_LEN):
        src_local = local_t - (STATE_LEN - 1) + j
        if src_local >= 0:
            val = tl.load(
                X_PTR + (req_start + src_local) * stride_xn + c_offs * stride_xc,
                mask=c_mask,
                other=0.0,
            )
        elif HAS_INIT:
            val = tl.load(
                INIT_STATE_PTR
                + b * stride_ib
                + c_offs * stride_ic
                + (src_local + STATE_LEN) * stride_ik,
                mask=c_mask,
                other=0.0,
            )
        else:
            val = tl.zeros((BLOCK_C,), dtype=dst_ty)
        tl.store(dst_base + j, val.to(dst_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Helper / Functional Wrappers
# ---------------------------------------------------------------------------


def _extract_conv_state(state: Optional[ConvStateLike]) -> Optional[torch.Tensor]:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state
    return state.conv_state


def causal_dwconv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: Optional[torch.Tensor] = None,
    *,
    residual: bool = False,
    use_activation: bool = False,
) -> torch.Tensor:
    """Exact pure-PyTorch FP32 reference for causal depthwise conv1d + state update."""
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, T, C], got {tuple(x.shape)}")
    B, T, C = x.shape
    kernel_size = int(weight.shape[-1])
    state_len = kernel_size - 1
    if T == 0:
        return torch.empty_like(x)

    x_t = x.transpose(1, 2)  # [B, C, T]
    if conv_state is not None:
        if tuple(conv_state.shape) != (B, C, state_len):
            raise ValueError(
                f"conv_state shape {tuple(conv_state.shape)} != {(B, C, state_len)}"
            )
        x_seq = torch.cat([conv_state, x_t], dim=-1)  # [B, C, state_len + T]
        if state_len > 0:
            conv_state.copy_(x_seq[:, :, -state_len:])
    else:
        x_seq = F.pad(x_t, (state_len, 0))

    y = F.conv1d(x_seq.float(), weight.float(), bias=None, stride=1, padding=0, groups=C)
    if use_activation:
        y = F.silu(y)
    if residual:
        y = x_t.float() + y
    return y.transpose(1, 2).contiguous().to(x.dtype)


def causal_dwconv1d_prefill_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: Optional[torch.Tensor] = None,
    *,
    residual: bool = False,
    use_activation: bool = False,
) -> torch.Tensor:
    """Fused Triton causal depthwise conv1d for Prefill / multi-step input [B, T, C].

    If ``conv_state`` ([B, C, K-1]) is provided, it is consumed as left-context
    and updated in-place with the last ``K-1`` tokens of ``[conv_state, x]``.
    """
    if not x.is_cuda:
        return causal_dwconv1d_ref(
            x, weight, conv_state, residual=residual, use_activation=use_activation
        )
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, T, C], got {tuple(x.shape)}")

    x_c = x.contiguous()
    w_c = weight.contiguous()
    B, T, C = x_c.shape
    kernel_size = int(w_c.shape[-1])
    state_len = kernel_size - 1
    out = torch.empty_like(x_c)
    if B == 0 or T == 0:
        return out

    if conv_state is not None:
        if tuple(conv_state.shape) != (B, C, state_len):
            raise ValueError(
                f"conv_state shape {tuple(conv_state.shape)} != {(B, C, state_len)}"
            )
        if not conv_state.is_contiguous():
            raise ValueError("conv_state must be contiguous for in-place Triton update")

    block_c = 64 if C <= 256 else 128
    block_t = 16 if (T <= 32 or C >= 4096) else 32
    num_warps = 4

    _causal_dwconv1d_prefill_kernel[
        (B, triton.cdiv(T, block_t), triton.cdiv(C, block_c))
    ](
        x_c,
        w_c,
        conv_state if conv_state is not None else x_c,
        out,
        T=T,
        C=C,
        KERNEL_SIZE=kernel_size,
        STATE_LEN=state_len,
        HAS_STATE=conv_state is not None and state_len > 0,
        RESIDUAL=residual,
        USE_SILU=use_activation,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )

    if conv_state is not None and state_len > 0:
        upd_block_c = 256 if C <= 512 else 512
        _update_conv_state_kernel[(B, triton.cdiv(C, upd_block_c))](
            x_c,
            conv_state,
            T=T,
            C=C,
            STATE_LEN=state_len,
            BLOCK_C=upd_block_c,
            num_warps=4,
        )
    return out


def causal_dwconv1d_decode_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    *,
    residual: bool = False,
    use_activation: bool = False,
    round_pre_act: bool = False,
    clamp_val: Optional[float] = None,
) -> torch.Tensor:
    """Single-launch fused Triton decode conv1d + optional SiLU/residual/clamp + in-place state roll.

    Specialized for ``T == 1, kernel_size == 4``; seamlessly handles multi-step
    decode (``T > 1``) or other kernel sizes via ``causal_dwconv1d_prefill_triton``.
    """
    if not x.is_cuda:
        out = causal_dwconv1d_ref(
            x, weight, conv_state, residual=residual, use_activation=use_activation
        )
        return out.clamp(-clamp_val, clamp_val) if clamp_val is not None else out
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, T, C], got {tuple(x.shape)}")

    B, T, C = x.shape
    kernel_size = int(weight.shape[-1])
    if T != 1 or kernel_size != 4:
        out = causal_dwconv1d_prefill_triton(
            x, weight, conv_state, residual=residual, use_activation=use_activation
        )
        return out.clamp(-clamp_val, clamp_val) if clamp_val is not None else out

    if tuple(conv_state.shape) != (B, C, 3):
        raise ValueError(
            f"conv_state shape {tuple(conv_state.shape)} != {(B, C, 3)}"
        )
    if not conv_state.is_contiguous():
        raise ValueError("conv_state must be contiguous for in-place Triton update")

    x_c = x.contiguous()
    w_c = weight.contiguous()
    out = torch.empty_like(x_c)
    if B == 0:
        return out

    block_c = 256 if C <= 512 else 512
    num_warps = 4 if block_c <= 256 else 8
    _causal_dwconv1d_decode_k4_kernel[(B, triton.cdiv(C, block_c))](
        x_c,
        w_c,
        conv_state,
        out,
        C=C,
        RESIDUAL=residual,
        USE_SILU=use_activation,
        ROUND_PRE_ACT=round_pre_act,
        HAS_CLAMP=clamp_val is not None,
        CLAMP_VAL=float(clamp_val) if clamp_val is not None else 0.0,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    return out


def causal_dwconv1d_verify_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    init_state: torch.Tensor,
    stage_states: Optional[torch.Tensor] = None,
    *,
    residual: bool = False,
    use_activation: bool = False,
    round_pre_act: bool = False,
    clamp_val: Optional[float] = None,
    update_init_inplace: bool = False,
) -> torch.Tensor:
    """Single-launch fused K=4 conv1d over V verify steps keeping state in registers.

    Args:
        x: Input tensor ``[B, V, C]``.
        weight: Depthwise conv weight ``[C, 1, 4]``.
        init_state: Initial rolling state ``[B, C, 3]``.
        stage_states: Optional preallocated rollback buffer ``[B, V, C, 3]`` (arbitrary strides).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, V, C], got {tuple(x.shape)}")
    B, V, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    kernel_size = int(weight.shape[-1])
    if (
        not x.is_cuda
        or kernel_size != 4
        or V <= 0
        or V > 16
        or x.stride(-1) != 1
    ):
        state_cur = init_state.clone() if not update_init_inplace else init_state
        steps = []
        for step in range(V):
            y_s = causal_dwconv1d_decode_triton(
                x[:, step : step + 1],
                weight,
                state_cur,
                residual=residual,
                use_activation=use_activation,
                round_pre_act=round_pre_act,
                clamp_val=clamp_val,
            )
            if stage_states is not None:
                stage_states[:, step].copy_(state_cur)
            steps.append(y_s)
        return torch.cat(steps, dim=1) if V > 0 else torch.empty_like(x)

    w_c = weight.contiguous()
    out = torch.empty((B, V, C), dtype=x.dtype, device=x.device)
    if B == 0:
        return out

    has_stage = stage_states is not None
    st_ptr = stage_states if has_stage else out
    block_c = 256 if C <= 512 else 512
    num_warps = 4 if block_c <= 256 else 8
    _causal_dwconv1d_verify_k4_kernel[(B, triton.cdiv(C, block_c))](
        x,
        w_c,
        init_state,
        st_ptr,
        out,
        int(x.stride(0)),
        int(x.stride(1)),
        int(x.stride(2)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        int(init_state.stride(0)),
        int(init_state.stride(1)),
        int(init_state.stride(2)),
        int(stage_states.stride(0)) if has_stage else 0,
        int(stage_states.stride(1)) if has_stage else 0,
        int(stage_states.stride(2)) if has_stage else 0,
        int(stage_states.stride(3)) if has_stage else 0,
        C=C,
        V=V,
        RESIDUAL=residual,
        USE_SILU=use_activation,
        ROUND_PRE_ACT=round_pre_act,
        HAS_STAGE=has_stage,
        UPDATE_INIT=update_init_inplace,
        HAS_CLAMP=clamp_val is not None,
        CLAMP_VAL=float(clamp_val) if clamp_val is not None else 0.0,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
    return out


def causal_dwconv1d_ragged_prefill_with_snapshots_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    tile_req: torch.Tensor,
    tile_t0: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
    *,
    snap_tok_idx: Optional[torch.Tensor] = None,
    snap_req_idx: Optional[torch.Tensor] = None,
    dst_pool: Optional[torch.Tensor] = None,
    snap_dst_idx: Optional[torch.Tensor] = None,
    dst_stride_row: int = 0,
    dst_col_offset: int = 0,
    skip_invalid_phys: bool = True,
    residual: bool = False,
    use_activation: bool = False,
    round_pre_act: bool = False,
    clamp_val: Optional[float] = None,
    block_t: int = 32,
) -> torch.Tensor:
    """Packed ragged causal depthwise conv1d prefill with optional boundary state snapshots.

    Args:
        x: Packed input ``[total_tokens, C]``.
        weight: Depthwise conv weight ``[C, 1, K]``.
        cu_seqlens: Cumulative request lengths ``[B + 1]`` int32/int64 on CUDA.
        tile_req: Per-tile request id ``[num_tiles]`` int32/int64 on CUDA.
        tile_t0: Per-tile local token start ``[num_tiles]`` int32/int64 on CUDA.
        init_state: Optional per-request prefix state ``[B, C, K - 1]``.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected packed 2D x [N, C], got {tuple(x.shape)}")
    N, C = int(x.shape[0]), int(x.shape[1])
    kernel_size = int(weight.shape[-1])
    state_len = kernel_size - 1
    out = torch.empty((N, C), dtype=x.dtype, device=x.device)
    if N == 0:
        return out

    if not x.is_cuda:
        cu_list = cu_seqlens.tolist()
        for b in range(len(cu_list) - 1):
            s, e = int(cu_list[b]), int(cu_list[b + 1])
            if e <= s:
                continue
            st_b = init_state[b : b + 1].clone() if init_state is not None else None
            y_b = causal_dwconv1d_ref(
                x[s:e].unsqueeze(0),
                weight,
                st_b,
                residual=residual,
                use_activation=use_activation,
            )
            if clamp_val is not None:
                y_b = y_b.clamp(-clamp_val, clamp_val)
            out[s:e].copy_(y_b.squeeze(0))
        if (
            snap_tok_idx is not None
            and snap_req_idx is not None
            and dst_pool is not None
            and snap_dst_idx is not None
            and state_len > 0
        ):
            row_st = int(dst_stride_row) if dst_stride_row > 0 else int(dst_pool.stride(0))
            dst_flat = dst_pool.view(-1)
            for idx_s in range(int(snap_tok_idx.numel())):
                phys = int(snap_dst_idx[idx_s].item())
                if skip_invalid_phys and (phys <= 0 or phys >= int(dst_pool.shape[0])):
                    continue
                g_tok = int(snap_tok_idx[idx_s].item())
                b = int(snap_req_idx[idx_s].item())
                req_start = int(cu_list[b])
                local_t = g_tok - req_start
                cols = []
                for j in range(state_len):
                    src_local = local_t - (state_len - 1) + j
                    if src_local >= 0:
                        cols.append(x[req_start + src_local])
                    elif init_state is not None:
                        cols.append(init_state[b, :, src_local + state_len])
                    else:
                        cols.append(torch.zeros((C,), dtype=x.dtype, device=x.device))
                st_b = torch.stack(cols, dim=-1).to(dst_pool.dtype).reshape(-1)
                off = phys * row_st + int(dst_col_offset)
                dst_flat[off : off + st_b.numel()].copy_(st_b)
        return out

    w_c = weight.contiguous()
    num_tiles = int(tile_req.numel())
    if num_tiles > 0:
        has_init = init_state is not None and state_len > 0
        init_arg = init_state if has_init else out
        if kernel_size == 4:
            block_c = min(512, max(64, triton.next_power_of_2(C)))
            if C <= 256 and block_t % 4 == 0:
                sub_num = 4
            elif block_t % 2 == 0:
                sub_num = 2
            else:
                sub_num = 1
            sub_t = block_t // sub_num
            contig_c = int(x.stride(1)) == 1 and int(out.stride(1)) == 1
            _causal_dwconv1d_ragged_prefill_k4_kernel[(num_tiles * sub_num, triton.cdiv(C, block_c))](
                x,
                w_c,
                init_arg,
                out,
                cu_seqlens,
                tile_req,
                tile_t0,
                int(x.stride(0)),
                int(x.stride(1)),
                int(out.stride(0)),
                int(out.stride(1)),
                int(init_state.stride(0)) if has_init else 0,
                int(init_state.stride(1)) if has_init else 0,
                int(init_state.stride(2)) if has_init else 0,
                C=C,
                CONTIG_C=contig_c,
                HAS_INIT=has_init,
                RESIDUAL=residual,
                USE_SILU=use_activation,
                ROUND_PRE_ACT=round_pre_act,
                HAS_CLAMP=clamp_val is not None,
                CLAMP_VAL=float(clamp_val) if clamp_val is not None else 0.0,
                SUB_T=sub_t,
                SUB_NUM=sub_num,
                BLOCK_C=block_c,
                num_warps=4,
            )
        else:
            block_c = 64 if C <= 256 else 128
            _causal_dwconv1d_ragged_prefill_kernel[(num_tiles, triton.cdiv(C, block_c))](
                x,
                w_c,
                init_arg,
                out,
                cu_seqlens,
                tile_req,
                tile_t0,
                int(x.stride(0)),
                int(x.stride(1)),
                int(out.stride(0)),
                int(out.stride(1)),
                int(init_state.stride(0)) if has_init else 0,
                int(init_state.stride(1)) if has_init else 0,
                int(init_state.stride(2)) if has_init else 0,
                C=C,
                KERNEL_SIZE=kernel_size,
                STATE_LEN=state_len,
                HAS_INIT=has_init,
                RESIDUAL=residual,
                USE_SILU=use_activation,
                ROUND_PRE_ACT=round_pre_act,
                HAS_CLAMP=clamp_val is not None,
                CLAMP_VAL=float(clamp_val) if clamp_val is not None else 0.0,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
                num_warps=4,
            )

    if (
        snap_tok_idx is not None
        and snap_req_idx is not None
        and dst_pool is not None
        and snap_dst_idx is not None
        and state_len > 0
    ):
        num_snaps = int(snap_tok_idx.numel())
        if num_snaps > 0:
            has_init = init_state is not None
            init_arg = init_state if has_init else out
            upd_block_c = 256 if C <= 512 else 512
            row_stride = int(dst_stride_row) if dst_stride_row > 0 else int(dst_pool.stride(0))
            _scatter_conv_snapshots_ragged_kernel[(num_snaps, triton.cdiv(C, upd_block_c))](
                x,
                init_arg,
                cu_seqlens,
                snap_tok_idx,
                snap_req_idx,
                snap_dst_idx,
                dst_pool,
                int(x.stride(0)),
                int(x.stride(1)),
                int(init_state.stride(0)) if has_init else 0,
                int(init_state.stride(1)) if has_init else 0,
                int(init_state.stride(2)) if has_init else 0,
                row_stride,
                int(dst_col_offset),
                int(dst_pool.shape[0]),
                C=C,
                STATE_LEN=state_len,
                HAS_INIT=has_init,
                SKIP_INVALID_PHYS=skip_invalid_phys,
                BLOCK_C=upd_block_c,
                num_warps=4,
            )
    return out


def causal_dwconv1d_prefill_with_snapshots_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
    *,
    seq_size_per_block: int = 64,
    prefix_lengths: Optional[Sequence[int]] = None,
    residual: bool = False,
    use_activation: bool = False,
    round_pre_act: bool = False,
    clamp_val: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-pass full-sequence conv1d returning ``(out [B, T, C], boundary_states [num_snaps, C, K-1])``."""
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, T, C], got {tuple(x.shape)}")
    B, T, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    kernel_size = int(weight.shape[-1])
    state_len = kernel_size - 1
    if T == 0 or B == 0:
        return (
            torch.empty_like(x),
            torch.empty((0, C, state_len), dtype=x.dtype, device=x.device),
        )
    pfx = [0] * B if prefix_lengths is None else [int(v) for v in prefix_lengths]
    dev = x.device
    block_t = 16 if (T <= 32 or C >= 4096) else 32
    cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=dev)
    t_reqs: list[int] = []
    t_t0s: list[int] = []
    s_toks: list[int] = []
    s_reqs: list[int] = []
    for b in range(B):
        for t0 in range(0, T, block_t):
            t_reqs.append(b)
            t_t0s.append(t0)
        tp = pfx[b]
        for t in range(T):
            abs_pos = tp + t
            if (abs_pos + 1) % seq_size_per_block == 0 or t == T - 1:
                s_toks.append(b * T + t)
                s_reqs.append(b)
    tile_req = torch.tensor(t_reqs, dtype=torch.int32, device=dev)
    tile_t0 = torch.tensor(t_t0s, dtype=torch.int32, device=dev)
    snap_tok = torch.tensor(s_toks, dtype=torch.int32, device=dev)
    snap_req = torch.tensor(s_reqs, dtype=torch.int32, device=dev)
    num_snaps = len(s_toks)
    snap_out = torch.empty((num_snaps, C, state_len), dtype=x.dtype, device=dev)
    snap_dst = torch.arange(num_snaps, dtype=torch.int32, device=dev)
    out_2d = causal_dwconv1d_ragged_prefill_with_snapshots_triton(
        x.reshape(B * T, C),
        weight,
        cu,
        tile_req,
        tile_t0,
        init_state=init_state,
        snap_tok_idx=snap_tok,
        snap_req_idx=snap_req,
        dst_pool=snap_out.view(num_snaps, -1),
        snap_dst_idx=snap_dst,
        skip_invalid_phys=False,
        residual=residual,
        use_activation=use_activation,
        round_pre_act=round_pre_act,
        clamp_val=clamp_val,
        block_t=block_t,
    )
    return out_2d.view(B, T, C), snap_out


# ---------------------------------------------------------------------------
# Module: Qwen4_5ShortConv
# ---------------------------------------------------------------------------


class Qwen4_5ShortConv(nn.Module):
    """Causal depthwise 1D short convolution (default k=4) for Qwen 4.5-Flash."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        residual: bool = False,
        use_activation: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.residual = residual
        self.use_activation = use_activation
        self.activation = "silu" if use_activation else None

        self.weight = nn.Parameter(
            torch.empty(channels, 1, kernel_size, dtype=dtype, device=device)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def alloc_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> ShortConvState:
        dev = device if device is not None else self.weight.device
        dt = dtype if dtype is not None else self.weight.dtype
        return ShortConvState(
            conv_state=torch.zeros(
                (batch_size, self.channels, self.kernel_size - 1),
                device=dev,
                dtype=dt,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[ConvStateLike] = None,
    ) -> torch.Tensor:
        squeeze_batch = False
        if x.ndim == 2:
            if state is not None:
                raise ValueError("Stateful ShortConv requires 3D input [B, T, C]")
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {tuple(x.shape)}")

        conv_state = _extract_conv_state(state)
        if x.is_cuda:
            if conv_state is not None and x.shape[1] == 1 and self.kernel_size == 4:
                y = causal_dwconv1d_decode_triton(
                    x,
                    self.weight,
                    conv_state,
                    residual=self.residual,
                    use_activation=self.use_activation,
                )
            else:
                y = causal_dwconv1d_prefill_triton(
                    x,
                    self.weight,
                    conv_state,
                    residual=self.residual,
                    use_activation=self.use_activation,
                )
        else:
            y = causal_dwconv1d_ref(
                x,
                self.weight,
                conv_state,
                residual=self.residual,
                use_activation=self.use_activation,
            )

        return y.squeeze(0) if squeeze_batch else y

    def forward_ref(
        self,
        x: torch.Tensor,
        state: Optional[ConvStateLike] = None,
    ) -> torch.Tensor:
        """Pure-PyTorch FP32 reference forward (updates state in-place if provided)."""
        squeeze_batch = False
        if x.ndim == 2:
            if state is not None:
                raise ValueError("Stateful ShortConv requires 3D input [B, T, C]")
            x = x.unsqueeze(0)
            squeeze_batch = True
        conv_state = _extract_conv_state(state)
        y = causal_dwconv1d_ref(
            x,
            self.weight,
            conv_state,
            residual=self.residual,
            use_activation=self.use_activation,
        )
        return y.squeeze(0) if squeeze_batch else y
