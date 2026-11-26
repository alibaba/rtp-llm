"""Qwen 4.5-Flash Causal Depthwise Short-Conv (k=4) with Triton Prefill & Decode kernels.

Supports:
  - KV short-conv (channels=256, residual=True, use_activation=False)
  - Output short-conv (channels=5120, residual=True/False, use_activation=False/True)
  - PLE short-conv (channels=20480, residual=False, use_activation=False)
  - Stateful Prefill, Chunked Prefill, Single-Step Decode (T=1), and Multi-Step Decode (T>1)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .contracts import PLELayerState, ShortConvState

ConvStateLike = Union[ShortConvState, PLELayerState, torch.Tensor]

_SC_TRITON_CACHE: dict = {}


def _triton_int_spec(v: int) -> int:
    """Encode Triton integer specialization bucket: 0 for 1, 1 for %16==0, 2 otherwise."""
    if v == 1:
        return 0
    return 1 if (v & 15) == 0 else 2


def _has_active_hooks(hook: Any) -> bool:
    if hook is None:
        return False
    calls = getattr(hook, "calls", None)
    return bool(calls) if calls is not None else True


def triton_fast_launch(
    cache: dict,
    key: Any,
    jit_fn: Any,
    gx: int,
    gy: int,
    gz: int,
    ptr_or: int,
    runtime_args: Tuple[Any, ...],
    constexpr_kwargs: dict,
    *,
    num_warps: int = 4,
    num_stages: Optional[int] = None,
) -> None:
    """Bypass Triton Python JIT binder/specialize overhead via cached CompiledKernel.run."""
    knobs = triton.knobs.runtime
    if (
        knobs.debug
        or _has_active_hooks(knobs.launch_enter_hook)
        or _has_active_hooks(knobs.launch_exit_hook)
        or (ptr_or & 15) != 0
    ):
        kw = dict(constexpr_kwargs)
        kw["num_warps"] = num_warps
        if num_stages is not None:
            kw["num_stages"] = num_stages
        jit_fn[(gx, gy, gz)](*runtime_args, **kw)
        return

    entry = cache.get(key)
    if entry is None:
        kw = dict(constexpr_kwargs)
        kw["num_warps"] = num_warps
        if num_stages is not None:
            kw["num_stages"] = num_stages
        ck = jit_fn[(gx, gy, gz)](*runtime_args, **kw)
        run_obj = getattr(ck, "run", None)
        meta = getattr(ck, "metadata", None)
        if (
            run_obj is not None
            and getattr(meta, "global_scratch_size", 0) == 0
            and getattr(meta, "profile_scratch_size", 0) == 0
            and getattr(run_obj, "global_scratch_size", 0) == 0
            and getattr(run_obj, "profile_scratch_size", 0) == 0
        ):
            params = jit_fn.params
            n_rt = len(runtime_args)
            if len(params) >= n_rt and all(p.is_constexpr for p in params[n_rt:]):
                tail_vals = tuple(
                    constexpr_kwargs[p.name] if p.name in constexpr_kwargs else p.default
                    for p in params[n_rt:]
                )
                get_stream = triton.runtime.driver.active.get_current_stream
                dev_idx = key[0]
                cache[key] = (run_obj, ck.function, ck.packed_metadata, get_stream, dev_idx, tail_vals)
        return

    run_obj, fn, pm, get_stream, dev_idx, tail_vals = entry
    run_obj(gx, gy, gz, get_stream(dev_idx), fn, pm, None, None, None, *runtime_args, *tail_vals)



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
    HAS_CLAMP: tl.constexpr = False,
    CLAMP_VAL: tl.constexpr = 0.0,
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
    if HAS_CLAMP:
        acc = acc.to(OUT_PTR.dtype.element_ty).to(tl.float32)
        acc = tl.minimum(tl.maximum(acc, -CLAMP_VAL), CLAMP_VAL)

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
    stride_xb: tl.int64,
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
    x0_raw = tl.load(X_PTR + batch * stride_xb + c_offs, mask=c_mask, other=0.0)

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


@triton.jit(do_not_specialize=["TPB", "MAX_BLOCKS", "NUM_BLOCKS"])
def _causal_dwconv1d_decode_opaque_k4_kernel(
    X_PTR,
    W_PTR,
    POOL_PTR,
    OUT_PTR,
    BT_PTR,
    SEQ_BEFORE_PTR,
    stride_xb: tl.int64,
    stride_pb: tl.int64,
    stride_bt_b: tl.int64,
    stride_bt_p: tl.int64,
    TPB,
    MAX_BLOCKS,
    NUM_BLOCKS,
    POOL_COL_OFF: tl.int64,
    C: tl.constexpr,
    RESIDUAL: tl.constexpr,
    USE_SILU: tl.constexpr,
    ROUND_PRE_ACT: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    CLAMP_VAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Single-launch T=1 K=4 conv1d reading & writing paged opaque pool directly."""
    batch = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    c_mask = c_offs < C

    seq_b = tl.load(SEQ_BEFORE_PTR + batch).to(tl.int32)
    has_pfx = seq_b > 0
    in_blk = tl.where(has_pfx, (seq_b - 1) // TPB, 0)
    ok_in_blk = has_pfx & (in_blk < MAX_BLOCKS)
    in_phys = tl.load(
        BT_PTR + batch * stride_bt_b + tl.minimum(in_blk, MAX_BLOCKS - 1) * stride_bt_p,
        mask=ok_in_blk,
        other=0,
    ).to(tl.int64)
    ok_in = ok_in_blk & (in_phys > 0) & (in_phys < NUM_BLOCKS)
    in_base = POOL_PTR + in_phys * stride_pb + POOL_COL_OFF + c_offs * 3
    m_in = ok_in & c_mask
    s0_raw = tl.load(in_base + 0, mask=m_in, other=0.0)
    s1_raw = tl.load(in_base + 1, mask=m_in, other=0.0)
    s2_raw = tl.load(in_base + 2, mask=m_in, other=0.0)

    x0_raw = tl.load(X_PTR + batch * stride_xb + c_offs, mask=c_mask, other=0.0)

    w_base = c_offs * 4
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

    tl.store(OUT_PTR + batch * C + c_offs, acc, mask=c_mask)

    out_blk = seq_b // TPB
    ok_out_blk = (seq_b >= 0) & (out_blk < MAX_BLOCKS)
    out_phys = tl.load(
        BT_PTR + batch * stride_bt_b + tl.minimum(tl.maximum(out_blk, 0), MAX_BLOCKS - 1) * stride_bt_p,
        mask=ok_out_blk,
        other=0,
    ).to(tl.int64)
    ok_out = ok_out_blk & (out_phys > 0) & (out_phys < NUM_BLOCKS)
    out_base = POOL_PTR + out_phys * stride_pb + POOL_COL_OFF + c_offs * 3
    m_out = ok_out & c_mask
    tl.store(out_base + 0, s1_raw, mask=m_out)
    tl.store(out_base + 1, s2_raw, mask=m_out)
    tl.store(out_base + 2, x0_raw, mask=m_out)


@triton.jit
def _causal_dwconv1d_verify_k4_kernel(
    X_PTR,
    W_PTR,
    INIT_STATE_PTR,
    STAGE_STATE_PTR,
    OUT_PTR,
    BT_PTR,
    SEQ_BEFORE_PTR,
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
    stride_bt_b: tl.int64,
    stride_bt_p: tl.int64,
    TPB,
    MAX_BLOCKS,
    NUM_BLOCKS,
    POOL_COL_OFF: tl.int64,
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
    IS_OPAQUE_INIT: tl.constexpr = False,
):
    """Single-launch K=4 causal dwconv1d across V speculative verify steps in registers."""
    batch = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
    c_mask = c_offs < C

    if IS_OPAQUE_INIT:
        seq_b = tl.load(SEQ_BEFORE_PTR + batch)
        has_pfx = seq_b > 0
        blk_idx = tl.where(has_pfx, (seq_b - 1) // TPB, 0)
        in_bt = has_pfx & (blk_idx < MAX_BLOCKS)
        phys = tl.load(
            BT_PTR + batch * stride_bt_b + tl.minimum(blk_idx, MAX_BLOCKS - 1) * stride_bt_p,
            mask=in_bt,
            other=0,
        ).to(tl.int64)
        ok_init = in_bt & (phys > 0) & (phys < NUM_BLOCKS)
        s_base = INIT_STATE_PTR + phys * stride_sb + POOL_COL_OFF + c_offs * stride_sc
        m_s = ok_init & c_mask
    else:
        s_base = INIT_STATE_PTR + batch * stride_sb + c_offs * stride_sc
        m_s = c_mask
    s0_raw = tl.load(s_base + 0 * stride_sk, mask=m_s, other=0.0)
    s1_raw = tl.load(s_base + 1 * stride_sk, mask=m_s, other=0.0)
    s2_raw = tl.load(s_base + 2 * stride_sk, mask=m_s, other=0.0)

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
    REQ_SNAP_START_PTR,
    SNAP_ENDS_PTR,
    SNAP_DST_PTR,
    DST_PTR,
    dst_stride_row: tl.int64,
    dst_col_offset: tl.int64,
    dst_rows: tl.int64,
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
    FUSED_SNAP: tl.constexpr = False,
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

    if FUSED_SNAP:
        tile_end = t_start + tile_len
        rs = tl.load(REQ_SNAP_START_PTR + b).to(tl.int64)
        re = tl.load(REQ_SNAP_START_PTR + b + 1).to(tl.int64)
        lo = rs
        hi = re
        while lo < hi:
            mid = (lo + hi) >> 1
            v_end = tl.load(SNAP_ENDS_PTR + mid).to(tl.int64)
            if v_end < tile_end:
                lo = mid + 1
            else:
                hi = mid
        if lo < re:
            if tl.load(SNAP_ENDS_PTR + lo).to(tl.int64) == tile_end:
                dst_idx = tl.load(SNAP_DST_PTR + lo).to(tl.int64)
                if (dst_idx > 0) & (dst_idx < dst_rows):
                    dst_ty = DST_PTR.dtype.element_ty
                    dst_base = DST_PTR + dst_idx * dst_stride_row + dst_col_offset + c_offs.to(tl.int64) * 3
                    tl.store(dst_base + 0, s0.to(dst_ty), mask=c_mask)
                    tl.store(dst_base + 1, s1.to(dst_ty), mask=c_mask)
                    tl.store(dst_base + 2, s2.to(dst_ty), mask=c_mask)


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
    clamp_val: Optional[float] = None,
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
    out = y.transpose(1, 2).contiguous().to(x.dtype)
    if clamp_val is not None:
        out = out.clamp(-clamp_val, clamp_val)
    return out


def causal_dwconv1d_prefill_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: Optional[torch.Tensor] = None,
    *,
    residual: bool = False,
    use_activation: bool = False,
    clamp_val: Optional[float] = None,
) -> torch.Tensor:
    """Fused Triton causal depthwise conv1d for Prefill / multi-step input [B, T, C].

    If ``conv_state`` ([B, C, K-1]) is provided, it is consumed as left-context
    and updated in-place with the last ``K-1`` tokens of ``[conv_state, x]``.
    """
    if not x.is_cuda:
        return causal_dwconv1d_ref(
            x,
            weight,
            conv_state,
            residual=residual,
            use_activation=use_activation,
            clamp_val=clamp_val,
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
        HAS_CLAMP=clamp_val is not None,
        CLAMP_VAL=float(clamp_val) if clamp_val is not None else 0.0,
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
        return causal_dwconv1d_prefill_triton(
            x,
            weight,
            conv_state,
            residual=residual,
            use_activation=use_activation,
            clamp_val=clamp_val,
        )

    if tuple(conv_state.shape) != (B, C, 3):
        raise ValueError(
            f"conv_state shape {tuple(conv_state.shape)} != {(B, C, 3)}"
        )
    if not conv_state.is_contiguous():
        raise ValueError("conv_state must be contiguous for in-place Triton update")

    x_c = x if x.stride(-1) == 1 else x.contiguous()
    w_c = weight if weight.is_contiguous() else weight.contiguous()
    out = torch.empty((B, T, C), dtype=x_c.dtype, device=x_c.device)
    if B == 0:
        return out

    block_c = 256 if C <= 512 else 512
    num_warps = 4 if block_c <= 256 else 8
    sxb = int(x_c.stride(0))
    has_c = clamp_val is not None
    cval = float(clamp_val) if has_c else 0.0
    ckey = (
        x_c.device.index,
        0,
        x_c.dtype,
        w_c.dtype,
        conv_state.dtype,
        _triton_int_spec(sxb),
        C,
        residual,
        use_activation,
        round_pre_act,
        has_c,
        cval,
        block_c,
        num_warps,
    )
    triton_fast_launch(
        _SC_TRITON_CACHE,
        ckey,
        _causal_dwconv1d_decode_k4_kernel,
        B,
        triton.cdiv(C, block_c),
        1,
        x_c.data_ptr() | w_c.data_ptr() | conv_state.data_ptr() | out.data_ptr(),
        (x_c, w_c, conv_state, out, sxb),
        {
            "C": C,
            "RESIDUAL": residual,
            "USE_SILU": use_activation,
            "ROUND_PRE_ACT": round_pre_act,
            "HAS_CLAMP": has_c,
            "CLAMP_VAL": cval,
            "BLOCK_C": block_c,
        },
        num_warps=num_warps,
    )
    return out


def causal_dwconv1d_decode_opaque_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    pool_2d: torch.Tensor,
    block_table: torch.Tensor,
    seq_before: torch.Tensor,
    *,
    page_size: int,
    pool_col_offset: int = 0,
    residual: bool = False,
    use_activation: bool = False,
    round_pre_act: bool = False,
    clamp_val: Optional[float] = None,
) -> torch.Tensor:
    """Single-launch K=4 T=1 decode conv1d reading & updating paged opaque pool directly."""
    B, T, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if T != 1:
        raise ValueError(f"causal_dwconv1d_decode_opaque_triton expects T=1, got T={T}")
    x_c = x if x.stride(-1) == 1 else x.contiguous()
    w_c = weight if weight.is_contiguous() else weight.contiguous()
    out = torch.empty((B, 1, C), dtype=x_c.dtype, device=x_c.device)
    if B == 0:
        return out
    block_c = 256 if C <= 512 else 512
    num_warps = 4 if block_c <= 256 else 8
    sxb = int(x_c.stride(0))
    spb = int(pool_2d.stride(0))
    sbt0 = int(block_table.stride(0))
    sbt1 = int(block_table.stride(1))
    pco = int(pool_col_offset)
    has_c = clamp_val is not None
    cval = float(clamp_val) if has_c else 0.0
    ckey = (
        x_c.device.index,
        1,
        x_c.dtype,
        w_c.dtype,
        pool_2d.dtype,
        block_table.dtype,
        seq_before.dtype,
        _triton_int_spec(sxb),
        _triton_int_spec(spb),
        _triton_int_spec(sbt0),
        _triton_int_spec(sbt1),
        _triton_int_spec(pco),
        C,
        residual,
        use_activation,
        round_pre_act,
        has_c,
        cval,
        block_c,
        num_warps,
    )
    triton_fast_launch(
        _SC_TRITON_CACHE,
        ckey,
        _causal_dwconv1d_decode_opaque_k4_kernel,
        B,
        triton.cdiv(C, block_c),
        1,
        x_c.data_ptr() | w_c.data_ptr() | pool_2d.data_ptr() | out.data_ptr() | block_table.data_ptr() | seq_before.data_ptr(),
        (
            x_c,
            w_c,
            pool_2d,
            out,
            block_table,
            seq_before,
            sxb,
            spb,
            sbt0,
            sbt1,
            int(page_size),
            int(block_table.shape[1]),
            int(pool_2d.shape[0]),
            pco,
        ),
        {
            "C": C,
            "RESIDUAL": residual,
            "USE_SILU": use_activation,
            "ROUND_PRE_ACT": round_pre_act,
            "HAS_CLAMP": has_c,
            "CLAMP_VAL": cval,
            "BLOCK_C": block_c,
        },
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

    return _launch_verify_k4_kernel(
        x,
        w_c,
        init_state,
        stage_states,
        out,
        stride_sb=int(init_state.stride(0)),
        stride_sc=int(init_state.stride(1)),
        stride_sk=int(init_state.stride(2)),
        residual=residual,
        use_activation=use_activation,
        round_pre_act=round_pre_act,
        clamp_val=clamp_val,
        update_init_inplace=update_init_inplace,
        is_opaque_init=False,
    )


def _launch_verify_k4_kernel(
    x: torch.Tensor,
    w_c: torch.Tensor,
    init_or_pool: torch.Tensor,
    stage_states: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    stride_sb: int,
    stride_sc: int,
    stride_sk: int,
    residual: bool,
    use_activation: bool,
    round_pre_act: bool,
    clamp_val: Optional[float],
    update_init_inplace: bool,
    is_opaque_init: bool,
    block_table: Optional[torch.Tensor] = None,
    seq_before: Optional[torch.Tensor] = None,
    page_size: int = 1,
    pool_col_offset: int = 0,
) -> torch.Tensor:
    B, V, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    has_stage = stage_states is not None
    st_ptr = stage_states if has_stage else out
    bt_ptr = block_table if block_table is not None else x
    seq_ptr = seq_before if seq_before is not None else x
    block_c = 128 if C > 512 else 256
    rt_args = (
        x,
        w_c,
        init_or_pool,
        st_ptr,
        out,
        bt_ptr,
        seq_ptr,
        int(x.stride(0)),
        int(x.stride(1)),
        int(x.stride(2)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        stride_sb,
        stride_sc,
        stride_sk,
        int(stage_states.stride(0)) if has_stage and stage_states is not None else 0,
        int(stage_states.stride(1)) if has_stage and stage_states is not None else 0,
        int(stage_states.stride(2)) if has_stage and stage_states is not None else 0,
        int(stage_states.stride(3)) if has_stage and stage_states is not None else 0,
        int(block_table.stride(0)) if block_table is not None else 0,
        int(block_table.stride(1)) if block_table is not None else 0,
        int(page_size),
        int(block_table.shape[1]) if block_table is not None and block_table.ndim >= 2 else 0,
        int(init_or_pool.shape[0]) if is_opaque_init else 0,
        int(pool_col_offset),
    )
    has_c = clamp_val is not None
    cval = float(clamp_val) if has_c else 0.0
    ckey = (
        x.device.index,
        2,
        x.dtype,
        w_c.dtype,
        init_or_pool.dtype,
        st_ptr.dtype,
        out.dtype,
        bt_ptr.dtype,
        seq_ptr.dtype,
        tuple(_triton_int_spec(v) for v in rt_args[7:]),
        C,
        V,
        residual,
        use_activation,
        round_pre_act,
        has_stage,
        update_init_inplace,
        has_c,
        cval,
        block_c,
        is_opaque_init,
    )
    ptr_or = (
        x.data_ptr()
        | w_c.data_ptr()
        | init_or_pool.data_ptr()
        | st_ptr.data_ptr()
        | out.data_ptr()
        | bt_ptr.data_ptr()
        | seq_ptr.data_ptr()
    )
    triton_fast_launch(
        _SC_TRITON_CACHE,
        ckey,
        _causal_dwconv1d_verify_k4_kernel,
        B,
        triton.cdiv(C, block_c),
        1,
        ptr_or,
        rt_args,
        {
            "C": C,
            "V": V,
            "RESIDUAL": residual,
            "USE_SILU": use_activation,
            "ROUND_PRE_ACT": round_pre_act,
            "HAS_STAGE": has_stage,
            "UPDATE_INIT": update_init_inplace,
            "HAS_CLAMP": has_c,
            "CLAMP_VAL": cval,
            "BLOCK_C": block_c,
            "IS_OPAQUE_INIT": is_opaque_init,
        },
        num_warps=4,
    )
    return out


def causal_dwconv1d_verify_opaque_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    pool_2d: torch.Tensor,
    block_table: torch.Tensor,
    seq_before: torch.Tensor,
    stage_states: Optional[torch.Tensor] = None,
    *,
    page_size: int,
    pool_col_offset: int = 0,
    residual: bool = False,
    use_activation: bool = False,
    round_pre_act: bool = False,
    clamp_val: Optional[float] = None,
) -> torch.Tensor:
    """Single-launch K=4 verify conv1d reading init_state directly from opaque pool."""
    if x.ndim != 3:
        raise ValueError(f"Expected x of shape [B, V, C], got {tuple(x.shape)}")
    B, V, C = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if (
        not x.is_cuda
        or int(weight.shape[-1]) != 4
        or V <= 0
        or V > 16
        or x.stride(-1) != 1
        or pool_2d.ndim != 2
        or pool_2d.dtype != x.dtype
        or pool_2d.stride(1) != 1
        or page_size <= 0
    ):
        raise ValueError("Unsupported configuration for direct-opaque ShortConv verify kernel")

    out = torch.empty((B, V, C), dtype=x.dtype, device=x.device)
    if B == 0:
        return out

    return _launch_verify_k4_kernel(
        x,
        weight.contiguous(),
        pool_2d,
        stage_states,
        out,
        stride_sb=int(pool_2d.stride(0)),
        stride_sc=3,
        stride_sk=1,
        residual=residual,
        use_activation=use_activation,
        round_pre_act=round_pre_act,
        clamp_val=clamp_val,
        update_init_inplace=False,
        is_opaque_init=True,
        block_table=block_table,
        seq_before=seq_before,
        page_size=page_size,
        pool_col_offset=pool_col_offset,
    )


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
    req_snap_start: Optional[torch.Tensor] = None,
    snap_local_ends: Optional[torch.Tensor] = None,
    snap_aligned: bool = False,
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
    fused_snap = False
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
            fused_snap = bool(
                snap_aligned
                and skip_invalid_phys
                and snap_dst_idx is not None
                and dst_pool is not None
                and req_snap_start is not None
                and snap_local_ends is not None
                and int(snap_dst_idx.numel()) > 0
                and int(dst_pool.stride(-1)) == 1
            )
            r_ss_arg = req_snap_start if fused_snap else out
            s_end_arg = snap_local_ends if fused_snap else out
            s_dst_arg = snap_dst_idx if fused_snap else out
            d_pool_arg = dst_pool if fused_snap else out
            d_row_st = (int(dst_stride_row) if dst_stride_row > 0 else int(dst_pool.stride(0))) if fused_snap else 0
            d_col_off = int(dst_col_offset) if fused_snap else 0
            d_rows = int(dst_pool.shape[0]) if fused_snap else 0
            rt_args = (
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
                r_ss_arg,
                s_end_arg,
                s_dst_arg,
                d_pool_arg,
                d_row_st,
                d_col_off,
                d_rows,
            )
            has_c = clamp_val is not None
            cval = float(clamp_val) if has_c else 0.0
            ckey = (
                x.device.index,
                3,
                x.dtype,
                w_c.dtype,
                init_arg.dtype,
                cu_seqlens.dtype,
                tile_req.dtype,
                tile_t0.dtype,
                r_ss_arg.dtype,
                s_end_arg.dtype,
                d_pool_arg.dtype,
                s_dst_arg.dtype,
                tuple(_triton_int_spec(v) for v in (rt_args[7:14] + rt_args[18:])),
                C,
                contig_c,
                has_init,
                residual,
                use_activation,
                round_pre_act,
                has_c,
                cval,
                sub_t,
                sub_num,
                block_c,
                fused_snap,
            )
            ptr_or = (
                x.data_ptr()
                | w_c.data_ptr()
                | init_arg.data_ptr()
                | out.data_ptr()
                | cu_seqlens.data_ptr()
                | tile_req.data_ptr()
                | tile_t0.data_ptr()
                | r_ss_arg.data_ptr()
                | s_end_arg.data_ptr()
                | s_dst_arg.data_ptr()
                | d_pool_arg.data_ptr()
            )
            triton_fast_launch(
                _SC_TRITON_CACHE,
                ckey,
                _causal_dwconv1d_ragged_prefill_k4_kernel,
                num_tiles * sub_num,
                triton.cdiv(C, block_c),
                1,
                ptr_or,
                rt_args,
                {
                    "C": C,
                    "CONTIG_C": contig_c,
                    "HAS_INIT": has_init,
                    "RESIDUAL": residual,
                    "USE_SILU": use_activation,
                    "ROUND_PRE_ACT": round_pre_act,
                    "HAS_CLAMP": has_c,
                    "CLAMP_VAL": cval,
                    "SUB_T": sub_t,
                    "SUB_NUM": sub_num,
                    "BLOCK_C": block_c,
                    "FUSED_SNAP": fused_snap,
                },
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

    if kernel_size == 4 and num_tiles > 0 and fused_snap:
        return out

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
        *,
        sp_size: int = 1,
        sp_rank: int = 0,
        sp_group: Any = None,
        dp_attn: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.residual = residual
        self.use_activation = use_activation
        self.activation = "silu" if use_activation else None
        self.sp_size = max(1, int(sp_size))
        self.sp_rank = int(sp_rank)
        self.sp_group = sp_group
        self.dp_attn = bool(dp_attn)

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

    def exchange_sp_left_halo(
        self,
        x_local: torch.Tensor,
        *,
        init_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exchange ``(kernel_size - 1)`` left-boundary tail tokens across SP ranks."""
        w_ctx = self.kernel_size - 1
        if x_local.ndim == 2:
            x_3d = x_local.unsqueeze(0)
        elif x_local.ndim == 3:
            x_3d = x_local
        else:
            raise ValueError(f"Expected 2D or 3D x_local, got {tuple(x_local.shape)}")
        bsz, t_loc, ch = x_3d.shape
        if w_ctx == 0:
            return torch.empty((bsz, ch, 0), dtype=x_3d.dtype, device=x_3d.device)

        if init_state is not None:
            left_halo = init_state.contiguous()
        else:
            left_halo = torch.zeros((bsz, ch, w_ctx), dtype=x_3d.dtype, device=x_3d.device)

        if self.sp_size <= 1 or self.sp_group is None or self.dp_attn:
            return left_halo

        import torch.distributed as dist

        if t_loc >= w_ctx:
            tail = x_3d[:, t_loc - w_ctx :, :].transpose(1, 2).contiguous()
        else:
            tail = torch.zeros((bsz, ch, w_ctx), dtype=x_3d.dtype, device=x_3d.device)
            if t_loc > 0:
                tail[:, :, w_ctx - t_loc :] = x_3d.transpose(1, 2)

        ops = []
        recv_buf = (
            torch.empty((bsz, ch, w_ctx), dtype=x_3d.dtype, device=x_3d.device)
            if self.sp_rank > 0
            else None
        )
        if self.sp_rank + 1 < self.sp_size:
            ops.append(dist.P2POp(dist.isend, tail, peer=self.sp_rank + 1, group=self.sp_group))
        if self.sp_rank > 0 and recv_buf is not None:
            ops.append(dist.P2POp(dist.irecv, recv_buf, peer=self.sp_rank - 1, group=self.sp_group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return recv_buf if recv_buf is not None else left_halo

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[ConvStateLike] = None,
        *,
        clip_value: Optional[float] = None,
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
                    clamp_val=clip_value,
                )
            else:
                y = causal_dwconv1d_prefill_triton(
                    x,
                    self.weight,
                    conv_state,
                    residual=self.residual,
                    use_activation=self.use_activation,
                    clamp_val=clip_value,
                )
        else:
            y = causal_dwconv1d_ref(
                x,
                self.weight,
                conv_state,
                residual=self.residual,
                use_activation=self.use_activation,
                clamp_val=clip_value,
            )

        return y.squeeze(0) if squeeze_batch else y

    def forward_ref(
        self,
        x: torch.Tensor,
        state: Optional[ConvStateLike] = None,
        *,
        clip_value: Optional[float] = None,
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
            clamp_val=clip_value,
        )
        return y.squeeze(0) if squeeze_batch else y
