"""Group-128 FP8 indexer scores for the pre-MX DeepSeek cache contract.

DeepGEMM's MX-only MQA API does not accept this cache or FP32 routing weights.
On CUDA SM89+ devices (when not running on PPU/HIP), the kernels keep FP8
operands in ``float8e4nv`` for native FP8 Tensor Core ``tl.dot`` with FP32
accumulation in a ``K @ Q^T`` orientation that avoids cross-warp head
reductions; PPU/HIP/pre-SM89 retain the ``tl.bfloat16`` operand cast. Scales
and routing weights remain FP32 throughout.
"""

from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

_NATIVE_FP8_DOT_CACHE: Dict[int, bool] = {}
_NUM_SMS_CACHE: Dict[int, int] = {}

_PREFILL_BN = 64
_PREFILL_MAX_SPLIT_KV = 2048
_PAGED_BN = 64


def is_supported(q, weights):
    return (
        q.is_cuda
        and q.dtype == torch.float8_e4m3fn
        and q.shape[-1] == 128
        and weights.dtype == torch.float32
    )


def _use_native_fp8_dot(device: torch.device) -> bool:
    """Return True when CUDA SM89+ supports native float8e4nv tl.dot -> fp32."""
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    if "ppu" in getattr(torch, "__version__", "").lower() or hasattr(torch, "ppu"):
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _NATIVE_FP8_DOT_CACHE.get(idx)
    if cached is not None:
        return cached
    try:
        from rtp_llm.device.device_type import is_cuda

        if not is_cuda():
            _NATIVE_FP8_DOT_CACHE[idx] = False
            return False
    except ImportError:
        pass
    supported = torch.cuda.get_device_capability(idx) >= (8, 9)
    _NATIVE_FP8_DOT_CACHE[idx] = supported
    return supported


def _get_num_sms(device: torch.device) -> int:
    idx = device.index if device.index is not None else 0
    cached = _NUM_SMS_CACHE.get(idx)
    if cached is None:
        try:
            cached = torch.cuda.get_device_properties(idx).multi_processor_count
        except Exception:
            cached = 108
        _NUM_SMS_CACHE[idx] = cached
    return cached


def _choose_split_kv(
    m_blocks: int,
    n: int,
    bn: int,
    target_ctas: int,
    max_split_kv: int,
) -> Tuple[int, int]:
    """Pick a BN-aligned K-split chunk size and CTA grid width along N."""
    min_splits = triton.cdiv(target_ctas, max(m_blocks, 1))
    raw_split = triton.cdiv(n, max(min_splits, 1))
    aligned = max(bn, triton.cdiv(raw_split, bn) * bn)
    split_kv = min(max_split_kv, aligned)
    num_splits = max(1, triton.cdiv(n, split_kv))
    return split_kv, num_splits


@triton.jit
def _fill_const_1d(
    out_ptr,
    start_n,
    end_n,
    n_total,
    n_base,
    val,
    BN: tl.constexpr,
):
    pad_vec = tl.full((BN,), val, dtype=tl.float32)
    for cur_n in range(start_n, end_n, BN):
        n_off = cur_n + n_base
        tl.store(out_ptr + n_off, pad_vec, mask=n_off < n_total)


@triton.jit
def _score_prefill_bq1(
    Q,
    K,
    SCALE,
    WEIGHTS,
    START,
    END,
    OUT,
    M,
    N,
    SPLIT_KV,
    Q_ROW,
    Q_HEAD,
    W_ROW,
    W_HEAD,
    H: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    MASK_H: tl.constexpr,
    CLEAN: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if row >= M:
        return
    row_i64 = row.to(tl.int64)
    split_lo = pid_n * SPLIT_KV
    split_hi = tl.minimum(split_lo + SPLIT_KV, N)

    start = tl.load(START + row)
    end = tl.load(END + row)
    eff_s = tl.maximum(start, 0)
    eff_e = tl.minimum(end, N)
    has_act = eff_s < eff_e
    k_lo = tl.maximum(split_lo, (eff_s // BN) * BN) if has_act else split_hi
    k_hi = tl.minimum(split_hi, eff_e) if has_act else split_hi

    pad_val = -float("inf") if CLEAN else 0.0
    n_base = tl.arange(0, BN)
    out_row = OUT + row_i64 * N

    if (not has_act) or (k_lo >= k_hi):
        _fill_const_1d(out_row, split_lo, split_hi, N, n_base, pad_val, BN)
        return

    if split_lo < k_lo:
        _fill_const_1d(out_row, split_lo, k_lo, N, n_base, pad_val, BN)

    h = tl.arange(0, BH)
    d = tl.arange(0, 128)
    q_base = Q + row_i64 * Q_ROW + h.to(tl.int64)[:, None] * Q_HEAD + d[None, :]
    w_base = WEIGHTS + row_i64 * W_ROW + h.to(tl.int64) * W_HEAD
    if MASK_H:
        h_ok = h < H
        q_tile = tl.load(q_base, mask=h_ok[:, None], other=0.0)
        w = tl.load(w_base, mask=h_ok, other=0.0).to(tl.float32)
    else:
        q_tile = tl.load(q_base)
        w = tl.load(w_base).to(tl.float32)
    if not USE_FP8_DOT:
        q_tile = q_tile.to(tl.bfloat16)
    q_t = tl.trans(q_tile)

    for cur_n in range(k_lo, k_hi, BN):
        n_off = cur_n + n_base
        n_ok = n_off < N
        valid = n_ok & (n_off >= start) & (n_off < end)
        k_tile = tl.load(
            K + n_off.to(tl.int64)[:, None] * 128 + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if not USE_FP8_DOT:
            k_tile = k_tile.to(tl.bfloat16)
        scale = tl.load(SCALE + n_off, mask=valid, other=0.0).to(tl.float32)
        dots = tl.dot(k_tile, q_t, out_dtype=tl.float32)
        scores = tl.sum(tl.maximum(dots, 0.0) * w[None, :], axis=1) * scale
        tl.store(out_row + n_off, tl.where(valid, scores, pad_val), mask=n_ok)

    k_done = k_lo + tl.cdiv(k_hi - k_lo, BN) * BN
    if k_done < split_hi:
        _fill_const_1d(out_row, k_done, split_hi, N, n_base, pad_val, BN)


@triton.jit
def _score_prefill_bq2(
    Q,
    K,
    SCALE,
    WEIGHTS,
    START,
    END,
    OUT,
    M,
    N,
    SPLIT_KV,
    Q_ROW,
    Q_HEAD,
    W_ROW,
    W_HEAD,
    H: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    MASK_H: tl.constexpr,
    CLEAN: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
):
    """Two-row sequential K @ Q^T prefill kernel keeping live state <= 128 regs."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    r0 = pid_m * 2
    r1 = r0 + 1
    m0_ok = r0 < M
    m1_ok = r1 < M
    split_lo = pid_n * SPLIT_KV
    split_hi = tl.minimum(split_lo + SPLIT_KV, N)

    s0 = tl.load(START + r0, mask=m0_ok, other=N)
    e0 = tl.load(END + r0, mask=m0_ok, other=0)
    s1 = tl.load(START + r1, mask=m1_ok, other=N)
    e1 = tl.load(END + r1, mask=m1_ok, other=0)
    es0 = tl.maximum(s0, 0)
    ee0 = tl.minimum(e0, N)
    es1 = tl.maximum(s1, 0)
    ee1 = tl.minimum(e1, N)
    a0 = m0_ok & (es0 < ee0)
    a1 = m1_ok & (es1 < ee1)
    has_any = a0 | a1
    min_s = tl.minimum(tl.where(a0, es0, N), tl.where(a1, es1, N))
    max_e = tl.maximum(tl.where(a0, ee0, 0), tl.where(a1, ee1, 0))
    k_lo = tl.maximum(split_lo, (min_s // BN) * BN) if has_any else split_hi
    k_hi = tl.minimum(split_hi, max_e) if has_any else split_hi

    pad_val = -float("inf") if CLEAN else 0.0
    n_base = tl.arange(0, BN)
    out0 = OUT + r0.to(tl.int64) * N
    out1 = OUT + r1.to(tl.int64) * N

    if (not has_any) or (k_lo >= k_hi):
        if m0_ok:
            _fill_const_1d(out0, split_lo, split_hi, N, n_base, pad_val, BN)
        if m1_ok:
            _fill_const_1d(out1, split_lo, split_hi, N, n_base, pad_val, BN)
        return

    if split_lo < k_lo:
        if m0_ok:
            _fill_const_1d(out0, split_lo, k_lo, N, n_base, pad_val, BN)
        if m1_ok:
            _fill_const_1d(out1, split_lo, k_lo, N, n_base, pad_val, BN)

    h = tl.arange(0, BH)
    d = tl.arange(0, 128)
    h_q_off = h.to(tl.int64)[:, None] * Q_HEAD + d[None, :]
    h_w_off = h.to(tl.int64) * W_HEAD
    if MASK_H:
        h_ok = h < H
        q0 = tl.load(
            Q + r0.to(tl.int64) * Q_ROW + h_q_off, mask=m0_ok & h_ok[:, None], other=0.0
        )
        q1 = tl.load(
            Q + r1.to(tl.int64) * Q_ROW + h_q_off, mask=m1_ok & h_ok[:, None], other=0.0
        )
        w0 = tl.load(
            WEIGHTS + r0.to(tl.int64) * W_ROW + h_w_off, mask=m0_ok & h_ok, other=0.0
        ).to(tl.float32)
        w1 = tl.load(
            WEIGHTS + r1.to(tl.int64) * W_ROW + h_w_off, mask=m1_ok & h_ok, other=0.0
        ).to(tl.float32)
    else:
        q0 = tl.load(Q + r0.to(tl.int64) * Q_ROW + h_q_off, mask=m0_ok, other=0.0)
        q1 = tl.load(Q + r1.to(tl.int64) * Q_ROW + h_q_off, mask=m1_ok, other=0.0)
        w0 = tl.load(
            WEIGHTS + r0.to(tl.int64) * W_ROW + h_w_off, mask=m0_ok, other=0.0
        ).to(tl.float32)
        w1 = tl.load(
            WEIGHTS + r1.to(tl.int64) * W_ROW + h_w_off, mask=m1_ok, other=0.0
        ).to(tl.float32)
    if not USE_FP8_DOT:
        q0 = q0.to(tl.bfloat16)
        q1 = q1.to(tl.bfloat16)
    q0_t = tl.trans(q0)
    q1_t = tl.trans(q1)

    for cur_n in range(k_lo, k_hi, BN):
        n_off = cur_n + n_base
        n_ok = n_off < N
        col_act = n_ok & (n_off >= min_s) & (n_off < max_e)
        k_tile = tl.load(
            K + n_off.to(tl.int64)[:, None] * 128 + d[None, :],
            mask=col_act[:, None],
            other=0.0,
        )
        if not USE_FP8_DOT:
            k_tile = k_tile.to(tl.bfloat16)
        scale = tl.load(SCALE + n_off, mask=col_act, other=0.0).to(tl.float32)

        # Evaluate rows sequentially so d0's [BN, BH] accumulator dies before d1 starts.
        d0 = tl.dot(k_tile, q0_t, out_dtype=tl.float32)
        sc0 = tl.sum(tl.maximum(d0, 0.0) * w0[None, :], axis=1) * scale
        v0 = n_ok & (n_off >= s0) & (n_off < e0)
        if m0_ok:
            tl.store(out0 + n_off, tl.where(v0, sc0, pad_val), mask=n_ok)

        d1 = tl.dot(k_tile, q1_t, out_dtype=tl.float32)
        sc1 = tl.sum(tl.maximum(d1, 0.0) * w1[None, :], axis=1) * scale
        v1 = n_ok & (n_off >= s1) & (n_off < e1)
        if m1_ok:
            tl.store(out1 + n_off, tl.where(v1, sc1, pad_val), mask=n_ok)

    k_done = k_lo + tl.cdiv(k_hi - k_lo, BN) * BN
    if k_done < split_hi:
        if m0_ok:
            _fill_const_1d(out0, k_done, split_hi, N, n_base, pad_val, BN)
        if m1_ok:
            _fill_const_1d(out1, k_done, split_hi, N, n_base, pad_val, BN)


@triton.jit
def _load_paged_k_and_scale(
    K,
    TABLE,
    b,
    cur_n,
    n_off,
    n_base,
    d,
    max_len,
    TABLE_WIDTH,
    N_PAGES,
    PAGE: tl.constexpr,
    BN: tl.constexpr,
    SINGLE_PAGE: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
):
    """Load [BN, 128] FP8/BF16 keys and [BN] FP32 scales from the 132B paged pool."""
    if SINGLE_PAGE:
        logical_page = cur_n // PAGE
        page_in_bounds = logical_page < TABLE_WIDTH
        physical_page = tl.load(
            TABLE + b.to(tl.int64) * TABLE_WIDTH + logical_page,
            mask=page_in_bounds,
            other=-1,
        )
        page_valid = page_in_bounds & (physical_page >= 0) & (physical_page < N_PAGES)
        safe_page = tl.where(page_valid, physical_page, 0)
        page_off = (cur_n % PAGE) + n_base
        block = safe_page.to(tl.int64) * (PAGE * 132)
        col_ok = page_valid & (n_off < max_len)
        raw = tl.load(
            K + block + page_off.to(tl.int64)[:, None] * 128 + d[None, :],
            mask=col_ok[:, None],
            other=0,
        )
        scale = tl.load(
            (K + block + PAGE * 128 + page_off.to(tl.int64) * 4).to(
                tl.pointer_type(tl.float32)
            ),
            mask=col_ok,
            other=0.0,
        )
        token_valid = col_ok
    else:
        n_in = n_off < max_len
        logical_page = n_off // PAGE
        page_in_bounds = n_in & (logical_page < TABLE_WIDTH)
        physical_page = tl.load(
            TABLE + b.to(tl.int64) * TABLE_WIDTH + logical_page,
            mask=page_in_bounds,
            other=-1,
        )
        page_valid = page_in_bounds & (physical_page >= 0) & (physical_page < N_PAGES)
        safe_page = tl.where(page_valid, physical_page, 0)
        page_off = n_off % PAGE
        block = safe_page.to(tl.int64) * (PAGE * 132)
        raw = tl.load(
            K + block[:, None] + page_off.to(tl.int64)[:, None] * 128 + d[None, :],
            mask=page_valid[:, None],
            other=0,
        )
        scale = tl.load(
            (K + block + PAGE * 128 + page_off.to(tl.int64) * 4).to(
                tl.pointer_type(tl.float32)
            ),
            mask=page_valid,
            other=0.0,
        )
        token_valid = page_valid

    k_tile = raw.to(tl.float8e4nv, bitcast=True)
    if not USE_FP8_DOT:
        k_tile = k_tile.to(tl.bfloat16)
    return k_tile, scale, token_valid


@triton.jit
def _score_paged_bq1(
    Q,
    K,
    WEIGHTS,
    END,
    TABLE,
    OUT,
    N,
    SPLIT_KV,
    Q_ROW,
    Q_HEAD,
    W_ROW,
    W_HEAD,
    TABLE_WIDTH,
    N_PAGES,
    NEXT_N,
    H: tl.constexpr,
    BH: tl.constexpr,
    PAGE: tl.constexpr,
    BN: tl.constexpr,
    MASK_H: tl.constexpr,
    SINGLE_PAGE: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_i64 = row.to(tl.int64)
    b = row // NEXT_N
    split_lo = pid_n * SPLIT_KV
    split_hi = tl.minimum(split_lo + SPLIT_KV, N)

    raw_len = tl.load(END + row)
    max_len = tl.maximum(tl.minimum(raw_len, N), 0)
    n_base = tl.arange(0, BN)
    out_row = OUT + row_i64 * N

    if split_lo >= max_len:
        _fill_const_1d(out_row, split_lo, split_hi, N, n_base, 0.0, BN)
        return

    k_hi = tl.minimum(split_hi, max_len)
    h = tl.arange(0, BH)
    d = tl.arange(0, 128)
    q_base = Q + row_i64 * Q_ROW + h.to(tl.int64)[:, None] * Q_HEAD + d[None, :]
    w_base = WEIGHTS + row_i64 * W_ROW + h.to(tl.int64) * W_HEAD
    if MASK_H:
        h_ok = h < H
        q_tile = tl.load(q_base, mask=h_ok[:, None], other=0.0)
        w = tl.load(w_base, mask=h_ok, other=0.0).to(tl.float32)
    else:
        q_tile = tl.load(q_base)
        w = tl.load(w_base).to(tl.float32)
    if not USE_FP8_DOT:
        q_tile = q_tile.to(tl.bfloat16)
    q_t = tl.trans(q_tile)

    for cur_n in range(split_lo, k_hi, BN):
        n_off = cur_n + n_base
        k_tile, scale, token_valid = _load_paged_k_and_scale(
            K,
            TABLE,
            b,
            cur_n,
            n_off,
            n_base,
            d,
            max_len,
            TABLE_WIDTH,
            N_PAGES,
            PAGE,
            BN,
            SINGLE_PAGE,
            USE_FP8_DOT,
        )
        dots = tl.dot(k_tile, q_t, out_dtype=tl.float32)
        scores = tl.sum(tl.maximum(dots, 0.0) * w[None, :], axis=1) * scale
        tl.store(out_row + n_off, tl.where(token_valid, scores, 0.0), mask=n_off < N)

    k_done = split_lo + tl.cdiv(k_hi - split_lo, BN) * BN
    if k_done < split_hi:
        _fill_const_1d(out_row, k_done, split_hi, N, n_base, 0.0, BN)


@triton.jit
def _score_paged_bq2(
    Q,
    K,
    WEIGHTS,
    END,
    TABLE,
    OUT,
    N,
    SPLIT_KV,
    Q_ROW,
    Q_HEAD,
    W_ROW,
    W_HEAD,
    TABLE_WIDTH,
    N_PAGES,
    NEXT_N,
    H: tl.constexpr,
    BH: tl.constexpr,
    PAGE: tl.constexpr,
    BN: tl.constexpr,
    MASK_H: tl.constexpr,
    SINGLE_PAGE: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
):
    """Two-token sequential K @ Q^T paged decode kernel for MTP next_n >= 2."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pairs_per_req = tl.cdiv(NEXT_N, 2)
    b = pid_m // pairs_per_req
    lm0 = (pid_m % pairs_per_req) * 2
    lm1 = lm0 + 1
    m0_ok = lm0 < NEXT_N
    m1_ok = lm1 < NEXT_N
    r0 = b * NEXT_N + lm0
    r1 = b * NEXT_N + lm1
    split_lo = pid_n * SPLIT_KV
    split_hi = tl.minimum(split_lo + SPLIT_KV, N)

    l0 = tl.load(END + r0, mask=m0_ok, other=0)
    l1 = tl.load(END + r1, mask=m1_ok, other=0)
    el0 = tl.where(m0_ok, tl.maximum(tl.minimum(l0, N), 0), 0)
    el1 = tl.where(m1_ok, tl.maximum(tl.minimum(l1, N), 0), 0)
    max_len = tl.maximum(el0, el1)

    n_base = tl.arange(0, BN)
    out0 = OUT + r0.to(tl.int64) * N
    out1 = OUT + r1.to(tl.int64) * N

    if split_lo >= max_len:
        if m0_ok:
            _fill_const_1d(out0, split_lo, split_hi, N, n_base, 0.0, BN)
        if m1_ok:
            _fill_const_1d(out1, split_lo, split_hi, N, n_base, 0.0, BN)
        return

    k_hi = tl.minimum(split_hi, max_len)
    h = tl.arange(0, BH)
    d = tl.arange(0, 128)
    h_q_off = h.to(tl.int64)[:, None] * Q_HEAD + d[None, :]
    h_w_off = h.to(tl.int64) * W_HEAD
    if MASK_H:
        h_ok = h < H
        q0 = tl.load(
            Q + r0.to(tl.int64) * Q_ROW + h_q_off, mask=m0_ok & h_ok[:, None], other=0.0
        )
        q1 = tl.load(
            Q + r1.to(tl.int64) * Q_ROW + h_q_off, mask=m1_ok & h_ok[:, None], other=0.0
        )
        w0 = tl.load(
            WEIGHTS + r0.to(tl.int64) * W_ROW + h_w_off, mask=m0_ok & h_ok, other=0.0
        ).to(tl.float32)
        w1 = tl.load(
            WEIGHTS + r1.to(tl.int64) * W_ROW + h_w_off, mask=m1_ok & h_ok, other=0.0
        ).to(tl.float32)
    else:
        q0 = tl.load(Q + r0.to(tl.int64) * Q_ROW + h_q_off, mask=m0_ok, other=0.0)
        q1 = tl.load(Q + r1.to(tl.int64) * Q_ROW + h_q_off, mask=m1_ok, other=0.0)
        w0 = tl.load(
            WEIGHTS + r0.to(tl.int64) * W_ROW + h_w_off, mask=m0_ok, other=0.0
        ).to(tl.float32)
        w1 = tl.load(
            WEIGHTS + r1.to(tl.int64) * W_ROW + h_w_off, mask=m1_ok, other=0.0
        ).to(tl.float32)
    if not USE_FP8_DOT:
        q0 = q0.to(tl.bfloat16)
        q1 = q1.to(tl.bfloat16)
    q0_t = tl.trans(q0)
    q1_t = tl.trans(q1)

    for cur_n in range(split_lo, k_hi, BN):
        n_off = cur_n + n_base
        n_ok = n_off < N
        k_tile, scale, token_valid = _load_paged_k_and_scale(
            K,
            TABLE,
            b,
            cur_n,
            n_off,
            n_base,
            d,
            max_len,
            TABLE_WIDTH,
            N_PAGES,
            PAGE,
            BN,
            SINGLE_PAGE,
            USE_FP8_DOT,
        )

        d0 = tl.dot(k_tile, q0_t, out_dtype=tl.float32)
        sc0 = tl.sum(tl.maximum(d0, 0.0) * w0[None, :], axis=1) * scale
        v0 = token_valid & (n_off < el0)
        if m0_ok:
            tl.store(out0 + n_off, tl.where(v0, sc0, 0.0), mask=n_ok)

        d1 = tl.dot(k_tile, q1_t, out_dtype=tl.float32)
        sc1 = tl.sum(tl.maximum(d1, 0.0) * w1[None, :], axis=1) * scale
        v1 = token_valid & (n_off < el1)
        if m1_ok:
            tl.store(out1 + n_off, tl.where(v1, sc1, 0.0), mask=n_ok)

    k_done = split_lo + tl.cdiv(k_hi - split_lo, BN) * BN
    if k_done < split_hi:
        if m0_ok:
            _fill_const_1d(out0, k_done, split_hi, N, n_base, 0.0, BN)
        if m1_ok:
            _fill_const_1d(out1, k_done, split_hi, N, n_base, 0.0, BN)


def legacy_fp8_mqa_logits(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False, max_seqlen_k=0
):
    """Score the original FP8 K + one FP32 scale per 128-wide key format."""
    del max_seqlen_k
    k, scales = kv
    if q.dim() != 3 or k.dim() != 2 or q.shape[-1] != k.shape[-1]:
        raise ValueError("legacy MQA expects Q[M,H,D] and K[N,D]")
    if weights.shape != q.shape[:2] or scales.numel() != k.shape[0]:
        raise ValueError("legacy MQA weight or per-key scale shape mismatch")
    if q.shape[-1] != 128:
        raise ValueError(
            "group-128 legacy MQA requires a single 128-wide quantization group"
        )
    scales = scales.reshape(-1).contiguous()
    m, h, _ = q.shape
    n = k.shape[0]
    output = torch.empty((m, n), dtype=torch.float32, device=q.device)
    if m == 0 or n == 0:
        return output
    if not is_supported(q, weights):
        pad = -float("inf") if clean_logits else 0.0
        for begin in range(0, m, 16):
            end = min(m, begin + 16)
            dot = torch.einsum("mhd,nd->mhn", q[begin:end].float(), k.float())
            values = (dot.relu() * weights[begin:end].float().unsqueeze(-1)).sum(
                1
            ) * scales.float()
            columns = torch.arange(n, device=q.device)
            valid = (columns >= cu_seqlen_ks[begin:end, None]) & (
                columns < cu_seqlen_ke[begin:end, None]
            )
            output[begin:end] = values.masked_fill(~valid, pad)
        return output

    if q.stride(-1) != 1:
        q = q.contiguous()
    k = k.contiguous()
    ks = cu_seqlen_ks.contiguous()
    ke = cu_seqlen_ke.contiguous()

    bn = _PREFILL_BN
    bh = max(16, triton.next_power_of_2(h))
    mask_h = h != bh
    use_fp8_dot = _use_native_fp8_dot(q.device)
    num_sms = _get_num_sms(q.device)

    use_bq2 = (m >= 2) and ((m // 2) * triton.cdiv(n, bn) >= 6 * num_sms)
    m_blocks = triton.cdiv(m, 2) if use_bq2 else m
    split_kv, num_splits = _choose_split_kv(
        m_blocks,
        n,
        bn,
        target_ctas=10 * num_sms,
        max_split_kv=_PREFILL_MAX_SPLIT_KV,
    )
    kernel = _score_prefill_bq2 if use_bq2 else _score_prefill_bq1
    kernel[(m_blocks, num_splits)](
        q,
        k,
        scales,
        weights,
        ks,
        ke,
        output,
        m,
        n,
        split_kv,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        weights.stride(1),
        H=h,
        BH=bh,
        BN=bn,
        MASK_H=mask_h,
        CLEAN=clean_logits,
        USE_FP8_DOT=use_fp8_dot,
        num_warps=4,
        num_stages=2,
    )
    return output


def legacy_fp8_paged_mqa_logits(
    q, kv_cache, weights, context_lens, block_table, max_context_len
):
    """Paged score preserving 132-byte slots and full FP32 per-head weights."""
    if q.dim() != 4 or q.shape[-1] != 128:
        raise ValueError("legacy paged MQA expects Q[B,next_n,H,128]")
    batch, next_n, heads, _ = q.shape
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.dim() != 4
        or kv_cache.shape[2:] != (1, 132)
    ):
        raise ValueError("legacy paged MQA expects uint8 cache[pages,page_size,1,132]")
    if (
        weights.shape != (batch * next_n, heads)
        or context_lens.numel() != batch * next_n
    ):
        raise ValueError("legacy paged MQA weights or context lengths mismatch")
    q = q.contiguous()
    kv_cache = kv_cache.contiguous()
    block_table = block_table.contiguous()
    context_lens = context_lens.reshape(-1).contiguous()
    m = batch * next_n
    output = torch.empty((m, max_context_len), dtype=torch.float32, device=q.device)
    if not m or not max_context_len:
        return output
    if not is_supported(q, weights):
        page_size = kv_cache.shape[1]
        pool_2d = kv_cache.view(kv_cache.shape[0], page_size * 132)
        for row in range(m):
            length = min(int(context_lens[row]), max_context_len)
            ids = torch.arange(length, device=q.device)
            # Keep only entries whose logical page fits the block table and
            # whose physical page lies in [0, pool); everything else scores 0.
            page_idx = ids // page_size
            page_idx_ok = page_idx < block_table.shape[1]
            valid = torch.zeros(length, dtype=torch.bool, device=ids.device)
            valid[page_idx_ok] = True
            pages = torch.empty(0, dtype=torch.long, device=ids.device)
            table_idx = page_idx[page_idx_ok]
            if table_idx.numel():
                pages = block_table[row // next_n, table_idx].long()
                pages_ok = (pages >= 0) & (pages < kv_cache.shape[0])
                valid[page_idx_ok] = pages_ok
                pages = pages[pages_ok]
            ids = ids[valid]
            output[row].fill_(0.0)
            if not ids.numel():
                continue
            offs = ids % page_size
            k_columns = offs[:, None] * 128 + torch.arange(128, device=q.device)
            scale_columns = (
                page_size * 128 + offs[:, None] * 4 + torch.arange(4, device=q.device)
            )
            k = (
                pool_2d[pages[:, None], k_columns]
                .contiguous()
                .view(torch.float8_e4m3fn)
            )
            scale = (
                pool_2d[pages[:, None], scale_columns]
                .contiguous()
                .view(torch.float32)
                .reshape(-1)
            )
            scores = legacy_fp8_mqa_logits(
                q.view(m, heads, 128)[row : row + 1],
                (k, scale),
                weights[row : row + 1],
                torch.zeros(1, dtype=torch.int32, device=q.device),
                torch.full((1,), int(ids.numel()), dtype=torch.int32, device=q.device),
            )[0]
            output[row, ids] = scores
        return output

    if kv_cache.shape[0] == 0 or block_table.shape[1] == 0:
        output.zero_()
        return output

    page_size = kv_cache.shape[1]
    bn = _PAGED_BN
    bh = max(16, triton.next_power_of_2(heads))
    mask_h = heads != bh
    single_page = (page_size >= bn) and (page_size % bn == 0)
    use_fp8_dot = _use_native_fp8_dot(q.device)
    num_sms = _get_num_sms(q.device)

    use_bq2 = (next_n >= 2) and (
        batch * triton.cdiv(next_n, 2) * triton.cdiv(max_context_len, bn) >= 16
    )
    m_blocks = batch * triton.cdiv(next_n, 2) if use_bq2 else m
    max_paged_split = 512 if max_context_len <= 16384 else 1024
    split_kv, num_splits = _choose_split_kv(
        m_blocks,
        max_context_len,
        bn,
        target_ctas=(5 * num_sms) // 2,
        max_split_kv=max_paged_split,
    )
    kernel = _score_paged_bq2 if use_bq2 else _score_paged_bq1
    kernel[(m_blocks, num_splits)](
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        output,
        max_context_len,
        split_kv,
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        block_table.shape[1],
        kv_cache.shape[0],
        next_n,
        H=heads,
        BH=bh,
        PAGE=page_size,
        BN=bn,
        MASK_H=mask_h,
        SINGLE_PAGE=single_page,
        USE_FP8_DOT=use_fp8_dot,
        num_warps=4,
        num_stages=2,
    )
    return output
