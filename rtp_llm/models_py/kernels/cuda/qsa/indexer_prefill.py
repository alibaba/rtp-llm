"""Device kernels; model state and execution policy belong to the caller."""

import triton
import triton.language as tl

from .abi import PREFILL_COMPRESS_RATIO, PREFILL_HEAD_DIM, PREFILL_STATE_RING_ROWS

STATE_RING_ROWS = tl.constexpr(PREFILL_STATE_RING_ROWS)
INDEXER_HEAD_DIM = tl.constexpr(PREFILL_HEAD_DIM)
COMPRESS_RATIO = tl.constexpr(PREFILL_COMPRESS_RATIO)


@triton.jit(do_not_specialize=["PAGES"])
def _state_snapshot(
    cu,
    table,
    pool,
    slots,
    original,
    WIDTH: tl.constexpr,
    PAGES,
    PAGE: tl.constexpr,
):
    request, page = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    length = tl.load(cu + request + 1) - tl.load(cu + request)
    r, d = tl.arange(0, STATE_RING_ROWS), tl.arange(0, INDEXER_HEAD_DIM)
    end = tl.minimum((page + 1) * PAGE, length)
    position = end - 1 - (end - 1 - r + 8) % 8
    valid = position >= page * PAGE
    block = tl.load(
        table + request * WIDTH + page, mask=page * PAGE < length, other=0
    ).to(tl.int64)
    slot = block * 8 + r
    row = (request * PAGES + page) * 8 + r
    value = tl.load(
        pool + slot[:, None] * 128 + d[None, :], mask=valid[:, None], other=0
    )
    tl.store(slots + row, tl.where(valid, slot, -1))
    tl.store(original + row[:, None] * 128 + d[None, :], value)


@triton.jit(do_not_specialize=["GROUPS"])
def _kv_snapshot(
    cu,
    table,
    pool,
    slots,
    original,
    WIDTH: tl.constexpr,
    GROUPS,
    PAGE: tl.constexpr,
):
    group, request = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    length = tl.load(cu + request + 1) - tl.load(cu + request)
    valid = group < length // 4
    block = tl.load(
        table + request * WIDTH + group * 4 // PAGE, mask=valid, other=0
    ).to(tl.int64)
    slot = block * (PAGE // 4) + group % (PAGE // 4)
    d = tl.arange(0, INDEXER_HEAD_DIM)
    value = tl.load(pool + slot * 128 + d, mask=valid, other=0)
    row = request * GROUPS + group
    tl.store(slots + row, tl.where(valid, slot, -1))
    tl.store(original + row * 128 + d, value)


@triton.jit(do_not_specialize=["PAGES"])
def _state_write(
    raw,
    cu,
    slots,
    pool,
    PAGES,
    PAGE: tl.constexpr,
):
    request, page = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    start = tl.load(cu + request)
    length = tl.load(cu + request + 1) - start
    r, d = tl.arange(0, STATE_RING_ROWS), tl.arange(0, INDEXER_HEAD_DIM)
    end = tl.minimum((page + 1) * PAGE, length)
    position = end - 1 - (end - 1 - r + 8) % 8
    row = (request * PAGES + page) * 8 + r
    slot = tl.load(slots + row)
    value = tl.load(
        raw + (start + position[:, None]) * 128 + d[None, :],
        mask=(slot >= 0)[:, None],
        other=0,
    ).to(tl.float32)
    tl.store(pool + slot[:, None] * 128 + d[None, :], value, mask=(slot >= 0)[:, None])


@triton.jit(do_not_specialize=["GROUPS", "MAX_LENGTH"])
def _kv_write(
    raw,
    cu,
    cos,
    sin,
    gamma,
    slots,
    pool,
    dense_keys,
    GROUPS,
    MAX_LENGTH,
    EPS: tl.constexpr,
    EMIT_DENSE: tl.constexpr,
):
    group, request = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    slot = tl.load(slots + request * GROUPS + group)
    if slot >= 0:
        start = tl.load(cu + request)
        r, d = tl.arange(0, 4), tl.arange(0, INDEXER_HEAD_DIM)
        value = tl.load(raw + (start + group * 4 + r[:, None]) * 128 + d[None, :]).to(
            tl.float32
        )
        pooled = (tl.sum(value, 0) * 0.25).to(tl.bfloat16).to(tl.float32)
        gain = 1.0 + tl.load(gamma + d).to(tl.float32)
        rms = tl.rsqrt(tl.sum(pooled * pooled, 0) / 128 + EPS)
        normalized = (pooled * rms * gain).to(tl.bfloat16)
        partner = tl.where(d < 64, tl.where(d < 32, d + 32, d - 32), d)
        rotated = tl.gather(normalized, partner, 0)
        rope_row = (request * MAX_LENGTH + group * 4) * 64
        c = tl.load(cos + rope_row + d, mask=d < 64, other=1).to(tl.float32)
        s = tl.load(sin + rope_row + d, mask=d < 64, other=0).to(tl.float32)
        left = (normalized.to(tl.float32) * c).to(tl.bfloat16)
        right = (rotated.to(tl.float32) * s * tl.where(d < 32, -1.0, 1.0)).to(
            tl.bfloat16
        )
        output = tl.where(
            d < 64,
            (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
            normalized,
        )
        tl.store(pool + slot * 128 + d, output)
        if EMIT_DENSE:
            # Selection's standalone CUDA mean uses 32 lanes, each reducing
            # four adjacent squared FP32 values. Keep cache bytes above
            # unchanged and reproduce that consumer's reduction separately.
            lane = tl.arange(0, 32)
            a = tl.gather(pooled, lane * 4, 0)
            b = tl.gather(pooled, lane * 4 + 1, 0)
            c0 = tl.gather(pooled, lane * 4 + 2, 0)
            d0 = tl.gather(pooled, lane * 4 + 3, 0)
            variance = tl.sum(((a * a + b * b) + c0 * c0) + d0 * d0, 0) / 128
            dense_norm = (pooled * tl.rsqrt(variance + EPS) * gain).to(tl.bfloat16)
            partner_value = tl.gather(dense_norm, partner, 0)
            left = (dense_norm.to(tl.float32) * c).to(tl.bfloat16)
            right = (partner_value.to(tl.float32) * s * tl.where(d < 32, -1.0, 1.0)).to(
                tl.bfloat16
            )
            dense = tl.where(
                d < 64,
                (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
                dense_norm,
            )
            tl.store(dense_keys + (request * GROUPS + group) * 128 + d, dense)


@triton.jit
def _metadata(
    cu,
    state_table,
    kv_table,
    state_slots,
    kv_slots,
    completed,
    STATE_WIDTH: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    position = tl.program_id(1).to(tl.int64) * 128 + tl.arange(0, INDEXER_HEAD_DIM)
    start, end = tl.load(cu + request), tl.load(cu + request + 1)
    live = position < end - start
    state_block = tl.load(
        state_table + request * STATE_WIDTH + position // PAGE, mask=live, other=0
    )
    done = (position + 1) % 4 == 0
    kv_block = tl.load(
        kv_table + request * KV_WIDTH + position // PAGE, mask=live & done, other=0
    )
    tl.store(state_slots + start + position, state_block * 8 + position % 8, mask=live)
    tl.store(
        kv_slots + start + position,
        tl.where(done, kv_block * (PAGE // 4) + position // 4 % (PAGE // 4), -1),
        mask=live,
    )
    tl.store(completed + start + position, done, mask=live)


@triton.jit
def _restore_slots(pool, slots, original):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row)
    d = tl.arange(0, INDEXER_HEAD_DIM)
    value = tl.load(original + row * 128 + d)
    tl.store(pool + slot * 128 + d, value, mask=slot >= 0)
