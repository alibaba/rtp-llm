"""Device kernels; model state and execution policy belong to the caller."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .abi import PREFILL_COMPRESS_RATIO, PREFILL_HEAD_DIM, PREFILL_STATE_RING_ROWS

STATE_RING_ROWS = tl.constexpr(PREFILL_STATE_RING_ROWS)
INDEXER_HEAD_DIM = tl.constexpr(PREFILL_HEAD_DIM)
COMPRESS_RATIO = tl.constexpr(PREFILL_COMPRESS_RATIO)


@triton.jit
def _prefill_rope_pair(
    positions,
    inv_freq,
    row,
    freq,
    INV_SCALE: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    axis = tl.where(
        (freq % 3 == 1) & (freq < 3 * H),
        1,
        tl.where((freq % 3 == 2) & (freq < 3 * W), 2, 0),
    )
    coordinate = tl.load(positions + row * 3 + axis).to(tl.float32)
    angle = (coordinate * INV_SCALE) * tl.load(inv_freq + freq)
    cosine = libdevice.cos(angle).to(tl.bfloat16).to(tl.float32)
    sine = libdevice.sin(angle).to(tl.bfloat16).to(tl.float32)
    return cosine, sine


@triton.jit
def _prefill_query(
    q,
    gamma,
    output,
    positions,
    inv_freq,
    row,
    Q_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    INV_SCALE: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    head, column = tl.arange(0, 8), tl.arange(0, 128)
    raw = tl.load(q + row * Q_STRIDE + head[:, None] * 128 + column[None, :])
    raw = raw.to(tl.float32)
    even, odd = tl.split(raw.reshape(8, 32, 2, 2))
    a, c = tl.split(even)
    b, d = tl.split(odd)
    squared = ((a * a + b * b) + c * c) + d * d
    inverse = tl.rsqrt(tl.sum(squared, 1) / 128.0 + EPS)
    gain = 1.0 + tl.load(gamma + column).to(tl.float32)
    normed = ((raw * inverse[:, None]) * gain[None, :]).to(tl.bfloat16)
    rotary, tail = tl.split(normed.reshape(8, 2, 64).permute(0, 2, 1))
    first, second = tl.split(rotary.reshape(8, 2, 32).permute(0, 2, 1))
    freq = tl.arange(0, 32)
    cosine, sine = _prefill_rope_pair(positions, inv_freq, row, freq, INV_SCALE, H, W)
    first, second = first.to(tl.float32), second.to(tl.float32)
    left = (first * cosine[None, :]).to(tl.bfloat16).to(tl.float32)
    left += (-second * sine[None, :]).to(tl.bfloat16).to(tl.float32)
    right = (second * cosine[None, :]).to(tl.bfloat16).to(tl.float32)
    right += (first * sine[None, :]).to(tl.bfloat16).to(tl.float32)
    out = output + (row * 8 + head[:, None]) * 128
    tl.store(out + freq[None, :], left)
    tl.store(out + 32 + freq[None, :], right)
    tl.store(out + 64 + tl.arange(0, 64)[None, :], tail)


@triton.jit
def _prefill_rotate_key(pooled, gain, squared, cosine, sine, column, EPS: tl.constexpr):
    inverse = tl.rsqrt(squared / 128.0 + EPS)
    normed = (pooled * inverse * gain).to(tl.bfloat16)
    partner = tl.where(
        column < 64, tl.where(column < 32, column + 32, column - 32), column
    )
    rotated = tl.gather(normed, partner, 0).to(tl.float32)
    left = (normed.to(tl.float32) * cosine).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sine * tl.where(column < 32, -1.0, 1.0)).to(tl.bfloat16)
    return tl.where(column < 64, (left + right.to(tl.float32)).to(tl.bfloat16), normed)


@triton.jit
def _prefill_key(
    raw_keys,
    gamma,
    positions,
    inv_freq,
    kv_pool,
    kv_table,
    dense,
    request,
    row,
    logical,
    group,
    RAW_STRIDE: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    POOL_ROWS: tl.constexpr,
    EPS: tl.constexpr,
    INV_SCALE: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    EMIT_DENSE: tl.constexpr,
):
    page = logical // 128
    physical = tl.load(kv_table + request * TABLE_ROWS + page, page < TABLE_COLS, 0)
    if (physical > 0) & (physical < POOL_ROWS):
        column = tl.arange(0, 128)
        tokens = tl.arange(0, 4)
        raw = tl.load(raw_keys + (row + tokens[:, None]) * RAW_STRIDE + column[None, :])
        pooled = (tl.sum(raw.to(tl.float32), 0) * 0.25).to(tl.bfloat16).to(tl.float32)
        gain = 1.0 + tl.load(gamma + column).to(tl.float32)
        freq = tl.arange(0, 32)
        cos_half, sin_half = _prefill_rope_pair(
            positions, inv_freq, row, freq, INV_SCALE, H, W
        )
        cosine = tl.gather(cos_half, column % 32, 0)
        sine = tl.gather(sin_half, column % 32, 0)

        # Dense cold-prefill scoring and the existing prefix writer use the
        # standalone RMSNorm reduction. Cold paged cache bytes use its writer's
        # full-head reduction, even when that writer emits the dense operand too.
        a = tl.gather(pooled, freq * 4, 0)
        b = tl.gather(pooled, freq * 4 + 1, 0)
        c = tl.gather(pooled, freq * 4 + 2, 0)
        d = tl.gather(pooled, freq * 4 + 3, 0)
        adjacent = tl.sum(((a * a + b * b) + c * c) + d * d, 0)
        cache_sum = tl.sum(pooled * pooled, 0) if EMIT_DENSE else adjacent
        cached = _prefill_rotate_key(pooled, gain, cache_sum, cosine, sine, column, EPS)
        slot = physical.to(tl.int64) * 32 + (logical // 4) % 32
        tl.store(kv_pool + slot * 128 + column, cached)
        if EMIT_DENSE:
            selected = _prefill_rotate_key(
                pooled, gain, adjacent, cosine, sine, column, EPS
            )
            tl.store(dense + group * 128 + column, selected)


@triton.jit(do_not_specialize=["max_length"])
def _prepare_fused_prefill(
    q,
    raw_keys,
    cu,
    prefixes,
    positions,
    q_gamma,
    k_gamma,
    inv_freq,
    state_pool,
    state_table,
    kv_pool,
    kv_table,
    position_pool,
    position_table,
    rotated_q,
    dense,
    visible,
    compressed,
    max_length,
    Q_STRIDE: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    STATE_COLS: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    KV_ROWS: tl.constexpr,
    KV_COLS: tl.constexpr,
    KV_BLOCKS: tl.constexpr,
    POSITION_ROWS: tl.constexpr,
    POSITION_COLS: tl.constexpr,
    POSITION_BLOCKS: tl.constexpr,
    EPS: tl.constexpr,
    INV_SCALE: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    EMIT_DENSE: tl.constexpr,
    Q_TILE: tl.constexpr,
):
    tile, request = tl.program_id(0), tl.program_id(1)
    start = tl.load(cu + request).to(tl.int64)
    length = tl.load(cu + request + 1).to(tl.int64) - start
    prefix = tl.load(prefixes + request).to(tl.int32)
    groups = max_length // 4
    query_tiles = tl.cdiv(max_length, Q_TILE)
    if tile < query_tiles:
        for row_id in tl.static_range(Q_TILE):
            local_row = tile * Q_TILE + row_id
            if local_row < length:
                row = start + local_row
                _prefill_query(
                    q,
                    q_gamma,
                    rotated_q,
                    positions,
                    inv_freq,
                    row,
                    Q_STRIDE,
                    EPS,
                    INV_SCALE,
                    H,
                    W,
                )
                if not EMIT_DENSE:
                    tl.store(visible + row, prefix + local_row + 1)
        if not EMIT_DENSE:
            if tile == 0:
                tl.store(compressed + request, (prefix + length) // 4)
    elif tile < query_tiles + groups:
        group = tile - query_tiles
        if group < length // 4:
            row = start + group * 4
            logical = prefix + group * 4
            _prefill_key(
                raw_keys,
                k_gamma,
                positions,
                inv_freq,
                kv_pool,
                kv_table,
                dense,
                request,
                row,
                logical,
                group,
                RAW_STRIDE,
                KV_ROWS,
                KV_COLS,
                KV_BLOCKS,
                EPS,
                INV_SCALE,
                H,
                W,
                EMIT_DENSE,
            )
    else:
        page = tile - query_tiles - groups
        if page * 128 < length:
            physical_page = prefix // 128 + page
            end = tl.minimum((page + 1) * 128, length)
            # Each page CTA stores at most one token per ring slot. For a short
            # last page only its live tokens update the rings.
            ring = tl.arange(0, 8)
            offset = end - 1 - (end - 1 - ring + 8) % 8
            live = offset >= page * 128
            row = start + offset
            state_block = tl.load(
                state_table + request * STATE_ROWS + physical_page,
                physical_page < STATE_COLS,
                0,
            )
            state_valid = live & (state_block > 0) & (state_block < STATE_BLOCKS)
            state_slot = state_block.to(tl.int64) * 8 + ring
            col = tl.arange(0, 128)
            raw = tl.load(
                raw_keys + row[:, None] * RAW_STRIDE + col[None, :],
                state_valid[:, None],
                0,
            )
            tl.store(
                state_pool + state_slot[:, None] * 128 + col[None, :],
                raw.to(tl.float32),
                state_valid[:, None],
            )
            pos_block = tl.load(
                position_table + request * POSITION_ROWS + physical_page,
                physical_page < POSITION_COLS,
                0,
            )
            pos_valid = live & (pos_block > 0) & (pos_block < POSITION_BLOCKS)
            pos_slot = pos_block.to(tl.int64) * 8 + ring
            for axis in tl.static_range(3):
                value = tl.load(positions + row * 3 + axis, pos_valid, 0)
                tl.store(position_pool + pos_slot * 3 + axis, value, pos_valid)


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
