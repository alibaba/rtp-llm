"""Device kernels; model state and execution policy belong to the caller."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _decode_rope(
    position,
    inv_freq,
    frequency,
    H_SECTION: tl.constexpr,
    W_SECTION: tl.constexpr,
    INV_SCALE: tl.constexpr,
):
    axis = tl.where(
        (frequency % 3 == 1) & (frequency < 3 * H_SECTION),
        1,
        tl.where((frequency % 3 == 2) & (frequency < 3 * W_SECTION), 2, 0),
    )
    coordinate = tl.load(position + axis).to(tl.float32)
    angle = (coordinate * INV_SCALE) * tl.load(inv_freq + frequency)
    return libdevice.cos(angle).to(tl.bfloat16), libdevice.sin(angle).to(tl.bfloat16)


@triton.jit
def _decode_rotate(
    x, partner, cosine, sine, sign, SPLIT_ROUNDING: tl.constexpr = False
):
    if SPLIT_ROUNDING:
        # Q had a separate, non-contracting kernel. Keep both BF16 products
        # when sharing a launch with the legacy contracting K writer.
        signed = (partner.to(tl.float32) * sign).to(tl.bfloat16)
        bits = tl.inline_asm_elementwise(
            """{
                .reg .b16 left, right;
                mul.rn.bf16 left, $1, $2;
                mul.rn.bf16 right, $3, $4;
                add.rn.bf16 $0, left, right;
            }""",
            "=h,h,h,h,h",
            (
                x.to(tl.uint16, bitcast=True),
                cosine.to(tl.uint16, bitcast=True),
                signed.to(tl.uint16, bitcast=True),
                sine.to(tl.uint16, bitcast=True),
            ),
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        )
        return bits.to(tl.bfloat16, bitcast=True)
    left = (x.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
    right = (partner.to(tl.float32) * sine.to(tl.float32) * sign).to(tl.bfloat16)
    return (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _decode_query(
    q,
    out,
    gamma,
    positions,
    inv_freq,
    EPS: tl.constexpr,
    H_SECTION: tl.constexpr,
    W_SECTION: tl.constexpr,
    INV_SCALE: tl.constexpr,
):
    lane = tl.arange(0, 32)
    column = lane * 4
    a = tl.load(q + column).to(tl.float32)
    b = tl.load(q + column + 1).to(tl.float32)
    c = tl.load(q + column + 2).to(tl.float32)
    d = tl.load(q + column + 3).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(((a * a + b * b) + c * c) + d * d, 0) / 128 + EPS)
    cosine, sine = _decode_rope(
        positions, inv_freq, lane, H_SECTION, W_SECTION, INV_SCALE
    )
    first = (
        (tl.load(q + lane).to(tl.float32) * inverse)
        * (1.0 + tl.load(gamma + lane).to(tl.float32))
    ).to(tl.bfloat16)
    second = (
        (tl.load(q + lane + 32).to(tl.float32) * inverse)
        * (1.0 + tl.load(gamma + lane + 32).to(tl.float32))
    ).to(tl.bfloat16)
    tl.store(out + lane, _decode_rotate(first, second, cosine, sine, -1.0, True))
    tl.store(out + lane + 32, _decode_rotate(second, first, cosine, sine, 1.0, True))
    tail = tl.arange(0, 64) + 64
    normalized = (tl.load(q + tail).to(tl.float32) * inverse) * (
        1.0 + tl.load(gamma + tail).to(tl.float32)
    )
    tl.store(out + tail, normalized.to(tl.bfloat16))


@triton.jit
def _prepare_fused_decode(
    q,
    raw,
    lengths,
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
    out,
    visible_lengths,
    Q_STRIDE: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    POSITION_WIDTH: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    KV_BLOCKS: tl.constexpr,
    POSITION_BLOCKS: tl.constexpr,
    H_SECTION: tl.constexpr,
    W_SECTION: tl.constexpr,
    INV_SCALE: tl.constexpr,
    EPS: tl.constexpr,
    PADDED: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    position_row = positions + row * 3
    if head < 8:
        _decode_query(
            q + row * Q_STRIDE + head * 128,
            out + (row * 8 + head) * 128,
            q_gamma,
            position_row,
            inv_freq,
            EPS,
            H_SECTION,
            W_SECTION,
            INV_SCALE,
        )
    else:
        length = tl.load(lengths + row)
        position = tl.maximum(length - 1, 0)
        page = position // 128
        state_id = tl.load(
            state_table + row * STATE_WIDTH + page,
            mask=(length > 0) & (page < STATE_WIDTH),
            other=0,
        ).to(tl.int64)
        live = (state_id > 0) & (state_id < STATE_BLOCKS)
        # Padding has no state page for the current token. Preserve the scorer's
        # zero visibility for those rows even when graph lengths are nonzero.
        end = tl.where(live, length, 0) if PADDED else length
        tl.store(visible_lengths + row, end)
        if live:
            coord_id = tl.load(
                position_table + row * POSITION_WIDTH + page,
                mask=page < POSITION_WIDTH,
                other=0,
            ).to(tl.int64)
            coord_live = (coord_id > 0) & (coord_id < POSITION_BLOCKS)
            ring_offset = position % 8
            state_slot = state_id * 8 + ring_offset
            coord_slot = coord_id * 8 + ring_offset
            d = tl.arange(0, 128)
            raw_key = tl.load(raw + row * RAW_STRIDE + d).to(tl.float32)
            tl.store(state_pool + state_slot * 128 + d, raw_key)

            if (length % 4 == 0) & coord_live:
                kv_id = tl.load(
                    kv_table + row * KV_WIDTH + page, mask=page < KV_WIDTH, other=0
                ).to(tl.int64)
                if (kv_id > 0) & (kv_id < KV_BLOCKS):
                    group = tl.arange(0, 4)
                    slots = state_slot - 3 + group
                    history = tl.load(
                        state_pool + slots[:, None] * 128 + d[None, :],
                        mask=group[:, None] < 3,
                        other=0.0,
                    )
                    # The fourth key is already in registers, so the group
                    # does not depend on a global read of its just-written row.
                    history = tl.where(group[:, None] == 3, raw_key[None, :], history)
                    pooled = (tl.sum(history, 0) * 0.25).to(tl.bfloat16).to(tl.float32)
                    inverse = tl.rsqrt(tl.sum(pooled * pooled, 0) / 128 + EPS)
                    gain = 1.0 + tl.load(k_gamma + d).to(tl.float32)
                    normalized = (pooled * inverse * gain).to(tl.bfloat16)
                    partner = tl.gather(normalized, d ^ 32, 0)
                    # Both the group start and its last token are in the same
                    # page and the same aligned four-slot half of the ring.
                    cosine, sine = _decode_rope(
                        position_pool + (coord_slot - 3) * 3,
                        inv_freq,
                        d % 32,
                        H_SECTION,
                        W_SECTION,
                        INV_SCALE,
                    )
                    rotated = _decode_rotate(
                        normalized, partner, cosine, sine, tl.where(d < 32, -1.0, 1.0)
                    )
                    kv_slot = kv_id * 32 + (position // 4) % 32
                    tl.store(
                        kv_pool + kv_slot * 128 + d,
                        tl.where(d < 64, rotated, normalized),
                    )
            for axis in tl.static_range(3):
                coordinate = tl.load(position_row + axis)
                tl.store(
                    position_pool + coord_slot * 3 + axis, coordinate, mask=coord_live
                )


@triton.jit
def _write_decode_key(
    raw_ptr,
    positions_ptr,
    valid_ptr,
    cos_ptr,
    sin_ptr,
    gamma_ptr,
    kv_pool_ptr,
    kv_table_ptr,
    state_pool_ptr,
    state_table_ptr,
    KV_TABLE_WIDTH: tl.constexpr,
    STATE_TABLE_WIDTH: tl.constexpr,
    KV_POOL_BLOCKS: tl.constexpr,
    STATE_POOL_BLOCKS: tl.constexpr,
    KV_TOKENS_PER_BLOCK: tl.constexpr,
    STATE_TOKENS_PER_BLOCK: tl.constexpr,
    KV_ENTRIES_PER_BLOCK: tl.constexpr,
    D: tl.constexpr,
    ROTARY: tl.constexpr,
    EPS: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    request = tl.program_id(0)
    d = tl.arange(0, D)
    active = tl.load(valid_ptr + request) != 0 if HAS_VALID else True
    position = tl.load(positions_ptr + request)
    state_column = position // STATE_TOKENS_PER_BLOCK
    state_id = tl.load(
        state_table_ptr + request * STATE_TABLE_WIDTH + state_column,
        mask=active & (position >= 0) & (state_column < STATE_TABLE_WIDTH),
        other=0,
    ).to(tl.int64)
    state_valid = (position >= 0) & (state_id > 0) & (state_id < STATE_POOL_BLOCKS)
    raw = tl.load(raw_ptr + request * D + d, mask=active, other=0).to(tl.float32)
    tl.store(
        state_pool_ptr + (state_id * 8 + position % 8) * D + d,
        raw,
        mask=active & state_valid,
    )

    if active and (position + 1) % 4 == 0:
        group = tl.arange(0, 4)
        source_positions = position - 3 + group
        source_columns = source_positions // STATE_TOKENS_PER_BLOCK
        source_ids = tl.load(
            state_table_ptr + request * STATE_TABLE_WIDTH + source_columns,
            mask=(source_positions >= 0) & (source_columns < STATE_TABLE_WIDTH),
            other=0,
        ).to(tl.int64)
        source_valid = (source_ids > 0) & (source_ids < STATE_POOL_BLOCKS)
        source_slots = source_ids * 8 + source_positions % 8
        history = tl.load(
            state_pool_ptr + source_slots[:, None] * D + d[None, :],
            mask=source_valid[:, None],
            other=0.0,
        )
        # The current key has just been written by this program. Read the
        # input directly so no global-memory fence is needed for that row.
        history = tl.where(group[:, None] == 3, raw[None, :], history)
        pooled = (tl.sum(history, axis=0) * 0.25).to(tl.bfloat16).to(tl.float32)
        gain = 1.0 + tl.load(gamma_ptr + d).to(tl.float32)
        rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / D + EPS)
        normalized = (pooled * rms * gain).to(tl.bfloat16)

        # Upstream applies RoPE to BF16 values: each multiplication and the
        # final addition rounds to BF16. Preserve that boundary here.
        partner = tl.where(
            d < ROTARY,
            tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2),
            d,
        )
        rotated_half = tl.gather(normalized, partner, axis=0)
        cosine = tl.load(cos_ptr + request * ROTARY + d, mask=d < ROTARY, other=1)
        sine = tl.load(sin_ptr + request * ROTARY + d, mask=d < ROTARY, other=0)
        left = (normalized.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
        right = (
            rotated_half.to(tl.float32)
            * sine.to(tl.float32)
            * tl.where(d < ROTARY // 2, -1.0, 1.0)
        ).to(tl.bfloat16)
        value = tl.where(
            d < ROTARY,
            (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
            normalized,
        )

        kv_column = position // KV_TOKENS_PER_BLOCK
        kv_id = tl.load(
            kv_table_ptr + request * KV_TABLE_WIDTH + kv_column,
            mask=(position >= 0) & (kv_column < KV_TABLE_WIDTH),
            other=0,
        ).to(tl.int64)
        kv_valid = (kv_id > 0) & (kv_id < KV_POOL_BLOCKS)
        entry = (position // 4) % KV_ENTRIES_PER_BLOCK
        tl.store(
            kv_pool_ptr + (kv_id * KV_ENTRIES_PER_BLOCK + entry) * D + d,
            value,
            mask=kv_valid,
        )
