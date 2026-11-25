"""Qwen4 Q/K normalization, RoPE, and paged BF16 cache stores for QSA."""

import triton
import triton.language as tl


@triton.jit
def _rope_factors(
    POS, row, logical_pos, PREFILL: tl.constexpr, BASE_ROPE: tl.constexpr
):
    frequency = tl.arange(0, 32)
    if BASE_ROPE:
        # Base consumes only the first text axis. The native uncached writer
        # uses the logical position when that explicit position is nonpositive.
        position = tl.load(POS + row.to(tl.int64) * 3)
        position = tl.where(position > 0, position, logical_pos)
    elif PREFILL:
        axis = tl.where(
            frequency % 3 == 1,
            1,
            tl.where((frequency % 3 == 2) & (frequency < 30), 2, 0),
        )
        position = tl.load(POS + row.to(tl.int64) * 3 + axis)
        position = tl.where(position > 0, position, logical_pos)
    else:
        # The decode writer uses the first position ID for all frequencies.
        position = tl.load(POS + row.to(tl.int64) * 3)

    # Preserve the native CUDA writer's fast-math sequence and rounding.
    log_base = tl.inline_asm_elementwise(
        "lg2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [tl.full((), 10000000.0, tl.float32)],
        tl.float32,
        True,
        1,
    )
    denominator = tl.exp2((frequency.to(tl.float32) / 32.0) * log_base)
    angle = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [position.to(tl.float32), denominator],
        tl.float32,
        True,
        1,
    )
    cosine = tl.inline_asm_elementwise(
        "cos.approx.ftz.f32 $0, $1;",
        "=f,f",
        [angle],
        tl.float32,
        True,
        1,
    )
    sine = tl.inline_asm_elementwise(
        "sin.approx.ftz.f32 $0, $1;",
        "=f,f",
        [angle],
        tl.float32,
        True,
        1,
    )
    return cosine, sine


@triton.jit
def _vector_head_parts(values, HEADS: tl.constexpr):
    even, odd = tl.split(tl.reshape(values, (HEADS, 32, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    return a0, a1, a2, a3


@triton.jit
def _norm_rope(
    QKV,
    POS,
    GAMMA,
    row,
    logical_pos,
    head,
    column,
    QKV_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    HEADS: tl.constexpr,
    WIDE_NORM: tl.constexpr,
    PREFILL: tl.constexpr,
    BASE_ROPE: tl.constexpr,
):
    source = row.to(tl.int64) * QKV_STRIDE + head[:, None] * 256 + column[None, :]
    lo = tl.load(QKV + source).to(tl.float32)
    hi = tl.load(QKV + source + 128).to(tl.float32)
    a0, a1, a2, a3 = _vector_head_parts(lo, HEADS)
    b0, b1, b2, b3 = _vector_head_parts(hi, HEADS)
    s0, s1, s2, s3 = a0 * a0, a1 * a1, a2 * a2, a3 * a3
    t0, t1, t2, t3 = b0 * b0, b1 * b1, b2 * b2, b3 * b3
    if WIDE_NORM:
        sums = (((s0 + s1) + s2) + s3) + (((t0 + t1) + t2) + t3)
    else:
        sums = (((s0 + t0) + (s1 + t1)) + (s2 + t2)) + (s3 + t3)
    # Torch's CUDA mean uses this order with either 32 or 64 reduction lanes.
    lane = tl.arange(0, 32)
    for shift in tl.static_range(4, -1, -1):
        peer = tl.broadcast_to((lane ^ (1 << shift))[None, :], (HEADS, 32))
        sums += tl.gather(sums, peer, 1)
    inverse = tl.rsqrt(sums / 256.0 + EPS)
    inverse = tl.reshape(
        tl.broadcast_to(inverse[:, :, None], (HEADS, 32, 4)), (HEADS, 128)
    )
    gain_lo = 1.0 + tl.load(GAMMA + column).to(tl.float32)
    gain_hi = 1.0 + tl.load(GAMMA + column + 128).to(tl.float32)
    nlo = ((lo * inverse) * gain_lo[None, :]).to(tl.bfloat16)
    nhi = ((hi * inverse) * gain_hi[None, :]).to(tl.bfloat16)

    # The native writer rotates the BF16-rounded normalized Q/K values.
    cosine, sine = _rope_factors(POS, row, logical_pos, PREFILL, BASE_ROPE)
    cosine = tl.gather(cosine, column % 32, 0)[None, :]
    sine = tl.gather(sine, column % 32, 0)[None, :]
    peer = tl.broadcast_to((column ^ 32)[None, :], (HEADS, 128))
    other = tl.gather(nlo, peer, 1).to(tl.float32)
    value = nlo.to(tl.float32)
    rotated = tl.where(
        column[None, :] < 32,
        tl.fma(value, cosine, -other * sine),
        tl.fma(other, sine, value * cosine),
    )
    return tl.where(column[None, :] < 64, rotated, value).to(tl.bfloat16), nhi


@triton.jit
def prepare_qkv(
    QKV,
    QUERY,
    POS,
    Q_GAMMA,
    K_GAMMA,
    CACHE,
    row,
    head_tile,
    logical_pos,
    physical_page,
    active,
    QKV_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    RMS_ROWS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    PREFILL: tl.constexpr,
    BASE_ROPE: tl.constexpr,
    Q_HEADS: tl.constexpr = 64,
    KV_HEADS: tl.constexpr = 4,
):
    """Store dense BF16 Q and write one K/V row to its allocated cache page.

    The grid has ceil((Q_HEADS + 2 * KV_HEADS) / BLOCK_HEADS) head tiles
    per query row. BLOCK_HEADS divides Q_HEADS. The input projection stays
    immutable, and graph padding never writes to physical page zero.
    """
    tl.static_assert(Q_HEADS % BLOCK_HEADS == 0)
    column = tl.arange(0, 128)
    first = head_tile * BLOCK_HEADS
    if first < Q_HEADS:
        head = first + tl.arange(0, BLOCK_HEADS)
        lo, hi = _norm_rope(
            QKV,
            POS,
            Q_GAMMA,
            row,
            logical_pos,
            head,
            column,
            QKV_STRIDE,
            EPS,
            BLOCK_HEADS,
            RMS_ROWS * Q_HEADS < 16,
            PREFILL,
            BASE_ROPE,
        )
        target = (row.to(tl.int64) * Q_HEADS + head[:, None]) * 256 + column[None, :]
        tl.store(QUERY + target, lo)
        tl.store(QUERY + target + 128, hi)
    elif active & (physical_page > 0):
        bank_head = first - Q_HEADS
        tile_heads: tl.constexpr = min(BLOCK_HEADS, KV_HEADS)
        lane = tl.arange(0, tile_heads)
        page = physical_page.to(tl.int64) * CACHE_STRIDE
        token = (logical_pos % PAGE_SIZE) * 256 + column[None, :]
        head_stride: tl.constexpr = PAGE_SIZE * 256
        if bank_head < KV_HEADS:
            cache_head = bank_head + lane
            lo, hi = _norm_rope(
                QKV,
                POS,
                K_GAMMA,
                row,
                logical_pos,
                Q_HEADS + cache_head,
                column,
                QKV_STRIDE,
                EPS,
                tile_heads,
                RMS_ROWS * KV_HEADS < 16,
                PREFILL,
                BASE_ROPE,
            )
            target = page + cache_head[:, None] * head_stride + token
            tl.store(CACHE + target, lo)
            tl.store(CACHE + target + 128, hi)

        # A wide tile covers K and V together. Smaller tiles copy V directly.
        if BLOCK_HEADS >= 2 * KV_HEADS or bank_head >= KV_HEADS:
            cache_head = tl.maximum(bank_head, KV_HEADS) + lane
            source = (
                row.to(tl.int64) * QKV_STRIDE
                + (Q_HEADS + cache_head[:, None]) * 256
                + column[None, :]
            )
            target = page + cache_head[:, None] * head_stride + token
            tl.store(CACHE + target, tl.load(QKV + source))
            tl.store(CACHE + target + 128, tl.load(QKV + source + 128))
