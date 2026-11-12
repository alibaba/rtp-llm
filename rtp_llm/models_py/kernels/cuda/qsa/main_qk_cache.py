"""Device kernels; model state and execution policy belong to the caller."""

import triton
import triton.language as tl


@triton.jit
def _main_qk_norm_rope_cache(
    QKV,
    OUT,
    POS,
    QG,
    KG,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    EPS: tl.constexpr,
    BASE: tl.constexpr,
    H_SECTION: tl.constexpr,
    W_SECTION: tl.constexpr,
    LENS,
    TABLE,
    CACHE,
    SF,
    QUERY: tl.constexpr,
    TW: tl.constexpr,
    CP: tl.constexpr,
    CKV: tl.constexpr,
    SP: tl.constexpr,
    SKV: tl.constexpr,
    PAGES: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 32)
    column = lane * 4
    base = row * STRIDE + head * 256
    a0 = tl.load(QKV + base + column).to(tl.float32)
    a1 = tl.load(QKV + base + column + 1).to(tl.float32)
    a2 = tl.load(QKV + base + column + 2).to(tl.float32)
    a3 = tl.load(QKV + base + column + 3).to(tl.float32)
    b0 = tl.load(QKV + base + column + 128).to(tl.float32)
    b1 = tl.load(QKV + base + column + 129).to(tl.float32)
    b2 = tl.load(QKV + base + column + 130).to(tl.float32)
    b3 = tl.load(QKV + base + column + 131).to(tl.float32)
    s0, s1, s2, s3 = a0 * a0, a1 * a1, a2 * a2, a3 * a3
    t0, t1, t2, t3 = b0 * b0, b1 * b1, b2 * b2, b3 * b3
    # CUDA mean uses 64 reduction lanes for fewer than 16 outputs, otherwise
    # 32. Q and K use their own output counts, including replicated KV heads.
    wide = ((head < Q_HEADS) & (ROWS * Q_HEADS < 16)) | (
        (head >= Q_HEADS) & (ROWS * KV_HEADS < 16)
    )
    sum_wide = (((s0 + s1) + s2) + s3) + (((t0 + t1) + t2) + t3)
    sum_narrow = (((s0 + t0) + (s1 + t1)) + (s2 + t2)) + (s3 + t3)
    variance = tl.sum(tl.where(wide, sum_wide, sum_narrow), 0) / 256
    scale = tl.rsqrt(variance + EPS)
    length = tl.load(LENS + row)
    logical_position = length - 1
    page = logical_position // 128
    # The table lookup itself must stay inside the block table; a stale or
    # truncated LENS value must not turn into an out-of-bounds TABLE read.
    # Triton integer division truncates toward zero: (-1) // 128 is zero,
    # while its remainder is negative. Inactive warmup rows must not write
    # that negative token offset into the dummy page or its scales.
    page_in_table = (length > 0) & (page >= 0) & (page < TW)
    physical = tl.load(TABLE + (row // QUERY) * TW + page, page_in_table, other=0).to(
        tl.int64
    )
    valid_page = page_in_table & (physical >= 0) & (physical < PAGES)
    frequency = lane
    axis = tl.where(
        (frequency % 3 == 1) & (frequency < H_SECTION * 3),
        1,
        tl.where((frequency % 3 == 2) & (frequency < W_SECTION * 3), 2, 0),
    )
    position = tl.load(POS + row * 3 + axis).to(tl.float32)
    # Native treats nonpositive per-axis coordinates as the logical
    # row (including the committed prefix), rather than explicit zero.
    position = tl.where(position > 0, position, logical_position.to(tl.float32))
    # Match the native CUDA --use_fast_math sequence, including
    # approximate division and trig. libdevice pow/sin/cos use
    # different rounding and fail BF16 KV byte parity.
    log_base = tl.inline_asm_elementwise(
        "lg2.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[tl.full((), BASE, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    denominator = tl.exp2((frequency.to(tl.float32) / 32.0) * log_base)
    angle = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[position, denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    cosine = tl.inline_asm_elementwise(
        "cos.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[angle],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    sine = tl.inline_asm_elementwise(
        "sin.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[angle],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    for j in tl.static_range(4):
        for half in tl.static_range(2):
            rd = column + j + half * 128
            x = tl.load(QKV + base + rd).to(tl.float32)
            qg = tl.load(QG + rd).to(tl.float32)
            kg = tl.load(KG + rd).to(tl.float32)
            g = tl.where(head < Q_HEADS, qg, kg)
            value = ((x * scale) * (1.0 + g)).to(tl.bfloat16).to(tl.float32)
            other_rd = tl.where(rd < 32, rd + 32, rd - 32)
            other_x = tl.load(QKV + base + other_rd, rd < 64, other=0).to(tl.float32)
            other_qg = tl.load(QG + other_rd, rd < 64, other=0).to(tl.float32)
            other_kg = tl.load(KG + other_rd, rd < 64, other=0).to(tl.float32)
            other_g = tl.where(head < Q_HEADS, other_qg, other_kg)
            other_value = (
                ((other_x * scale) * (1.0 + other_g)).to(tl.bfloat16).to(tl.float32)
            )
            cosine_value = tl.gather(cosine, rd % 32, 0)
            sine_value = tl.gather(sine, rd % 32, 0)
            signed_other = tl.where(rd < 32, -other_value, other_value)
            rotated = tl.where(
                rd < 32,
                tl.fma(value, cosine_value, signed_other * sine_value),
                tl.fma(signed_other, sine_value, value * cosine_value),
            )
            rounded = tl.where(rd < 64, rotated, value).to(tl.bfloat16)
            if head < Q_HEADS:
                tl.store(OUT + (row * Q_HEADS + head) * 256 + rd, rounded)
            else:
                tl.store(
                    CACHE + physical * CP + (logical_position % 128) * 256 + rd,
                    rounded,
                    valid_page,
                )
            if head == 0:
                v_offset = row * STRIDE + (Q_HEADS + KV_HEADS) * 256 + rd
                v = tl.load(QKV + v_offset)
                tl.store(
                    CACHE + physical * CP + CKV + (logical_position % 128) * 256 + rd,
                    v,
                    valid_page,
                )
            if (j == 0) and (half == 0):
                if head == Q_HEADS:
                    tl.store(
                        SF + physical * SP + logical_position % 128, 1.0, valid_page
                    )
                    tl.store(
                        SF + physical * SP + SKV + logical_position % 128,
                        1.0,
                        valid_page,
                    )
