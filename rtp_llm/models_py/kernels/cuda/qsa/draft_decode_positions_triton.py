import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["ROWS"])
def _draft_decode_positions(
    SEQ,
    POS,
    INV,
    CURRENT_COS,
    CURRENT_SIN,
    BLOCK_COS,
    BLOCK_SIN,
    VISIBLE,
    COMPRESSED,
    ROWS,
    FACTOR: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    MROPE: tl.constexpr,
    H_SECTION: tl.constexpr,
    W_SECTION: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    frequency = tl.arange(0, 32)
    seq = tl.load(SEQ + row, row < ROWS, other=0)
    if MROPE:
        axis = tl.where((frequency % 3 == 1) & (frequency < 3 * H_SECTION), 1, 0)
        axis = tl.where((frequency % 3 == 2) & (frequency < 3 * W_SECTION), 2, axis)
        position = tl.load(
            POS + row[:, None] * FACTOR + axis[None, :], row[:, None] < ROWS, other=0
        ).to(tl.int64)
        # MRoPE can be offset from the cache coordinate after image tokens.
        block_position = (position - (seq[:, None] & 3)).to(tl.float32)
        position = position.to(tl.float32)
    else:
        position = tl.load(POS + row * FACTOR, row < ROWS, other=0).to(tl.float32)[
            :, None
        ]
        block_position = ((seq.to(tl.int64) >> 2) << 2).to(tl.float32)[:, None]
    inverse = tl.load(INV + frequency)
    current_angle = (position * INV_SCALE) * inverse[None, :]
    block_angle = (block_position * INV_SCALE) * inverse[None, :]
    offset = row[:, None] * 64 + frequency[None, :]
    cc, cs = libdevice.cos(current_angle), libdevice.sin(current_angle)
    bc, bs = libdevice.cos(block_angle), libdevice.sin(block_angle)
    tl.store(CURRENT_COS + offset, cc, row[:, None] < ROWS)
    tl.store(CURRENT_COS + offset + 32, cc, row[:, None] < ROWS)
    tl.store(CURRENT_SIN + offset, cs, row[:, None] < ROWS)
    tl.store(CURRENT_SIN + offset + 32, cs, row[:, None] < ROWS)
    tl.store(BLOCK_COS + offset, bc, row[:, None] < ROWS)
    tl.store(BLOCK_COS + offset + 32, bc, row[:, None] < ROWS)
    tl.store(BLOCK_SIN + offset, bs, row[:, None] < ROWS)
    tl.store(BLOCK_SIN + offset + 32, bs, row[:, None] < ROWS)
    tl.store(VISIBLE + row, seq + 1, row < ROWS)
    tl.store(COMPRESSED + row, (seq + 1) >> 2, row < ROWS)
