import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
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
    ROWS: tl.constexpr,
    FACTOR: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    frequency = tl.arange(0, 32)
    seq = tl.load(SEQ + row, row < ROWS, other=0)
    position = tl.load(POS + row * FACTOR, row < ROWS, other=0).to(tl.float32)
    block_position = ((seq.to(tl.int64) >> 2) << 2).to(tl.float32)
    inverse = tl.load(INV + frequency)
    current_angle = (position[:, None] * INV_SCALE) * inverse[None, :]
    block_angle = (block_position[:, None] * INV_SCALE) * inverse[None, :]
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
