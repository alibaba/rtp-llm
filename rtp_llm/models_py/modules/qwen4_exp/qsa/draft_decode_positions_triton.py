"""Fuse existing draft Graph position arithmetic and cached-frequency RoPE.

Current transported positions deliberately remain distinct from logical cache
positions. This experiment keeps the qualified libdevice/cast math unchanged.
"""

import math

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


def try_prepare_decode_positions(sequence, positions, frequency, config):
    from .indexer import is_qsa_rope_style

    if not (
        sequence.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(sequence.device) == (12, 0)
        and sequence.ndim == 1
        and sequence.dtype == torch.int32
        and sequence.is_contiguous()
        and 1 <= sequence.numel() <= 32
        and positions.ndim == 1
        and positions.dtype in (torch.int32, torch.int64)
        and positions.is_contiguous()
        and positions.device == sequence.device
        and isinstance(frequency, torch.Tensor)
        and frequency.dtype == torch.float32
        and frequency.shape == (32,)
        and frequency.is_contiguous()
        and frequency.device == sequence.device
        and int(config.dim) == 64
        and int(config.index_factor) in (1, 3)
        and positions.numel() == sequence.numel() * int(config.index_factor)
        and is_qsa_rope_style(config, "Base")
        and bool(getattr(config, "indexer_is_neox_style", True))
    ):
        return None
    scale = float(getattr(config, "scale", 1.0))
    base = float(config.base)
    if not (math.isfinite(scale) and scale > 0 and math.isfinite(base) and base > 0):
        return None
    rows = sequence.numel()
    cc = torch.empty((rows, 64), device=sequence.device, dtype=torch.bfloat16)
    cs, bc, bs = torch.empty_like(cc), torch.empty_like(cc), torch.empty_like(cc)
    visible, compressed = torch.empty_like(sequence), torch.empty_like(sequence)
    _draft_decode_positions[(triton.cdiv(rows, 4),)](
        sequence,
        positions,
        frequency,
        cc,
        cs,
        bc,
        bs,
        visible,
        compressed,
        ROWS=rows,
        FACTOR=int(config.index_factor),
        INV_SCALE=1.0 / scale,
        BLOCK=4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return cc, cs, bc, bs, visible, compressed
