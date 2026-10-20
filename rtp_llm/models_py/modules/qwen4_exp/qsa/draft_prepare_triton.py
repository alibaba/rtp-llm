"""Fuse draft RoPE generation/application, packed layout and score metadata.

Uses the same libdevice trig and two BF16 product rounding boundaries as the
qualified target preparation and rope_triton kernels. Read-only preparation;
the separately validated writer still owns side-cache mutation and undo.
"""

import math
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _prepare_draft_qsa(
    Q,
    POS,
    SOURCES,
    VISIBLE,
    FREQ,
    PADDED,
    COS,
    SIN,
    WEIGHTS,
    LENGTHS,
    QS: tl.constexpr,
    QH: tl.constexpr,
    H: tl.constexpr,
    INV_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, 128)
    source = tl.load(SOURCES + row)
    position = tl.load(POS + source).to(tl.float32)
    frequency = tl.load(FREQ + col % 32)
    angle = (position * INV_SCALE) * frequency
    cosine = libdevice.cos(angle).to(tl.bfloat16).to(tl.float32)
    sine = libdevice.sin(angle).to(tl.bfloat16).to(tl.float32)
    offset = source * QS + head * QH
    value = tl.load(Q + offset + col)
    partner = tl.where(col < 32, col + 32, col - 32)
    other = tl.load(Q + offset + partner, col < 64, other=0).to(tl.float32)
    rotated = tl.where(col < 32, -other, other)
    first = (value.to(tl.float32) * cosine).to(tl.bfloat16).to(tl.float32)
    second = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    result = tl.where(col < 64, (first + second).to(tl.bfloat16), value)
    tl.store(PADDED + (row * H + head) * 128 + col, result)
    tl.store(WEIGHTS + row * H + head, 0.08838834764831845)
    if head == 0:
        # Writer RoPE remains in packed order, unlike the padded scorer query.
        logical = tl.load(POS + row).to(tl.int64)
        block_start = (logical - logical % 4).to(tl.float32)
        block_angle = (block_start * INV_SCALE) * frequency
        tl.store(COS + row * 64 + col, libdevice.cos(block_angle), col < 64)
        tl.store(SIN + row * 64 + col, libdevice.sin(block_angle), col < 64)
        tl.store(LENGTHS + row, tl.load(VISIBLE + row) // 4)


def try_prepare(q, positions, sources, visible, frequency, *, rope_config):
    from .indexer import is_qsa_rope_style

    scale = float(getattr(rope_config, "scale", 1.0))
    # The caller retains graph metadata/ownership validation before mutation.
    if not (
        os.environ.get("QWEN4_DRAFT_QSA_FUSED", "0") == "1"
        and q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (12, 0)
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and q.stride(-1) == 1
        and q.stride(-2) == 128
        and q.stride(0) >= 512
        and 1 <= q.shape[0] <= 32
        and positions.shape == sources.shape == (q.shape[0],)
        and positions.dtype == sources.dtype == torch.int64
        and visible.dtype == torch.int32
        and visible.ndim == 2
        and visible.shape[1] == 4
        and visible.numel() == q.shape[0]
        and isinstance(frequency, torch.Tensor)
        and frequency.dtype == torch.float32
        and frequency.shape == (32,)
        and all(
            x.device == q.device and x.is_contiguous()
            for x in (positions, sources, visible, frequency)
        )
        and math.isfinite(scale)
        and scale > 0
        and int(rope_config.dim) == 64
        and int(rope_config.index_factor) in (1, 3)
        and is_qsa_rope_style(rope_config, "Base")
        and bool(getattr(rope_config, "indexer_is_neox_style", True))
        and not q.requires_grad
    ):
        return None
    padded = torch.empty((*visible.shape, 4, 128), dtype=q.dtype, device=q.device)
    cosine = torch.empty((q.shape[0], 64), dtype=q.dtype, device=q.device)
    sine = torch.empty_like(cosine)
    weights = torch.empty((q.shape[0], 4), dtype=torch.float32, device=q.device)
    lengths = torch.empty_like(visible)
    _prepare_draft_qsa[(q.shape[0], 4)](
        q,
        positions,
        sources,
        visible,
        frequency,
        padded,
        cosine,
        sine,
        weights,
        lengths,
        QS=q.stride(0),
        QH=q.stride(1),
        H=4,
        INV_SCALE=1.0 / scale,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return padded, cosine, sine, weights, lengths
