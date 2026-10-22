"""Fuse draft RoPE generation/application, packed layout and score metadata.

Uses the same libdevice trig and two BF16 product rounding boundaries as the
qualified target preparation and rope_triton kernels. Read-only preparation;
the separately validated writer still owns side-cache mutation and undo.
"""

import math

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
    FROM_METADATA: tl.constexpr = False,
    ROW_LENGTHS=None,
    PREFIX=None,
    CU=None,
    BATCH: tl.constexpr = 0,
    RB: tl.constexpr = 1,
    RAW_QUERY: tl.constexpr = False,
    QG=None,
    EPS: tl.constexpr = 1e-6,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, 128)
    if FROM_METADATA:
        request, lane = row // 4, row % 4
        count = tl.load(ROW_LENGTHS + request)
        active = lane < count
        source = tl.where(active, tl.load(CU + request) + lane, 0)
        requests = tl.arange(0, RB)
        ends = tl.load(CU + requests + 1, requests < BATCH, other=2147483647)
        first_owner = tl.minimum(
            tl.sum(((0 >= ends) & (requests < BATCH)).to(tl.int32), 0), BATCH - 1
        )
        total = tl.load(CU + BATCH)
        first_position = tl.where(
            total > 0, tl.load(PREFIX + first_owner).to(tl.int64), 0
        )
        prefix = tl.load(PREFIX + request).to(tl.int64)
        position = tl.where(active, prefix + lane, first_position).to(tl.float32)
    else:
        source = tl.load(SOURCES + row)
        position = tl.load(POS + source).to(tl.float32)
    frequency = tl.load(FREQ + col % 32)
    angle = (position * INV_SCALE) * frequency
    cosine = libdevice.cos(angle).to(tl.bfloat16).to(tl.float32)
    sine = libdevice.sin(angle).to(tl.bfloat16).to(tl.float32)
    offset = source * QS + head * QH
    if RAW_QUERY:
        norm_lane = tl.arange(0, 32)
        norm_col = norm_lane * 4
        a = tl.load(Q + offset + norm_col).to(tl.float32)
        b = tl.load(Q + offset + norm_col + 1).to(tl.float32)
        c = tl.load(Q + offset + norm_col + 2).to(tl.float32)
        d = tl.load(Q + offset + norm_col + 3).to(tl.float32)
        variance = tl.sum(((a * a + b * b) + c * c) + d * d, 0) / 128.0
        inverse = tl.rsqrt(variance + EPS)
        raw = tl.load(Q + offset + col).to(tl.float32)
        gain = 1.0 + tl.load(QG + col).to(tl.float32)
        value = ((raw * inverse) * gain).to(tl.bfloat16)
        other = tl.gather(value, (col + 32) % 64, 0).to(tl.float32)
    else:
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
        if FROM_METADATA:
            owner = tl.minimum(
                tl.sum(((row >= ends) & (requests < BATCH)).to(tl.int32), 0), BATCH - 1
            )
            logical = tl.where(
                row < total,
                tl.load(PREFIX + owner).to(tl.int64)
                + row
                - tl.load(CU + owner).to(tl.int64),
                0,
            )
        else:
            logical = tl.load(POS + row).to(tl.int64)
        block_start = (logical - logical % 4).to(tl.float32)
        block_angle = (block_start * INV_SCALE) * frequency
        tl.store(COS + row * 64 + col, libdevice.cos(block_angle), col < 64)
        tl.store(SIN + row * 64 + col, libdevice.sin(block_angle), col < 64)
        if FROM_METADATA:
            visible = tl.where(active, prefix + lane + 1, 0).to(tl.int32)
            tl.store(VISIBLE + row, visible)
        else:
            visible = tl.load(VISIBLE + row)
        tl.store(LENGTHS + row, visible // 4)


def try_prepare(q, positions, sources, visible, frequency, *, rope_config):
    from .indexer import is_qsa_rope_style

    scale = float(getattr(rope_config, "scale", 1.0))
    # The caller retains graph metadata/ownership validation before mutation.
    if not (
        q.is_cuda
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


def try_prepare_metadata(
    q, row_lengths, prefixes, cu, frequency, *, rope_config, q_gamma=None, norm_eps=1e-6
):
    """Prepare fixed four-lane queries directly from owner-validated metadata.

    If q_gamma is supplied, consume raw Q and preserve the exact RMSNorm
    reduction and BF16 boundary before RoPE. Unsupported inputs return None
    before any writes; the owner normalizes Q before using its original path.
    """
    from .indexer import is_qsa_rope_style

    scale = float(getattr(rope_config, "scale", 1.0))
    if not (
        q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (12, 0)
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and q.stride(-1) == 1
        and q.stride(-2) == 128
        and q.stride(0) >= 512
        and row_lengths.ndim == prefixes.ndim == cu.ndim == 1
        and 1 <= row_lengths.numel() <= 8
        and q.shape[0] == row_lengths.numel() * 4
        and prefixes.shape == row_lengths.shape
        and cu.numel() == row_lengths.numel() + 1
        and all(
            t.dtype == torch.int32 and t.device == q.device and t.is_contiguous()
            for t in (row_lengths, prefixes, cu)
        )
        and isinstance(frequency, torch.Tensor)
        and frequency.shape == (32,)
        and frequency.dtype == torch.float32
        and frequency.device == q.device
        and frequency.is_contiguous()
        and not frequency.requires_grad
        and not q.requires_grad
        and math.isfinite(scale)
        and scale > 0
        and int(rope_config.dim) == 64
        and int(rope_config.index_factor) in (1, 3)
        and is_qsa_rope_style(rope_config, "Base")
        and bool(getattr(rope_config, "indexer_is_neox_style", True))
        and (
            q_gamma is None
            or (
                q_gamma.shape == (128,)
                and q_gamma.dtype in (torch.bfloat16, torch.float32)
                and q_gamma.device == q.device
                and q_gamma.is_contiguous()
                and not q_gamma.requires_grad
                and isinstance(norm_eps, (float, int))
                and math.isfinite(norm_eps)
                and norm_eps >= 0
            )
        )
    ):
        return None
    batch, rows = row_lengths.numel(), q.shape[0]
    padded = torch.empty((batch, 4, 4, 128), dtype=q.dtype, device=q.device)
    cosine = torch.empty((rows, 64), dtype=q.dtype, device=q.device)
    sine = torch.empty_like(cosine)
    weights = torch.empty((rows, 4), dtype=torch.float32, device=q.device)
    lengths = torch.empty((batch, 4), dtype=torch.int32, device=q.device)
    visible = torch.empty_like(lengths)
    _prepare_draft_qsa[(rows, 4)](
        q,
        None,
        None,
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
        FROM_METADATA=True,
        ROW_LENGTHS=row_lengths,
        PREFIX=prefixes,
        CU=cu,
        BATCH=batch,
        RB=triton.next_power_of_2(batch),
        RAW_QUERY=q_gamma is not None,
        QG=q_gamma,
        EPS=float(norm_eps) if q_gamma is not None else 1e-6,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return padded, cosine, sine, weights, lengths, visible
