"""Restore opaque NVFP4 prefix pages without materializing gathered planes."""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _restore_prefix_planes(
    VALUES,
    SIDE,
    RESTORE,
    DST,
    K,
    V,
    KS,
    VS,
    IK,
    IS,
    VALUE_STRIDE: tl.constexpr,
    SIDE_STRIDE: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    CAPACITIES: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    I: tl.constexpr,
    T: tl.constexpr,
    IDENTITY: tl.constexpr,
    TILE: tl.constexpr,
):
    row, tile, plane = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if plane == 0:
        source, output, stride, start, size = VALUES, K, VALUE_STRIDE, 0, M
        capacity = CAPACITIES[0]
    elif plane == 1:
        source, output, stride, start, size = VALUES, V, VALUE_STRIDE, M, M
        capacity = CAPACITIES[1]
    elif plane == 2:
        source, output, stride, start, size = SIDE, KS, SIDE_STRIDE, 0, S
        capacity = CAPACITIES[2]
    elif plane == 3:
        source, output, stride, start, size = SIDE, VS, SIDE_STRIDE, S, S
        capacity = CAPACITIES[3]
    elif plane == 4:
        source, output, stride, start, size = SIDE, IK, SIDE_STRIDE, 2 * S, I
        capacity = CAPACITIES[4]
    else:
        source, output, stride, start, size = SIDE, IS, SIDE_STRIDE, 2 * S + I, T
        capacity = CAPACITIES[5]
    if tile * TILE < size:
        if IDENTITY:
            src = row.to(tl.int64)
        else:
            src = tl.load(RESTORE + row).to(tl.int64)
        dst = tl.load(DST + row).to(tl.int64)
        if (src >= 0) & (src < SOURCE_ROWS) & (dst >= 0) & (dst < capacity):
            x = tile * TILE + tl.arange(0, TILE)
            value = tl.load(source + src * stride + start + x, x < size, 0)
            tl.store(output + dst * size + x, value, x < size)


def restore_prefix_planes(
    values: torch.Tensor,
    side: torch.Tensor,
    restore_ids: Optional[torch.Tensor],
    dst_ids: torch.Tensor,
    planes: Tuple[torch.Tensor, ...],
) -> None:
    """Copy selected opaque page rows into six independent working planes.

    ``values`` stores [K, V]; ``side`` stores [K scale, V scale, index K,
    index scale]. Both are uint8 matrices with contiguous bytes within each
    row; trailing padding and larger row strides are supported. ``planes``
    follows that same six-plane order and contains contiguous tensors with
    page as dimension 0. Capacity slices and independent base pointers work;
    one-byte scale dtypes are reinterpreted as uint8, never converted.

    CUDA int32/int64 index vectors must have equal lengths. ``restore_ids=None``
    means source row i (without allocating an identity vector). Repeated source
    IDs are legal; destination IDs must be unique and inputs/outputs must not
    overlap. Invalid indices are skipped, with destination bounds checked for
    each plane. The caller owns index validity and uniqueness: host validation
    only inspects metadata and never synchronizes device data. Tensor addresses
    and shapes can stay fixed while indices/content change during graph replay.
    No temporary device storage is allocated.
    """
    if len(planes) != 6:
        raise ValueError("expected six destination planes")
    for source in (values, side):
        if source.ndim != 2 or source.dtype != torch.uint8:
            raise ValueError("sources must be two-dimensional uint8 tensors")
        if source.stride(1) != 1 or source.stride(0) < source.shape[1]:
            raise ValueError(
                "source rows must contain contiguous, nonoverlapping bytes"
            )
    if values.shape[0] != side.shape[0]:
        raise ValueError("source row counts must match")
    indices = (dst_ids,) if restore_ids is None else (restore_ids, dst_ids)
    for ids in indices:
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be one-dimensional int32/int64 tensors")
        if not ids.is_contiguous():
            raise ValueError("indices must be contiguous")
    if restore_ids is not None and restore_ids.shape != dst_ids.shape:
        raise ValueError("source and destination index counts must match")
    for plane in planes:
        if plane.ndim < 2 or plane.element_size() != 1 or not plane.is_contiguous():
            raise ValueError(
                "planes must be contiguous page tensors with one-byte elements"
            )
    sizes = tuple(math.prod(p.shape[1:]) for p in planes)
    m, v, s, vs, i, t = sizes
    if min(sizes) <= 0 or m != v or s != vs:
        raise ValueError("plane sizes must be positive with matching K/V sizes")
    if values.shape[1] < 2 * m or side.shape[1] < 2 * s + i + t:
        raise ValueError("source row is too short for destination planes")
    tensors = (values, side, *indices, *planes)
    if not values.is_cuda or any(x.device != values.device for x in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if dst_ids.numel() == 0:
        return
    _restore_prefix_planes[(dst_ids.numel(), triton.cdiv(max(sizes), 4096), 6)](
        values,
        side,
        restore_ids,
        dst_ids,
        *(p.view(torch.uint8) for p in planes),
        values.stride(0),
        side.stride(0),
        values.shape[0],
        tuple(p.shape[0] for p in planes),
        m,
        s,
        i,
        t,
        restore_ids is None,
        4096,
        num_warps=4,
    )
