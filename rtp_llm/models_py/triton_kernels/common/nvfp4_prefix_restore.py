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
    ACTIVE_TILES: tl.constexpr = False,
    PAGE_STRIDES: tl.constexpr = None,
    FI_WORKING_LAYOUT: tl.constexpr = False,
):
    # Preserve the original private-call ABI when strides are omitted.
    STRIDES: tl.constexpr = (M, M, S, S, I, T) if PAGE_STRIDES is None else PAGE_STRIDES
    if ACTIVE_TILES:
        # Exactly the disjoint tiles that copy bytes, rather than max-size
        # padding repeated for each of the six differently sized planes.
        nm, ns, ni, nt = tl.cdiv(M, TILE), tl.cdiv(S, TILE), tl.cdiv(I, TILE), tl.cdiv(T, TILE)
        tiles_per_row = 2 * nm + 2 * ns + ni + nt
        row = tl.program_id(0) // tiles_per_row
        active = tl.program_id(0) % tiles_per_row
        if active < nm:
            plane, tile = 0, active
        elif active < 2 * nm:
            plane, tile = 1, active - nm
        elif active < 2 * nm + ns:
            plane, tile = 2, active - 2 * nm
        elif active < 2 * nm + 2 * ns:
            plane, tile = 3, active - 2 * nm - ns
        elif active < 2 * nm + 2 * ns + ni:
            plane, tile = 4, active - 2 * nm - 2 * ns
        else:
            plane, tile = 5, active - 2 * nm - 2 * ns - ni
    else:
        row, tile, plane = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if plane == 0:
        source, output, stride, start, size = VALUES, K, VALUE_STRIDE, 0, M
        capacity = CAPACITIES[0]
        page_stride = STRIDES[0]
    elif plane == 1:
        source, output, stride, start, size = VALUES, V, VALUE_STRIDE, M, M
        capacity = CAPACITIES[1]
        page_stride = STRIDES[1]
    elif plane == 2:
        source, output, stride, start, size = SIDE, KS, SIDE_STRIDE, 0, S
        capacity = CAPACITIES[2]
        page_stride = STRIDES[2]
    elif plane == 3:
        source, output, stride, start, size = SIDE, VS, SIDE_STRIDE, S, S
        capacity = CAPACITIES[3]
        page_stride = STRIDES[3]
    elif plane == 4:
        source, output, stride, start, size = SIDE, IK, SIDE_STRIDE, 2 * S, I
        capacity = CAPACITIES[4]
        page_stride = STRIDES[4]
    else:
        source, output, stride, start, size = SIDE, IS, SIDE_STRIDE, 2 * S + I, T
        capacity = CAPACITIES[5]
        page_stride = STRIDES[5]
    if tile * TILE < size:
        if IDENTITY:
            src = row.to(tl.int64)
        else:
            src = tl.load(RESTORE + row).to(tl.int64)
        dst = tl.load(DST + row).to(tl.int64)
        if (src >= 0) & (src < SOURCE_ROWS) & (dst >= 0) & (dst < capacity):
            x = tile * TILE + tl.arange(0, TILE)
            if FI_WORKING_LAYOUT:
                if (plane == 2) | (plane == 3):
                    # Enumerate physical output bytes so both FI scale planes
                    # have contiguous stores. K is token-major. Inverse V's
                    # four-token permutation before gathering MMA source bytes.
                    head, physical = x // 1024, x % 1024
                    if plane == 2:
                        token, group = physical // 8, physical % 8
                    else:
                        token = (physical // 32) * 4 + physical % 4
                        group = (physical // 4) % 8
                    source_x = (head * 1024 + group // 4 * 512 + token % 32 * 16
                                + token // 32 * 4 + group % 4)
                    value = tl.load(source + src * stride + start + source_x, x < size, 0)
                    tl.store(output + dst * page_stride + x, value, x < size)
                else:
                    # Keep flat-copy x separate from permuted scale offsets.
                    # A merged source_x/output_x phi loses vectorization even
                    # for main values and index planes in the same kernel.
                    value = tl.load(source + src * stride + start + x, x < size, 0)
                    tl.store(output + dst * page_stride + x, value, x < size)
            else:
                value = tl.load(source + src * stride + start + x, x < size, 0)
                tl.store(output + dst * page_stride + x, value, x < size)



def restore_prefix_planes(
    values: torch.Tensor,
    side: torch.Tensor,
    restore_ids: Optional[torch.Tensor],
    dst_ids: torch.Tensor,
    planes: Tuple[torch.Tensor, ...],
    *,
    active_tiles: bool = False,
    fi_working_layout: bool = False,
) -> None:
    """Copy selected opaque page rows into six independent working planes.

    ``values`` stores [K, V]; ``side`` stores [K scale, V scale, index K,
    index scale]. Both are uint8 matrices with contiguous bytes within each
    row; trailing padding and larger row strides are supported. ``planes``
    follows that same six-plane order and contains page tensors with
    page as dimension 0. Capacity slices and independent base pointers work;
    one-byte scale dtypes are reinterpreted as uint8, never converted.

    CUDA int32/int64 index vectors must have equal lengths. ``restore_ids=None``
    means source row i (without allocating an identity vector). Repeated source
    IDs are legal; destination IDs must be unique and inputs/outputs must not
    overlap. Invalid indices are skipped, with destination bounds checked for
    each plane. The caller owns index validity and uniqueness: host validation
    only inspects metadata and never synchronizes device data. Tensor addresses
    and shapes can stay fixed while indices/content change during graph replay.
    No temporary device storage is allocated. ``active_tiles=True`` selects
    the experimental flattened launch with only byte-copying CTAs; the default
    retains the original rectangular launch for matched benchmarks.
    ``fi_working_layout=True`` accepts page-strided Hkv4/page128/D128 main
    planes and permutes persistent MMA scale bytes directly to linear K and
    FI-swizzled V working scales. Index bytes/scales remain opaque MMA data.
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
        if (plane.ndim < 2 or plane.element_size() != 1
                or (not plane.is_contiguous() and not fi_working_layout)):
            raise ValueError(
                "planes must be contiguous page tensors with one-byte elements"
            )
    for plane in planes:
        inner = 1
        for size, stride in zip(reversed(plane.shape[1:]), reversed(plane.stride()[1:])):
            if stride != inner:
                raise ValueError("planes require contiguous bytes within each page")
            inner *= size
        if plane.stride(0) < inner:
            raise ValueError("plane pages must not overlap")
    sizes = tuple(math.prod(p.shape[1:]) for p in planes)
    if fi_working_layout and sizes[:4] != (32768, 32768, 4096, 4096):
        raise ValueError("FI working restore requires Hkv4/page128/D128")
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
    tiles = tuple(triton.cdiv(size, 4096) for size in sizes)
    grid = ((dst_ids.numel() * sum(tiles),) if active_tiles
            else (dst_ids.numel(), max(tiles), 6))
    _restore_prefix_planes[grid](
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
        active_tiles,
        tuple(p.stride(0) for p in planes),
        fi_working_layout,
        num_warps=4,
    )
