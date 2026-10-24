"""MiniMax-M3.1 CP suffix transport using the existing NVFP4 cache encoding."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.model_desc.minimax_m31_kv_write import (
    _e2m1_encode,
    _scale_128x4_offset,
)

_WIRE_BYTES = 648


@triton.jit(do_not_specialize=["ROWS"])
def _pack_m31_cp_wire(packed, wire, ROWS, S0: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    plane = tl.program_id(1)
    groups = tl.arange(0, 8)
    pair = tl.arange(0, 8)[None, :]
    element = groups[:, None] * 16 + pair * 2
    offset = row * S0 + plane * 128 + element
    even = tl.load(packed + offset, mask=row < ROWS, other=0).to(tl.float32)
    odd = tl.load(packed + offset + 1, mask=row < ROWS, other=0).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even), tl.abs(odd)), 1)
    raw_scale = tl.minimum(tl.maximum(amax / 6.0, 1.0 / 512.0), 448.0)
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)[:, None]
    codes = _e2m1_encode(even, scale) | (_e2m1_encode(odd, scale) << 4)
    base = row * 648 + plane * 72
    tl.store(wire + base + groups[:, None] * 8 + pair, codes, mask=row < ROWS)
    tl.store(
        wire + base + 64 + groups,
        stored_scale.to(tl.uint8, bitcast=True),
        mask=row < ROWS,
    )


@triton.jit(do_not_specialize=["ROWS", "SOURCE_ROWS", "MAIN_PAGES", "INDEX_PAGES"])
def _write_m31_cp_wire(
    wire,
    restore,
    slots,
    main_k,
    main_ks,
    main_v,
    main_vs,
    index_k,
    index_ks,
    ROWS,
    SOURCE_ROWS,
    MAIN_PAGES,
    INDEX_PAGES,
    MAIN_S0: tl.constexpr,
    MAIN_SF0: tl.constexpr,
    INDEX_S0: tl.constexpr,
    INDEX_SF0: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    plane = tl.program_id(1)
    head = tl.where(plane < 4, plane, tl.where(plane < 8, plane - 4, 0))
    source = tl.load(restore + row, mask=row < ROWS, other=-1).to(tl.int64)
    slot = tl.load(slots + row, mask=row < ROWS, other=-1).to(tl.int64)
    valid_source = (row < ROWS) & (source >= 0) & (source < SOURCE_ROWS)
    is_main = plane < 8
    valid = valid_source & tl.where(
        is_main,
        (slot >= 0) & (slot < MAIN_PAGES * 128),
        (slot >= 0) & (slot < INDEX_PAGES * 128),
    )
    groups = tl.arange(0, 8)
    pair = tl.arange(0, 8)[None, :]
    byte = groups[:, None] * 8 + pair
    base = source * 648 + plane * 72
    codes = tl.load(wire + base + byte, mask=valid, other=0)
    scales = tl.load(wire + base + 64 + groups, mask=valid, other=0)
    page = slot // 128
    page_row = slot % 128
    main_byte = page * MAIN_S0 + (head * 128 + page_row) * 64 + byte
    main_scale = page * MAIN_SF0 + head * 1024 + _scale_128x4_offset(page_row, groups)
    index_byte = page * INDEX_S0 + page_row * 64 + byte
    index_scale = page * INDEX_SF0 + _scale_128x4_offset(page_row, groups)
    tl.store(main_k + main_byte, codes, mask=valid & (plane < 4))
    tl.store(main_ks + main_scale, scales, mask=valid & (plane < 4))
    tl.store(main_v + main_byte, codes, mask=valid & (plane >= 4) & (plane < 8))
    tl.store(main_vs + main_scale, scales, mask=valid & (plane >= 4) & (plane < 8))
    tl.store(index_k + index_byte, codes, mask=valid & (plane == 8))
    tl.store(index_ks + index_scale, scales, mask=valid & (plane == 8))


def pack_m31_cp_wire(packed: torch.Tensor) -> torch.Tensor:
    if (
        packed.ndim != 2
        or packed.shape[1] != 1152
        or packed.dtype != torch.bfloat16
        or not packed.is_cuda
        or not packed.is_contiguous()
    ):
        raise ValueError("M3.1 CP wire requires contiguous CUDA BF16 [rows,1152]")
    wire = torch.empty(
        (packed.shape[0], _WIRE_BYTES), device=packed.device, dtype=torch.uint8
    )
    if packed.shape[0]:
        _pack_m31_cp_wire[(packed.shape[0], 9)](
            packed, wire, packed.shape[0], packed.stride(0), num_warps=1
        )
    return wire


def write_m31_cp_wire(
    wire: torch.Tensor,
    restore: torch.Tensor,
    slots: torch.Tensor,
    views,
) -> None:
    rows = restore.numel()
    if (
        wire.ndim != 2
        or wire.shape[1] != _WIRE_BYTES
        or wire.dtype != torch.uint8
        or not wire.is_contiguous()
        or restore.shape != (rows,)
        or restore.dtype != torch.int64
        or slots.shape != (rows,)
        or slots.dtype != torch.int64
        or not wire.is_cuda
        or any(
            x.device != wire.device
            for x in (restore, slots, views.main_k, views.index_k)
        )
    ):
        raise ValueError("M3.1 CP wire inputs must share CUDA geometry and device")
    if rows:
        _write_m31_cp_wire[(rows, 9)](
            wire,
            restore,
            slots,
            views.main_k,
            views.main_k_scale.view(torch.uint8),
            views.main_v,
            views.main_v_scale.view(torch.uint8),
            views.index_k,
            views.index_k_scale.view(torch.uint8),
            rows,
            wire.shape[0],
            views.main_k.shape[0],
            views.index_k.shape[0],
            views.main_k.stride(0),
            views.main_k_scale.stride(0),
            views.index_k.stride(0),
            views.index_k_scale.stride(0),
            num_warps=1,
        )
