"""MiniMax-M3.1 fused NVFP4 writer for independent main and indexer pools."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.model_desc.minimax_m31_cache import M31CacheViews


@triton.jit
def _e2m1_encode(values, scale):
    magnitude = tl.abs(values)
    code = tl.where(
        magnitude > 5.0 * scale,
        7,
        tl.where(
            magnitude >= 3.5 * scale,
            6,
            tl.where(
                magnitude > 2.5 * scale,
                5,
                tl.where(
                    magnitude >= 1.75 * scale,
                    4,
                    tl.where(
                        magnitude > 1.25 * scale,
                        3,
                        tl.where(
                            magnitude >= 0.75 * scale,
                            2,
                            tl.where(magnitude > 0.25 * scale, 1, 0),
                        ),
                    ),
                ),
            ),
        ),
    ).to(tl.uint8)
    return code | tl.where((values < 0.0) & (code != 0), 8, 0).to(tl.uint8)


@triton.jit
def _scale_128x4_offset(row, group):
    return (group // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + group % 4


@triton.jit(do_not_specialize=["ROWS", "MAIN_BLOCKS", "INDEX_BLOCKS"])
def _write_m31_nvfp4_rows(
    k_ptr,
    v_ptr,
    index_ptr,
    main_slots_ptr,
    index_slots_ptr,
    k_packed_ptr,
    k_scale_ptr,
    v_packed_ptr,
    v_scale_ptr,
    index_packed_ptr,
    index_scale_ptr,
    ROWS,
    MAIN_BLOCKS,
    INDEX_BLOCKS,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    I_S0: tl.constexpr,
    I_S1: tl.constexpr,
    MP_S0: tl.constexpr,
    MS_S0: tl.constexpr,
    IP_S0: tl.constexpr,
    IS_S0: tl.constexpr,
    HEADS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    plane = tl.program_id(1)
    is_k = plane < HEADS
    is_v = (plane >= HEADS) & (plane < 2 * HEADS)
    is_index = plane == 2 * HEADS
    head = tl.where(is_k, plane, tl.where(is_v, plane - HEADS, 0))
    main_slot = tl.load(main_slots_ptr + row, mask=row < ROWS, other=-1).to(tl.int64)
    index_slot = tl.load(index_slots_ptr + row, mask=row < ROWS, other=-1).to(tl.int64)
    main_valid = (row < ROWS) & (main_slot >= 0) & (main_slot < MAIN_BLOCKS * 128)
    index_valid = (row < ROWS) & (index_slot >= 0) & (index_slot < INDEX_BLOCKS * 128)
    slot = tl.where(is_index, index_slot, main_slot)
    valid = tl.where(is_index, index_valid, main_valid)
    block = slot // 128
    page_row = slot % 128
    groups = tl.arange(0, 8)
    group = groups[:, None]
    pair = tl.arange(0, 8)[None, :]
    element = group * 16 + pair * 2

    k_offset = row * K_S0 + head * K_S1 + element * K_S2
    v_offset = row * V_S0 + head * V_S1 + element * V_S2
    i_offset = row * I_S0 + element * I_S1
    even = tl.where(
        is_k,
        tl.load(k_ptr + k_offset, mask=main_valid & is_k, other=0),
        tl.where(
            is_v,
            tl.load(v_ptr + v_offset, mask=main_valid & is_v, other=0),
            tl.load(index_ptr + i_offset, mask=index_valid & is_index, other=0),
        ),
    ).to(tl.float32)
    odd = tl.where(
        is_k,
        tl.load(k_ptr + k_offset + K_S2, mask=main_valid & is_k, other=0),
        tl.where(
            is_v,
            tl.load(v_ptr + v_offset + V_S2, mask=main_valid & is_v, other=0),
            tl.load(index_ptr + i_offset + I_S1, mask=index_valid & is_index, other=0),
        ),
    ).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even), tl.abs(odd)), 1)
    raw_scale = tl.minimum(tl.maximum(amax / 6.0, 1.0 / 512.0), 448.0)
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)[:, None]
    packed = _e2m1_encode(even, scale) | (_e2m1_encode(odd, scale) << 4)

    main_byte = block * MP_S0 + (head * 128 + page_row) * 64 + group * 8 + pair
    main_sf = block * MS_S0 + head * 1024 + _scale_128x4_offset(page_row, groups)
    index_byte = block * IP_S0 + page_row * 64 + group * 8 + pair
    index_sf = block * IS_S0 + _scale_128x4_offset(page_row, groups)
    tl.store(k_packed_ptr + main_byte, packed, mask=valid & is_k)
    tl.store(k_scale_ptr + main_sf, stored_scale, mask=valid & is_k)
    tl.store(v_packed_ptr + main_byte, packed, mask=valid & is_v)
    tl.store(v_scale_ptr + main_sf, stored_scale, mask=valid & is_v)
    tl.store(index_packed_ptr + index_byte, packed, mask=valid & is_index)
    tl.store(index_scale_ptr + index_sf, stored_scale, mask=valid & is_index)


def write_m31_nvfp4_rows(
    k: torch.Tensor,
    v: torch.Tensor,
    index_k: torch.Tensor,
    main_slots: torch.Tensor,
    index_slots: torch.Tensor,
    views: M31CacheViews,
) -> None:
    """Write token rows to independently allocated main/indexer physical slots."""
    if k.ndim != 3 or v.shape != k.shape or k.shape[-1] != 128:
        raise ValueError("M3.1 K/V must be matching [rows,heads,128] tensors")
    rows, heads, _ = k.shape
    if index_k.shape != (rows, 128):
        raise ValueError("M3.1 index K must be [rows,128]")
    if k.dtype != torch.bfloat16 or v.dtype != k.dtype or index_k.dtype != k.dtype:
        raise ValueError("M3.1 writer requires BF16 K/V/index K")
    if any(
        slots.shape != (rows,) or slots.dtype != torch.int64
        for slots in (main_slots, index_slots)
    ):
        raise ValueError("M3.1 writer requires separate int64 slots per row")
    tensors = (k, v, index_k, main_slots, index_slots)
    if any(t.device != k.device for t in tensors):
        raise ValueError("M3.1 writer inputs must share a device")
    if not k.is_cuda:
        raise ValueError("M3.1 NVFP4 writer requires CUDA")
    if any(t.stride(-1) != 1 for t in (k, v, index_k)):
        raise ValueError("M3.1 writer inputs require unit inner stride")
    if views.main_k.shape != (views.main_v.shape[0], heads * 8192):
        raise ValueError("M3.1 main KV view shape does not match heads")
    if views.index_k.shape[1] != 8192:
        raise ValueError("M3.1 indexer view shape is invalid")
    if rows == 0:
        return
    _write_m31_nvfp4_rows[(rows, 2 * heads + 1)](
        k,
        v,
        index_k,
        main_slots,
        index_slots,
        views.main_k,
        views.main_k_scale,
        views.main_v,
        views.main_v_scale,
        views.index_k,
        views.index_k_scale,
        rows,
        views.main_k.shape[0],
        views.index_k.shape[0],
        *k.stride(),
        *v.stride(),
        *index_k.stride(),
        views.main_k.stride(0),
        views.main_k_scale.stride(0),
        views.index_k.stride(0),
        views.index_k_scale.stride(0),
        heads,
        num_warps=1,
    )
