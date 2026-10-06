"""Frozen pre-mapping d128 writer; test-only numerical oracle."""

import triton
import triton.language as tl
from nvfp4_cp_production_test_module import (
    _TL_E2M1_MAX,
    _TL_SCALE_MAX,
    _TL_SCALE_MIN,
    _e2m1_encode,
    _scale_128x4_offset,
)


@triton.jit
def _quantize_main_index_rows_d128_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    slots_ptr,
    k_packed_ptr,
    k_scales_ptr,
    v_packed_ptr,
    v_scales_ptr,
    idx_packed_ptr,
    idx_scales_ptr,
    N,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    K_S2: tl.constexpr,
    V_S0: tl.constexpr,
    V_S1: tl.constexpr,
    V_S2: tl.constexpr,
    IDX_S0: tl.constexpr,
    IDX_S2: tl.constexpr,
    MAIN_PACKED_S0: tl.constexpr,
    MAIN_SCALE_S0: tl.constexpr,
    IDX_PACKED_S0: tl.constexpr,
    IDX_SCALE_S0: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MMA_SCALE_LAYOUT: tl.constexpr,
):
    """One row/plane CTA, eight independent groups by eight nibble pairs.

    Reduction axis 1 preserves each group of sixteen. Scale stores are 1D;
    packed byte stores are 2D. Row and slot offsets promote to int64.
    """
    row = tl.program_id(0).to(tl.int64)
    plane = tl.program_id(1)
    is_k = plane < NUM_HEADS
    is_v = (plane >= NUM_HEADS) & (plane < 2 * NUM_HEADS)
    is_idx = plane == 2 * NUM_HEADS
    head = tl.where(is_k, plane, tl.where(is_v, plane - NUM_HEADS, 0))

    valid_row = row < N
    slot = tl.load(slots_ptr + row, mask=valid_row, other=-1).to(tl.int64)
    valid_slot = valid_row & (slot >= 0) & (slot < NUM_BLOCKS * PAGE_SIZE)
    block = slot // PAGE_SIZE
    page_offset = slot - block * PAGE_SIZE
    pair = tl.arange(0, 8)[None, :]
    groups = tl.arange(0, 8)
    group = groups[:, None]
    element = group * 16 + 2 * pair
    main_even_offset = row * K_S0 + head * K_S1 + element * K_S2
    main_odd_offset = main_even_offset + K_S2
    v_even_offset = row * V_S0 + head * V_S1 + element * V_S2
    v_odd_offset = v_even_offset + V_S2
    idx_even_offset = row * IDX_S0 + element * IDX_S2
    idx_odd_offset = idx_even_offset + IDX_S2

    k_even = tl.load(k_ptr + main_even_offset, mask=valid_slot & is_k, other=0.0)
    k_odd = tl.load(k_ptr + main_odd_offset, mask=valid_slot & is_k, other=0.0)
    v_even = tl.load(v_ptr + v_even_offset, mask=valid_slot & is_v, other=0.0)
    v_odd = tl.load(v_ptr + v_odd_offset, mask=valid_slot & is_v, other=0.0)
    idx_even = tl.load(idx_ptr + idx_even_offset, mask=valid_slot & is_idx, other=0.0)
    idx_odd = tl.load(idx_ptr + idx_odd_offset, mask=valid_slot & is_idx, other=0.0)
    even_values = tl.where(is_k, k_even, tl.where(is_v, v_even, idx_even)).to(
        tl.float32
    )
    odd_values = tl.where(is_k, k_odd, tl.where(is_v, v_odd, idx_odd)).to(tl.float32)
    amax = tl.max(tl.maximum(tl.abs(even_values), tl.abs(odd_values)), axis=1)
    raw_scale = tl.minimum(
        tl.maximum(amax / _TL_E2M1_MAX, _TL_SCALE_MIN), _TL_SCALE_MAX
    )
    stored_scale = raw_scale.to(tl.float8e4nv)
    scale = stored_scale.to(tl.float32)[:, None]
    packed_codes = _e2m1_encode(even_values, scale) | (
        _e2m1_encode(odd_values, scale) << 4
    )

    main_packed_offset = (
        block * MAIN_PACKED_S0
        + (head * PAGE_SIZE + page_offset) * 64
        + group * 8
        + pair
    )
    if MMA_SCALE_LAYOUT:
        main_scale_offset = (
            block * MAIN_SCALE_S0
            + head * PAGE_SIZE * 8
            + _scale_128x4_offset(page_offset, groups)
        )
    else:
        main_scale_offset = (
            block * MAIN_SCALE_S0 + (head * PAGE_SIZE + page_offset) * 8 + groups
        )
    idx_packed_offset = block * IDX_PACKED_S0 + page_offset * 64 + group * 8 + pair
    if MMA_SCALE_LAYOUT:
        idx_scale_offset = block * IDX_SCALE_S0 + _scale_128x4_offset(
            page_offset, groups
        )
    else:
        idx_scale_offset = block * IDX_SCALE_S0 + page_offset * 8 + groups

    tl.store(
        k_packed_ptr + main_packed_offset,
        packed_codes,
        mask=valid_slot & is_k,
    )
    tl.store(
        k_scales_ptr + main_scale_offset,
        stored_scale,
        mask=valid_slot & is_k,
    )
    tl.store(
        v_packed_ptr + main_packed_offset,
        packed_codes,
        mask=valid_slot & is_v,
    )
    tl.store(
        v_scales_ptr + main_scale_offset,
        stored_scale,
        mask=valid_slot & is_v,
    )
    tl.store(
        idx_packed_ptr + idx_packed_offset,
        packed_codes,
        mask=valid_slot & is_idx,
    )
    tl.store(
        idx_scales_ptr + idx_scale_offset,
        stored_scale,
        mask=valid_slot & is_idx,
    )


def write_reference(k, v, idx, slots, planes):
    rows = slots.numel()
    if not rows:
        return
    kp, ks, vp, vs, ip, ix = planes
    _quantize_main_index_rows_d128_kernel[(rows, 9)](
        k,
        v,
        idx,
        slots,
        *planes,
        rows,
        K_S0=k.stride(0),
        K_S1=k.stride(1),
        K_S2=k.stride(2),
        V_S0=v.stride(0),
        V_S1=v.stride(1),
        V_S2=v.stride(2),
        IDX_S0=idx.stride(0),
        IDX_S2=idx.stride(2),
        MAIN_PACKED_S0=kp.stride(0),
        MAIN_SCALE_S0=ks.stride(0),
        IDX_PACKED_S0=ip.stride(0),
        IDX_SCALE_S0=ix.stride(0),
        NUM_BLOCKS=kp.shape[0],
        NUM_HEADS=4,
        PAGE_SIZE=128,
        MMA_SCALE_LAYOUT=True,
        num_warps=1,
    )
