"""Group32 activation encoding for the V4.1 dense GEMM boundary."""

import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._compact_writer_triton import (
    _round_power_of_two_scale,
)


@triton.jit
def quantize_block32_vector_kernel(
    values,
    output,
    scales,
    rows,
    K: tl.constexpr,
    PACKED: tl.constexpr,
    PACKS: tl.constexpr,
    SWIZZLED: tl.constexpr = False,
):
    packs = tl.program_id(0) * PACKS + tl.arange(0, PACKS)
    groups = packs[:, None] * 4 + tl.arange(0, 4)[None, :]
    offsets = groups[:, :, None] * 32 + tl.arange(0, 32)[None, None, :]
    active = offsets < rows * K
    x = tl.load(values + offsets, active, 0).to(tl.float32)
    finite = tl.abs(x) < float("inf")
    x = tl.where(finite, x, 0.0)
    maximum = tl.maximum(tl.max(tl.abs(x), axis=2), 1.0e-4)
    scale, exponent = _round_power_of_two_scale(maximum, 1.0 / 448.0)
    encoded = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, :, None]), -448.0), 448.0)
    tl.store(
        output + offsets,
        encoded.to(tl.float8e4nv, fp_downcast_rounding="rtne"),
        active,
    )
    if PACKED or SWIZZLED:
        packed = tl.sum(
            exponent.to(tl.uint32) << (8 * tl.arange(0, 4)[None, :]), axis=1
        )
        row = packs // (K // 128)
        column = packs % (K // 128)
        if SWIZZLED:
            # F8_128x4 stores four adjacent K scales in one uint32.
            offset = (
                ((row // 128) * (K // 128) + column) * 128
                + (row % 32) * 4
                + (row % 128) // 32
            )
            tl.store(
                scales.to(tl.pointer_type(tl.uint32)) + offset,
                tl.where(row < rows, packed, 0),
                row < tl.cdiv(rows, 128) * 128,
            )
        else:
            tl.store(scales + row + column * tl.cdiv(rows, 4) * 4, packed, row < rows)
    else:
        tl.store(scales + groups, scale, groups < rows * (K // 32))
