# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-local gather from registered shared-host Engram FP8/UE8M0 tables."""

import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._compact_writer_triton import (
    _round_power_of_two_scale,
)


@triton.jit
def engram_gather_kernel(
    Weight,
    Scale,
    Indices,
    Valid,
    Output,
    ROWS: tl.constexpr,
    DIM: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    row = tl.load(Indices + query).to(tl.int64)
    active = (row >= 0) & (row < ROWS)
    if HAS_VALID:
        active &= tl.load(Valid + query)
    raw = tl.load(Weight + row * DIM + columns, active & (columns < DIM), 0)
    codes = tl.load(
        Scale + row * (DIM // 32) + columns // 32, active & (columns < DIM), 127
    ).to(tl.uint32)
    # E8M0 code0 is 2^-127, not zero; code255 represents NaN.
    bits = tl.where(codes == 0, 0x00400000, codes << 23)
    bits = tl.where(codes == 255, 0x7FC00000, bits)
    scale = bits.to(tl.float32, bitcast=True)
    value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    tl.store(
        Output + query * DIM + columns,
        tl.where(active, value, 0).to(tl.bfloat16),
        columns < DIM,
    )


@triton.jit
def engram_gather_quantize_kernel(
    Weight,
    Scale,
    Indices,
    Valid,
    Output,
    Encoded,
    OutputScale,
    Finite,
    ROWS: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    query = tl.program_id(0).to(tl.int64)
    groups = tl.arange(0, 8)
    columns = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    row = tl.load(Indices + query).to(tl.int64)
    active = (row >= 0) & (row < ROWS)
    if HAS_VALID:
        active &= tl.load(Valid + query)
    row = tl.where(active, row, 0)
    raw = tl.load(Weight + row * 256 + columns, active, 0)
    codes = tl.load(Scale + row * 8 + groups, active, 127).to(tl.uint32)
    bits = tl.where(codes == 0, 0x00400000, codes << 23)
    bits = tl.where(codes == 255, 0x7FC00000, bits)
    scale = bits.to(tl.float32, bitcast=True)
    values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale[:, None]
    # Keep both the model's BF16 rounding boundary and the public lookup output.
    rounded = tl.where(active, values, 0).to(tl.bfloat16)
    tl.store(Output + query * 256 + columns, rounded)
    values = rounded.to(tl.float32)
    finite = tl.abs(values) < float("inf")
    values = tl.where(finite, values, 0.0)
    maximum = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4)
    output_scale, _ = _round_power_of_two_scale(maximum, 1.0 / 448.0)
    encoded = tl.minimum(
        tl.maximum(tl.div_rn(values, output_scale[:, None]), -448.0), 448.0
    )
    tl.store(
        Encoded + query * 256 + columns,
        encoded.to(tl.float8e4nv, fp_downcast_rounding="rtne"),
    )
    tl.store(OutputScale + query * 8 + groups, output_scale)
    tl.store(Finite + query * 8 + groups, tl.sum((~finite).to(tl.int32), 1) == 0)
