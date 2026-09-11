"""GPU-local gather from registered shared-host Engram FP8/UE8M0 tables."""

import triton
import triton.language as tl


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
