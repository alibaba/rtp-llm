"""Raw-byte staging and metadata for the fixed V4.1 FlashMLA reader."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice



@triton.jit
def finalize_native_kernel(
    native_output,
    native_lse,
    sinks,
    query_valid,
    have_kv,
    row_status,
    output,
    lse,
    status,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    head = tl.program_id(0) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    valid = head < ROWS * HEADS
    row = head // HEADS
    active = tl.load(query_valid + row, valid, other=0)
    present = tl.load(have_kv + row, valid, other=0)
    dims = tl.arange(0, 512)
    values = tl.load(
        native_output + head[:, None] * 512 + dims[None, :],
        valid[:, None] & active[:, None],
        other=0,
    )
    tl.store(output + head[:, None] * 512 + dims[None, :], values, valid[:, None])
    attention = tl.load(native_lse + head, valid & present, other=-float("inf"))
    sink = tl.load(sinks + head % HEADS, valid, other=-float("inf"))
    # logaddexp must retain equal infinities and suppress upstream no-KV +inf.
    combined = tl.where(
        attention == sink,
        attention + 0.6931471805599453,
        tl.maximum(attention, sink)
        + libdevice.log1p(libdevice.exp(-tl.abs(attention - sink))),
    )
    tl.store(lse + head, tl.where(active, combined, -float("inf")), valid)
    error = tl.load(row_status + row, valid, other=0)
    tl.store(status + head, error, valid)
