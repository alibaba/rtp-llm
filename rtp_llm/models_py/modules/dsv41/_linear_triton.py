"""Group32 activation encoding for the V4.1 dense GEMM boundary."""

import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._compact_writer_triton import (
    _round_power_of_two_scale,
)


@triton.jit
def quantize_block32_kernel(values, output, scales, valid, K: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    columns = group * 32 + tl.arange(0, 32)
    x = tl.load(values + row * K + columns).to(tl.float32)
    finite = tl.abs(x) < float("inf")
    x = tl.where(finite, x, 0.0)
    maximum = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-4)
    scale, _ = _round_power_of_two_scale(maximum, 1.0 / 448.0)
    encoded = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
    tl.store(
        output + row * K + columns,
        encoded.to(tl.float8e4nv, fp_downcast_rounding="rtne"),
    )
    offset = row * (K // 32) + group
    tl.store(scales + offset, scale)
    tl.store(valid + offset, tl.sum((~finite).to(tl.int32), axis=0) == 0)
