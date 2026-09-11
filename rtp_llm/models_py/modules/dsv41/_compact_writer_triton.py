# Quantization formulas adapted from DeepSeek-V4.1-Flash inference/kernel.py,
# revision 2bc89ac599031fa673cab993f1df02fc4a98c673.
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton
import triton.language as tl


@triton.jit
def _round_power_of_two_scale(maximum, INVERSE_MAX: tl.constexpr):
    scaled = maximum * INVERSE_MAX
    bits = scaled.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    return scale, exponent.to(tl.uint8)


@triton.jit
def _round_e2m1(values):
    magnitude = tl.abs(values)
    # Alternating strict/non-strict boundaries implement round-to-nearest-even.
    code = (magnitude > 0.25).to(tl.uint8)
    code += (magnitude >= 0.75).to(tl.uint8)
    code += (magnitude > 1.25).to(tl.uint8)
    code += (magnitude >= 1.75).to(tl.uint8)
    code += (magnitude > 2.5).to(tl.uint8)
    code += (magnitude >= 3.5).to(tl.uint8)
    code += (magnitude > 5.0).to(tl.uint8)
    sign = ((values.to(tl.uint32, bitcast=True) >> 28) & 8).to(tl.uint8)
    return code | sign


@triton.jit
def encode_compact_kernel(
    values,
    slot_mapping,
    output,
    status,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    FORMAT: tl.constexpr,
    PAGED: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ENTRIES: tl.constexpr,
    NUM_PAGES: tl.constexpr,
):
    row = tl.program_id(0)
    channels = tl.arange(0, DIM)
    x = tl.load(values + row * DIM + channels).to(tl.float32)
    finite = tl.abs(x) < float("inf")
    nonfinite = tl.sum((~finite).to(tl.int32), axis=0) != 0
    x = tl.where(finite, x, 0.0).reshape((DIM // GROUP, GROUP))
    maximum = tl.max(tl.abs(x), axis=1)

    if FORMAT == 0:
        maximum = tl.maximum(maximum, 1.0e-4)
        scale, scale_bytes = _round_power_of_two_scale(maximum, 1.0 / 448.0)
        normalized = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -448.0), 448.0)
        payload = normalized.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(
            tl.uint8, bitcast=True
        )
        payload = payload.reshape((DIM,))
        payload_size: tl.constexpr = DIM
    else:
        if FORMAT == 1:
            maximum = tl.maximum(maximum, 6.0 * (2.0**-9))
            scale_fp8 = tl.div_rn(maximum, 6.0).to(
                tl.float8e4nv, fp_downcast_rounding="rtne"
            )
            scale = scale_fp8.to(tl.float32)
            scale_bytes = scale_fp8.to(tl.uint8, bitcast=True)
        else:
            maximum = tl.maximum(maximum, 6.0 * (2.0**-126))
            scale, scale_bytes = _round_power_of_two_scale(maximum, 1.0 / 6.0)
        normalized = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -6.0), 6.0)
        codes = _round_e2m1(normalized).reshape((DIM // 2, 2))
        even, odd = tl.split(codes)
        payload = even | (odd << 4)
        payload_size: tl.constexpr = DIM // 2

    scale_valid = (scale > 0.0) & (scale < float("inf"))
    bad_scale = tl.sum((~scale_valid).to(tl.int32), axis=0) != 0
    if PAGED:
        slot = tl.load(slot_mapping + row).to(tl.int64)
        page = tl.maximum(slot, 0) // ENTRIES
        offset = tl.maximum(slot, 0) % ENTRIES
        active = slot >= 0
        valid_destination = active & (page > 0) & (page < NUM_PAGES)
        invalid_destination = (slot < -1) | (active & ~valid_destination)
        base = output + page * PAGE_STRIDE + offset * ROW_BYTES
        store = valid_destination & ~nonfinite & ~bad_scale
    else:
        active = True
        invalid_destination = False
        base = output + row.to(tl.int64) * ROW_BYTES
        store = True
        payload = tl.where(nonfinite | bad_scale, 0, payload)
        scale_bytes = tl.where(nonfinite | bad_scale, 0, scale_bytes)

    tl.store(base + tl.arange(0, payload_size), payload, mask=store)
    tl.store(base + payload_size + tl.arange(0, DIM // GROUP), scale_bytes, mask=store)
    error = tl.where(active & nonfinite, 1, 0)
    error |= tl.where(invalid_destination, 2, 0)
    error |= tl.where(active & bad_scale, 4, 0)
    tl.store(status + row, error)
