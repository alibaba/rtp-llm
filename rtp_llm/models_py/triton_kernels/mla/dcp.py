"""MLA DCP wire packing and base-2 LSE reduction kernels."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _pack_a2a(
    partial, lse, lengths, packed_o, packed_lse, tokens,
    heads: tl.constexpr, local_heads: tl.constexpr, dim: tl.constexpr, block: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    destination = head // local_heads
    output_row = (destination * tokens + token) * local_heads + head % local_heads
    # Each wire row is dim BF16 values followed by one bit-exact FP32 LSE.
    offset = output_row * (dim + 2)
    nonempty = tl.load(lengths + token) > 0
    logsumexp = tl.load(lse + token * heads + head)
    tl.store(packed_lse + offset // 2 + dim // 2, tl.where(nonempty, logsumexp, -float("inf")))
    d = tl.arange(0, block)
    value = tl.load(partial + (token * heads + head) * dim + d, d < dim, other=0)
    tl.store(packed_o + offset + d, tl.where(nonempty, value, 0.0), d < dim)


@triton.jit
def _combine_a2a(
    received_o, received_lse, output, tokens,
    local_heads: tl.constexpr, dim: tl.constexpr, cp_size: tl.constexpr, block: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    offset = (token * local_heads + head) * (dim + 2)
    rank_stride = tokens * local_heads * (dim + 2)
    maximum = tl.full((), -float("inf"), tl.float32)
    for rank in tl.static_range(cp_size):
        lse = tl.load(received_lse + (rank * rank_stride + offset + dim) // 2)
        maximum = tl.maximum(maximum, lse, propagate_nan=tl.PropagateNan.ALL)
    has_keys = maximum != -float("inf")
    denominator = tl.full((), 0.0, tl.float32)
    value_sum = tl.full((block,), 0.0, tl.float32)
    d = tl.arange(0, block)
    for rank in tl.static_range(cp_size):
        lse = tl.load(received_lse + (rank * rank_stride + offset + dim) // 2)
        weight = tl.where(has_keys, tl.exp2(lse - maximum), 0.0)
        value = tl.load(received_o + rank * rank_stride + offset + d, d < dim, other=0).to(tl.float32)
        value = tl.where(lse == -float("inf"), 0.0, value)
        denominator += weight
        value_sum += value * weight
    result = tl.where(has_keys, value_sum / denominator, 0.0)
    tl.store(output + (head * tokens + token) * dim + d, result, d < dim)
