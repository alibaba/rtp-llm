# SPDX-License-Identifier: Apache-2.0
# Adapted from RTP-LLM _engram_triton.py.
"""Read selected E4M3/group-32 UE8M0 rows from mapped host storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _lookup_host_kernel(
    weight,
    scales,
    indices,
    valid_rows,
    output,
    tokens,
    table_rows,
    ids_stride_t,
    ids_stride_h,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    column = tl.arange(0, BLOCK_D)
    token, head = row // HEADS, row % HEADS
    live = tl.load(valid_rows + token, token < tokens, other=False)
    index = tl.load(
        indices + token * ids_stride_t + head * ids_stride_h, token < tokens, other=-1
    ).to(tl.int64)
    live &= (token < tokens) & (index >= 0) & (index < table_rows)
    index = tl.where(live, index, 0)
    raw = tl.load(
        weight + index[:, None] * DIM + column[None, :],
        live[:, None] & (column[None, :] < DIM),
        other=0,
    )
    values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    encoded = tl.load(
        scales + index[:, None] * (DIM // 32) + column[None, :] // 32,
        live[:, None] & (column[None, :] < DIM),
        other=127,
    ).to(tl.uint32)
    bits = tl.where(encoded == 0, 0x00400000, encoded << 23)
    bits = tl.where(encoded == 255, 0x7FC00000, bits)
    scale = bits.to(tl.uint32).to(tl.float32, bitcast=True)
    result = tl.where(live[:, None], values * scale, 0)
    tl.store(
        output + row[:, None] * DIM + column[None, :],
        result,
        (token < tokens)[:, None] & (column[None, :] < DIM),
    )


def lookup_rows(weight, scales, indices, valid_rows):
    tokens, heads = indices.shape
    dim = weight.shape[1]
    output = torch.empty(
        (tokens, heads, dim), dtype=torch.bfloat16, device=indices.device
    )
    if tokens:
        _lookup_host_kernel[(triton.cdiv(tokens * heads, 16),)](
            weight,
            scales,
            indices,
            valid_rows,
            output,
            tokens,
            weight.shape[0],
            indices.stride(0),
            indices.stride(1),
            HEADS=heads,
            DIM=dim,
            BLOCK_D=triton.next_power_of_2(dim),
            BLOCK_R=16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
