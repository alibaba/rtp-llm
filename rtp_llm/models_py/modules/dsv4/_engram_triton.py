# SPDX-License-Identifier: Apache-2.0
"""Graph-safe Engram hashing and reads from CUDA-registered host tables."""

import os

import torch
import triton
import triton.language as tl


def is_supported(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and os.environ.get("DSV41_ENGRAM_UVA", "1") != "0"


@triton.jit
def _hash_windows_kernel(
    windows,
    dead_mask,
    token_map,
    multipliers,
    primes,
    offsets,
    output,
    num_tokens,
    window_stride,
    dead_stride,
    pad_id,
    VOCAB: tl.constexpr,
    LAYERS: tl.constexpr,
    NGRAM: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_DEAD_MASK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    head = tl.arange(0, BLOCK_H)
    valid = token < num_tokens
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(NGRAM):
        raw = tl.load(windows + token * window_stride + shift, valid, other=-1)
        token_valid = valid & (raw >= 0) & (raw < VOCAB)
        mapped = tl.load(token_map + raw, token_valid, other=pad_id).to(tl.int64)
        dead = tl.full((BLOCK_T,), False, tl.int1)
        if HAS_DEAD_MASK:
            dead = tl.load(dead_mask + token * dead_stride + shift, valid, other=False)
        blocked |= ~token_valid | dead
        value = tl.where(blocked, pad_id, mapped)
        multiplier = tl.load(multipliers + layer * NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            column = (shift - 1) * HEADS + head
            param = layer * (NGRAM - 1) * HEADS + column
            prime = tl.load(primes + param, head < HEADS, other=1)
            offset = tl.load(offsets + param, head < HEADS, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out = (token.to(tl.int64) * LAYERS + layer)[:, None]
            out = out * ((NGRAM - 1) * HEADS) + column[None, :]
            tl.store(output + out, hashed, valid[:, None] & (head < HEADS))


def hash_token_windows(windows, dead_mask, buffers, layout, pad_id):
    if windows.ndim != 2 or windows.shape[1] != layout.max_ngram_size:
        raise ValueError("Engram GPU token windows have the wrong width")
    if windows.stride(1) != 1:
        raise ValueError("Engram GPU token windows must have contiguous columns")
    if dead_mask is not None and (
        dead_mask.shape != windows.shape or dead_mask.stride(1) != 1
    ):
        raise ValueError("Engram GPU dead mask must match token windows")
    output = torch.empty(
        (windows.shape[0], len(layout.layer_ids), layout.n_hash_cols),
        dtype=torch.int64,
        device=windows.device,
    )
    if windows.shape[0] == 0:
        return output
    token_map, multipliers, primes, offsets = buffers
    _hash_windows_kernel[(triton.cdiv(windows.shape[0], 32), len(layout.layer_ids))](
        windows,
        dead_mask if dead_mask is not None else windows,
        token_map,
        multipliers,
        primes,
        offsets,
        output,
        windows.shape[0],
        windows.stride(0),
        dead_mask.stride(0) if dead_mask is not None else 0,
        pad_id,
        VOCAB=token_map.numel(),
        LAYERS=len(layout.layer_ids),
        NGRAM=layout.max_ngram_size,
        HEADS=layout.n_heads,
        BLOCK_T=32,
        BLOCK_H=triton.next_power_of_2(layout.n_heads),
        HAS_DEAD_MASK=dead_mask is not None,
        num_warps=4,
    )
    return output


@triton.jit
def _lookup_host_kernel(
    weight,
    scales,
    indices,
    output,
    rows,
    table_rows,
    ids_stride_t,
    ids_stride_h,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID: tl.constexpr,
):
    columns = tl.arange(0, DIM)
    for base in tl.range(tl.program_id(0) * BLOCK_R, rows, GRID * BLOCK_R):
        row = base + tl.arange(0, BLOCK_R)
        token = row // HEADS
        head = row % HEADS
        index = tl.load(
            indices + token.to(tl.int64) * ids_stride_t + head * ids_stride_h,
            row < rows,
            other=-1,
        ).to(tl.int64)
        valid = (row < rows) & (index >= 0) & (index < table_rows)
        index = tl.where(valid, index, 0)
        raw = tl.load(
            weight + index[:, None] * DIM + columns[None, :],
            valid[:, None],
            other=0,
        )
        value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        exponent = tl.load(
            scales
            + index[:, None] * (DIM // QUANT_BLOCK)
            + columns[None, :] // QUANT_BLOCK,
            valid[:, None],
            other=0,
        )
        scale = (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(
            output + row.to(tl.int64)[:, None] * DIM + columns[None, :],
            (value * scale).to(tl.bfloat16),
            (row < rows)[:, None],
        )


def lookup_host_rows(weight_uva, scales_uva, indices, num_sms):
    if indices.ndim != 2:
        raise ValueError("Engram GPU hashes must be [tokens, heads]")
    tokens, heads = indices.shape
    dim = weight_uva.shape[1]
    output = torch.empty(
        (tokens, heads, dim), dtype=torch.bfloat16, device=indices.device
    )
    rows = tokens * heads
    if not rows:
        return output
    grid = min(triton.cdiv(rows, 16), num_sms)
    _lookup_host_kernel[(grid,)](
        weight_uva,
        scales_uva,
        indices,
        output,
        rows,
        weight_uva.shape[0],
        indices.stride(0),
        indices.stride(1),
        HEADS=heads,
        DIM=dim,
        QUANT_BLOCK=32,
        BLOCK_R=16,
        GRID=grid,
        num_warps=4,
    )
    return output
