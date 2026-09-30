# SPDX-License-Identifier: Apache-2.0
# Adapted from RTP-LLM develop/dsv41_ced_release _engram_triton.py.
"""Engram hash arithmetic over explicit current-first token windows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hash_windows_kernel(
    windows,
    token_map,
    multipliers,
    primes,
    offsets,
    output,
    num_tokens,
    pad_id,
    VOCAB: tl.constexpr,
    LAYERS: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = (tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    layer = tl.program_id(1)
    head = tl.arange(0, BLOCK_H)
    valid = token < num_tokens
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(4):
        raw = tl.load(windows + token * 4 + shift, valid, other=-1).to(tl.int64)
        live = valid & (raw >= 0) & (raw < VOCAB)
        safe_raw = tl.where(live, raw, 0)
        mapped = tl.load(token_map + safe_raw, live, other=pad_id)
        blocked |= ~live
        value = tl.where(blocked, pad_id, mapped)
        rolling ^= value * tl.load(multipliers + layer * 4 + shift)
        if shift > 0:
            column = (shift - 1) * HEADS + head
            param = layer * 3 * HEADS + column
            prime = tl.load(primes + param, head < HEADS, other=1)
            offset = tl.load(offsets + param, head < HEADS, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            destination = (token * LAYERS + layer)[:, None] * (3 * HEADS) + column[
                None, :
            ]
            tl.store(output + destination, hashed, valid[:, None] & (head < HEADS))


def hash_windows(windows, hasher):
    layout = hasher.layout
    output = torch.empty(
        (windows.shape[0], len(layout.layer_ids), layout.n_hash_cols),
        dtype=torch.int64,
        device=windows.device,
    )
    if windows.shape[0]:
        _hash_windows_kernel[
            (triton.cdiv(windows.shape[0], 32), len(layout.layer_ids))
        ](
            windows,
            hasher.token_map,
            hasher.multipliers,
            hasher.primes,
            hasher.offsets,
            output,
            windows.shape[0],
            hasher.pad_id,
            VOCAB=hasher.token_map.numel(),
            LAYERS=len(layout.layer_ids),
            HEADS=layout.n_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(layout.n_heads),
            num_warps=4,
        )
    return output
