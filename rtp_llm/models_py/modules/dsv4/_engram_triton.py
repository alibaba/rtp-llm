# SPDX-License-Identifier: Apache-2.0
"""Graph-safe Engram hashing and reads from CUDA-registered host tables."""

import torch
import triton
import triton.language as tl


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


# Reduce prefetch CTA register pressure on concurrent prefill kernels.
# The synchronous/decode launch contract remains BLOCK_R=16, num_warps=4.
PREFETCH_LOOKUP_CONFIG = (4, 4)
PREFETCH_LOOKUP_CARVEOUT = 100
_PREFETCH_LOOKUP_PREPARED = set()


def _prepare_prefetch_kernel(kernel):
    """Startup only: avoid carveout switches with large-shared-memory kernels."""
    if kernel in _PREFETCH_LOOKUP_PREPARED:
        return False
    from cuda.bindings import driver

    function = driver.CUfunction(kernel.function)
    attribute = (
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    )
    (status,) = driver.cuFuncSetAttribute(function, attribute, PREFETCH_LOOKUP_CARVEOUT)
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Engram prefetch carveout setup failed: {status}")
    status, value = driver.cuFuncGetAttribute(attribute, function)
    if status != driver.CUresult.CUDA_SUCCESS or value != PREFETCH_LOOKUP_CARVEOUT:
        raise RuntimeError(
            f"Engram prefetch carveout verification failed: {status}, {value}"
        )
    # Keep the compiled handle alive; a recycled CUDA function must be prepared.
    _PREFETCH_LOOKUP_PREPARED.add(kernel)
    return True


def lookup_host_rows(weight_uva, scales_uva, indices, num_sms):
    return _lookup_host_rows(weight_uva, scales_uva, indices, num_sms, 16, 4)


def lookup_prefetch_rows(weight_uva, scales_uva, indices, num_sms):
    return _lookup_host_rows(
        weight_uva, scales_uva, indices, num_sms, *PREFETCH_LOOKUP_CONFIG
    )


def warmup_lookup_prefetch_rows(weight_uva, scales_uva, indices, num_sms):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Engram prefetch kernel preparation cannot run in capture")
    return _lookup_host_rows(
        weight_uva,
        scales_uva,
        indices,
        num_sms,
        *PREFETCH_LOOKUP_CONFIG,
        prepare=True,
    )


def _lookup_host_rows(
    weight_uva, scales_uva, indices, num_sms, block_rows, num_warps, *, prepare=False
):
    tokens, heads = indices.shape
    dim = weight_uva.shape[1]
    output = torch.empty(
        (tokens, heads, dim), dtype=torch.bfloat16, device=indices.device
    )
    rows = tokens * heads
    if not rows:
        return output
    grid = min(triton.cdiv(rows, block_rows), num_sms)
    kernel = _lookup_host_kernel[(grid,)](
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
        BLOCK_R=block_rows,
        GRID=grid,
        num_warps=num_warps,
    )
    if prepare and _prepare_prefetch_kernel(kernel):
        # Exercise the configured function before readiness is published.
        return _lookup_host_rows(
            weight_uva, scales_uva, indices, num_sms, block_rows, num_warps
        )
    return output
