# SPDX-License-Identifier: Apache-2.0
"""Native FP8 lookup slices into a complete request's fixed Q/S buffers.

Each launch writes [start_token,start_token+count). The nonempty final slice
also initializes all scale padding to E8M0=127; count0 writes nothing. IDs and
outputs always retain the complete positive-M request layout. Valid FP8/scale
bytes, including scale0/255, are preserved without requantization.

The caller owns scheduling and lifetime: keep IDs immutable until all slices
finish, join every producer before consuming Q/S, and retain the outputs and
registered table/shard owner through all graph replays. Concurrent requests
must have separate outputs. Start/count are runtime kernel arguments but their
values are fixed in a captured graph. This module enables no scheduling policy
and does not modify the whole-lookup implementation.
"""

import torch
import triton
import triton.language as tl

from . import _engram_quantized_triton as direct

NUM_WARPS = 1


@triton.jit(
    do_not_specialize=[
        "tokens",
        "table_rows",
        "stride_t",
        "stride_h",
        "scale_stride",
        "shard_rows",
        "start_token",
        "count",
    ]
)
def _lookup_quantized_slice_kernel(
    weight,
    gpu_scales,
    indices,
    values,
    packed_scales,
    tokens,
    table_rows,
    stride_t,
    stride_h,
    scale_stride,
    scale_ptrs,
    shard_rows,
    start_token,
    count,
    GRID: tl.constexpr,
    SHARDED: tl.constexpr,
):
    weight32 = weight.to(tl.pointer_type(tl.uint32))
    values32 = values.to(tl.pointer_type(tl.uint32))
    words = tl.arange(0, 64)
    start64 = tl.cast(start_token, tl.int64)
    for row in tl.range(tl.program_id(0), count * 24, GRID):
        row64 = tl.cast(row, tl.int64)
        token, head = start64 + row64 // 24, row64 % 24
        index = tl.load(indices + token * stride_t + head * stride_h).to(tl.int64)
        valid = (index >= 0) & (index < table_rows)
        index = tl.where(valid, index, 0)
        raw = tl.load(weight32 + index * 64 + words, valid, other=0)
        global_row = token * 24 + head
        tl.store(values32 + global_row * 64 + words, raw)

    # Relative tiles keep an unaligned slice from writing its neighbor. Only
    # the nonempty final slice extends its stores through the TMA padding.
    final = (count > 0) & (start_token + count == tokens)
    scale_count = tl.where(final, scale_stride - start_token, count)
    scales64 = gpu_scales.to(tl.pointer_type(tl.uint64))
    lanes = tl.arange(0, 32)
    tasks = 24 * tl.cdiv(scale_count, 32)
    for task in tl.range(tl.program_id(0), tasks, GRID):
        task64 = tl.cast(task, tl.int64)
        head = task64 % 24
        relative_token = (task64 // 24) * 32 + lanes
        token = start64 + relative_token
        active = relative_token < count
        index = tl.load(
            indices + token * stride_t + head * stride_h, active, other=-1
        ).to(tl.int64)
        valid = active & (index >= 0) & (index < table_rows)
        index = tl.where(valid, index, 0)
        if SHARDED:
            owner = index // shard_rows
            local_row = index - owner * shard_rows
            address = tl.load(scale_ptrs + owner, valid, other=0).to(tl.uint64)
            source = (address + local_row.to(tl.uint64) * 8).to(
                tl.pointer_type(tl.uint64)
            )
        else:
            source = scales64 + index
        bits = tl.load(source, valid, other=0x7F7F7F7F7F7F7F7F)
        low = bits.to(tl.uint32).to(tl.int32, bitcast=True)
        high = (bits >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
        target = head * 2 * scale_stride + token
        store = relative_token < scale_count
        tl.store(packed_scales + target, low, store)
        tl.store(packed_scales + target + scale_stride, high, store)


def _validate_range(tokens, start_token, count, grid):
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("Native lookup slices require a positive request size")
    if type(grid) is not int or grid <= 0:
        raise ValueError("Slice grid must be a positive integer")
    if (
        type(start_token) is not int
        or type(count) is not int
        or start_token < 0
        or count < 0
        or start_token > tokens
        or count > tokens - start_token
    ):
        raise ValueError("Slice must satisfy0 <= start <= start+count <= M")


def allocate_outputs(ids):
    """Allocate the original full-request Q/S layout; positive M only."""
    return direct.allocate_outputs(ids)


def _inputs(weight, scales, ids, outputs, start_token, count, grid):
    direct._check_indices(ids)
    _validate_range(ids.shape[0], start_token, count, grid)
    if not direct.is_supported(weight, scales, ids, grid):
        raise ValueError("Native lookup slices require supported SM100/SM103 inputs")
    outputs = direct._as_outputs(outputs)
    local, pointers, shard_rows, group_size = direct._scale_arguments(
        scales, ids, weight.shape[0]
    )
    direct._check_outputs(ids, outputs, (weight, local, pointers))
    return outputs, local, pointers, shard_rows, group_size


def launch(weight, scales, ids, outputs, start_token, count, grid=80):
    """Launch into existing full Q/S buffers; return the compiled kernel handle."""
    out, local, pointers, shard_rows, group_size = _inputs(
        weight, scales, ids, outputs, start_token, count, grid
    )
    with torch.cuda.device(ids.device):
        kernel = _lookup_quantized_slice_kernel[(grid,)](
            weight,
            local,
            ids,
            out.values,
            out.scales,
            ids.shape[0],
            weight.shape[0],
            ids.stride(0),
            ids.stride(1),
            out.scales.stride(1),
            pointers,
            shard_rows,
            start_token,
            count,
            GRID=grid,
            SHARDED=group_size > 1,
            num_warps=NUM_WARPS,
        )
        if not isinstance(scales, torch.Tensor):
            scales.record_stream(torch.cuda.current_stream(ids.device))
    return kernel


def prepare(weight, scales, ids, outputs, start_token, count, grid=80):
    """Startup-only compilation/carveout preparation and actual resource receipt."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Lookup slice preparation must run outside capture")
    out, local, pointers, shard_rows, group_size = _inputs(
        weight, scales, ids, outputs, start_token, count, grid
    )
    with torch.cuda.device(ids.device):
        kernel = launch(weight, scales, ids, out, start_token, count, grid)
        carveout = direct._prepare_kernel_cached(kernel)
        launch(weight, scales, ids, out, start_token, count, grid)
        torch.cuda.current_stream(ids.device).synchronize()
    final = count > 0 and start_token + count == ids.shape[0]
    scale_count = out.scales.stride(1) - start_token if final else count
    return {
        "kernel_name": kernel.name,
        "grid": [grid, 1, 1],
        "num_warps": NUM_WARPS,
        "threads_per_cta": 32,
        "block_rows": 1,
        "n_regs": kernel.n_regs,
        "shared_bytes": kernel.metadata.shared,
        "n_spills": getattr(kernel, "n_spills", None),
        "preferred_shared_memory_carveout": carveout,
        "tokens": ids.shape[0],
        "start_token": start_token,
        "count": count,
        "token_end": start_token + count,
        "owns_padding": final,
        "padding_rows_written": out.scales.stride(1) - ids.shape[0] if final else 0,
        "scale_rows_written": scale_count,
        "values_shape": list(out.values.shape),
        "scales_shape": list(out.scales.shape),
        "scales_stride": list(out.scales.stride()),
        "scale_group_size": group_size,
        "scale_rows_per_shard": shard_rows,
        "local_scale_bytes": local.numel() * local.element_size(),
        "scale_pointer_table_bytes": (
            pointers.numel() * pointers.element_size()
            if not isinstance(scales, torch.Tensor)
            else 0
        ),
    }
