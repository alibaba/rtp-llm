# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small-batch MoE routing helpers implemented in FlyDSL."""

import functools
from contextlib import contextmanager

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, buffer_ops, gpu
from flydsl.expr import Stream as _FlyStream
from flydsl.expr.typing import T

_route_cf_cache = {}


def _launch_cached(cache, key, launch_fn, args, stream):
    cf = cache.get(key)
    stream_arg = _FlyStream(stream)
    if cf is None:
        cf = flyc.compile(launch_fn, *args, stream_arg)
        cache[key] = cf
        return
    cf(*args, stream_arg)


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@functools.lru_cache(maxsize=32)
def compile_direct_moe_route(*, topk: int, tile_m: int):
    """Compile a direct route metadata kernel for very small MoE batches.

    The kernel emits one padded GEMM tile for every `(token, topk_slot)` route:
    row 0 contains the real packed route id, rows 1..tile_m-1 contain the
    sentinel `(topk << 24) | tokens`. This matches the existing stage1/stage2
    sorted-buffer contract without invoking CK `moe_sorting`.
    """

    @flyc.kernel
    def direct_moe_route_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        route = gpu.block_id("x")
        tid = gpu.thread_id("x")
        route_i32 = arith.index_cast(T.i32, route)
        tid_i32 = arith.index_cast(T.i32, tid)
        tokens_idx = arith.index_cast(T.index, i32_tokens)
        routes_idx = tokens_idx * fx.Index(topk)

        ids_rsrc = buffer_ops.create_buffer_resource(
            topk_ids, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        weights_rsrc = buffer_ops.create_buffer_resource(
            topk_weights, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        sorted_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids,
            max_size=False,
            num_records_bytes=routes_idx * fx.Index(tile_m * 4),
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights,
            max_size=False,
            num_records_bytes=routes_idx * fx.Index(tile_m * 4),
        )
        expert_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids,
            max_size=False,
            num_records_bytes=routes_idx * fx.Index(4),
        )
        num_valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))

        token = route // fx.Index(topk)
        slot = route % fx.Index(topk)
        token_i32 = arith.index_cast(T.i32, token)
        slot_i32 = arith.index_cast(T.i32, slot)
        topk_i32 = arith.constant(topk, type=T.i32)
        shift24 = arith.constant(24, type=T.i32)

        expert = buffer_ops.buffer_load(ids_rsrc, route, vec_width=1, dtype=T.i32)
        route_weight = buffer_ops.buffer_load(weights_rsrc, route, vec_width=1, dtype=T.f32)
        valid_fused = (slot_i32 << shift24) | token_i32
        sentinel = (topk_i32 << shift24) | i32_tokens

        is_first_row = arith.cmpi(arith.CmpIPredicate.eq, tid_i32, arith.constant(0, type=T.i32))
        fused = arith.select(is_first_row, valid_fused, sentinel)
        weight = arith.select(is_first_row, route_weight, arith.constant(0.0, type=T.f32))
        out_row = route * fx.Index(tile_m) + tid
        buffer_ops.buffer_store(fused, sorted_rsrc, out_row)
        buffer_ops.buffer_store(weight, sorted_w_rsrc, out_row)

        first_row_if = scf.IfOp(is_first_row)
        with _if_then(first_row_if):
            buffer_ops.buffer_store(expert, expert_rsrc, route)

        route_zero = arith.cmpi(arith.CmpIPredicate.eq, route_i32, arith.constant(0, type=T.i32))
        first_route = is_first_row & route_zero
        first_route_if = scf.IfOp(first_route)
        with _if_then(first_route_if):
            padded = i32_tokens * topk_i32 * arith.constant(tile_m, type=T.i32)
            buffer_ops.buffer_store(padded, num_valid_rsrc, fx.Index(0))
            buffer_ops.buffer_store(i32_tokens * topk_i32, num_valid_rsrc, fx.Index(1))

    @flyc.jit
    def launch_direct_moe_route(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        routes = fx.Index(i32_tokens) * fx.Index(topk)
        direct_moe_route_kernel(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            i32_tokens,
        ).launch(grid=(routes, 1, 1), block=(tile_m, 1, 1), stream=stream)

    return launch_direct_moe_route


def direct_moe_route(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    topk: int,
    tile_m: int,
    stream,
):
    tokens = int(topk_ids.shape[0])
    routes = tokens * int(topk)
    sorted_ids = torch.empty((routes * tile_m,), dtype=torch.int32, device=topk_ids.device)
    sorted_weights = torch.empty((routes * tile_m,), dtype=torch.float32, device=topk_ids.device)
    sorted_expert_ids = torch.empty((routes,), dtype=torch.int32, device=topk_ids.device)
    num_valid_ids = torch.empty((2,), dtype=torch.int32, device=topk_ids.device)
    launcher = compile_direct_moe_route(topk=int(topk), tile_m=int(tile_m))
    _r_args = (
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        tokens,
    )
    _launch_cached(_route_cf_cache, id(launcher), launcher, _r_args, stream)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


@functools.lru_cache(maxsize=32)
def compile_grouped_moe_route(*, experts: int, topk: int, tile_m: int):
    """Compile a pure FlyDSL expert-grouped route metadata path.

    This emits the same sorted-buffer contract consumed by the existing MoE
    GEMMs, but compacts rows by expert instead of emitting one M tile per
    token-slot route. It is intentionally simple: one workgroup owns one expert
    and serially scans all routes, reserving compact output tiles with one
    global atomic per active expert.
    """

    @flyc.kernel
    def grouped_route_init_kernel(
        route_tile_count: fx.Tensor,
        num_valid_ids: fx.Tensor,
    ):
        counter_rsrc = buffer_ops.create_buffer_resource(
            route_tile_count, max_size=False, num_records_bytes=fx.Index(4)
        )
        num_valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))
        zero = arith.constant(0, type=T.i32)
        buffer_ops.buffer_store(zero, counter_rsrc, fx.Index(0))
        buffer_ops.buffer_store(zero, num_valid_rsrc, fx.Index(0))
        buffer_ops.buffer_store(zero, num_valid_rsrc, fx.Index(1))

    @flyc.kernel
    def grouped_moe_route_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        route_tile_count: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        expert = gpu.block_id("x")
        expert_i32 = arith.index_cast(T.i32, expert)
        tokens_idx = arith.index_cast(T.index, i32_tokens)
        routes_idx = tokens_idx * fx.Index(topk)

        ids_rsrc = buffer_ops.create_buffer_resource(
            topk_ids, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        weights_rsrc = buffer_ops.create_buffer_resource(
            topk_weights, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        sorted_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids,
            max_size=False,
            num_records_bytes=routes_idx * fx.Index(tile_m * 4),
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights,
            max_size=False,
            num_records_bytes=routes_idx * fx.Index(tile_m * 4),
        )
        expert_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids,
            max_size=False,
            num_records_bytes=routes_idx * fx.Index(4),
        )

        zero_i32 = arith.constant(0, type=T.i32)
        one_i32 = arith.constant(1, type=T.i32)
        topk_i32 = arith.constant(topk, type=T.i32)
        tile_m_i32 = arith.constant(tile_m, type=T.i32)
        shift24 = arith.constant(24, type=T.i32)

        # Count routes for this expert.
        for route, state in range(fx.Index(0), routes_idx, fx.Index(1), init=[zero_i32]):
            count = state[0]
            route_expert = buffer_ops.buffer_load(ids_rsrc, route, vec_width=1, dtype=T.i32)
            is_match = arith.cmpi(arith.CmpIPredicate.eq, route_expert, expert_i32)
            inc = arith.select(is_match, one_i32, zero_i32)
            results = yield [count + inc]
        count = results

        has_routes = arith.cmpi(arith.CmpIPredicate.ugt, count, zero_i32)
        count_if = scf.IfOp(has_routes)
        with _if_then(count_if):
            tiles_i32 = (count + tile_m_i32 - one_i32) // tile_m_i32
            counter_base = buffer_ops.extract_base_index(route_tile_count)
            counter_ptr = buffer_ops.create_llvm_ptr(counter_base, address_space=1)
            counter_ptr_v = counter_ptr._value if hasattr(counter_ptr, "_value") else counter_ptr
            base_i32 = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                counter_ptr_v,
                tiles_i32,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result

            base_idx = arith.index_cast(T.index, base_i32)
            tiles_idx = arith.index_cast(T.index, tiles_i32)
            count_idx = arith.index_cast(T.index, count)

            # Per-output-tile expert ids.
            for tile in range(fx.Index(0), tiles_idx, fx.Index(1)):
                tile_idx = arith.index_cast(T.index, tile)
                buffer_ops.buffer_store(expert_i32, expert_rsrc, base_idx + tile_idx)

            # Scatter matching routes into compact expert-contiguous rows.
            for route, state in range(fx.Index(0), routes_idx, fx.Index(1), init=[zero_i32]):
                rank = state[0]
                route_expert = buffer_ops.buffer_load(ids_rsrc, route, vec_width=1, dtype=T.i32)
                is_match = arith.cmpi(arith.CmpIPredicate.eq, route_expert, expert_i32)
                match_if = scf.IfOp(is_match)
                with _if_then(match_if):
                    token = route // fx.Index(topk)
                    slot = route % fx.Index(topk)
                    token_i32 = arith.index_cast(T.i32, token)
                    slot_i32 = arith.index_cast(T.i32, slot)
                    fused = arith.ori(arith.shli(slot_i32, shift24), token_i32)
                    route_weight = buffer_ops.buffer_load(weights_rsrc, route, vec_width=1, dtype=T.f32)
                    rank_idx = arith.index_cast(T.index, rank)
                    out_row = base_idx * fx.Index(tile_m) + rank_idx
                    buffer_ops.buffer_store(fused, sorted_rsrc, out_row)
                    buffer_ops.buffer_store(route_weight, sorted_w_rsrc, out_row)
                inc = arith.select(is_match, one_i32, zero_i32)
                results_scatter = yield [rank + inc]

            # Pad the tail of the final tile with the CK-style sentinel.
            del results_scatter
            sentinel = arith.ori(arith.shli(topk_i32, shift24), arith.unwrap(i32_tokens))
            padded_rows_idx = tiles_idx * fx.Index(tile_m)
            for pad in range(count_idx, padded_rows_idx, fx.Index(1)):
                pad_idx = arith.index_cast(T.index, pad)
                out_row = base_idx * fx.Index(tile_m) + pad_idx
                buffer_ops.buffer_store(sentinel, sorted_rsrc, out_row)
                buffer_ops.buffer_store(arith.constant(0.0, type=T.f32), sorted_w_rsrc, out_row)

    @flyc.kernel
    def grouped_route_finalize_kernel(
        route_tile_count: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        counter_rsrc = buffer_ops.create_buffer_resource(
            route_tile_count, max_size=False, num_records_bytes=fx.Index(4)
        )
        num_valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))
        tiles = buffer_ops.buffer_load(counter_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
        tile_m_i32 = arith.constant(tile_m, type=T.i32)
        topk_i32 = arith.constant(topk, type=T.i32)
        buffer_ops.buffer_store(tiles * tile_m_i32, num_valid_rsrc, fx.Index(0))
        buffer_ops.buffer_store(i32_tokens * topk_i32, num_valid_rsrc, fx.Index(1))

    @flyc.jit
    def launch_grouped_moe_route(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        route_tile_count: fx.Tensor,
        i32_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        grouped_route_init_kernel(route_tile_count, num_valid_ids).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )
        grouped_moe_route_kernel(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            route_tile_count,
            i32_tokens,
        ).launch(grid=(experts, 1, 1), block=(1, 1, 1), stream=stream)
        grouped_route_finalize_kernel(route_tile_count, num_valid_ids, i32_tokens).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    return launch_grouped_moe_route


@functools.lru_cache(maxsize=32)
def compile_atomic_grouped_moe_route(
    *,
    experts: int,
    topk: int,
    tile_m: int,
    tiles_per_expert: int,
    block_threads: int = 256,
):
    """Compile a parallel fixed-slot grouped route path.

    The output uses `experts * tiles_per_expert` metadata blocks. Inactive blocks
    are marked with expert id -1 and skipped by the GEMMs before any weight
    loads. This avoids the serial all-expert scan in `compile_grouped_moe_route`.
    """

    max_blocks = int(experts) * int(tiles_per_expert)
    max_rows = max_blocks * int(tile_m)

    @flyc.kernel
    def atomic_grouped_route_init_kernel(
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        expert_counts: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        linear = bid * fx.Index(block_threads) + tid
        linear_i32 = arith.index_cast(T.i32, linear)

        sorted_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=fx.Index(max_rows * 4)
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights, max_size=False, num_records_bytes=fx.Index(max_rows * 4)
        )
        expert_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids,
            max_size=False,
            num_records_bytes=fx.Index(max_blocks * 4),
        )
        counts_rsrc = buffer_ops.create_buffer_resource(
            expert_counts, max_size=False, num_records_bytes=fx.Index(experts * 4)
        )
        num_valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))

        topk_i32 = arith.constant(topk, type=T.i32)
        shift24 = arith.constant(24, type=T.i32)
        sentinel = arith.ori(arith.shli(topk_i32, shift24), arith.unwrap(i32_tokens))

        row_ok = arith.cmpi(arith.CmpIPredicate.ult, linear_i32, arith.constant(max_rows, type=T.i32))
        row_if = scf.IfOp(row_ok)
        with _if_then(row_if):
            buffer_ops.buffer_store(sentinel, sorted_rsrc, linear)
            buffer_ops.buffer_store(arith.constant(0.0, type=T.f32), sorted_w_rsrc, linear)

        block_ok = arith.cmpi(arith.CmpIPredicate.ult, linear_i32, arith.constant(max_blocks, type=T.i32))
        block_if = scf.IfOp(block_ok)
        with _if_then(block_if):
            buffer_ops.buffer_store(arith.constant(-1, type=T.i32), expert_rsrc, linear)

        expert_ok = arith.cmpi(arith.CmpIPredicate.ult, linear_i32, arith.constant(experts, type=T.i32))
        expert_if = scf.IfOp(expert_ok)
        with _if_then(expert_if):
            buffer_ops.buffer_store(arith.constant(0, type=T.i32), counts_rsrc, linear)

        first = arith.cmpi(arith.CmpIPredicate.eq, linear_i32, arith.constant(0, type=T.i32))
        first_if = scf.IfOp(first)
        with _if_then(first_if):
            buffer_ops.buffer_store(arith.constant(max_rows, type=T.i32), num_valid_rsrc, fx.Index(0))
            buffer_ops.buffer_store(i32_tokens * topk_i32, num_valid_rsrc, fx.Index(1))

    @flyc.kernel
    def atomic_grouped_moe_route_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        expert_counts: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        route = bid * fx.Index(block_threads) + tid
        route_i32 = arith.index_cast(T.i32, route)
        tokens_idx = arith.index_cast(T.index, i32_tokens)
        routes_idx = tokens_idx * fx.Index(topk)
        routes_i32 = arith.index_cast(T.i32, routes_idx)

        ids_rsrc = buffer_ops.create_buffer_resource(
            topk_ids, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        weights_rsrc = buffer_ops.create_buffer_resource(
            topk_weights, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        sorted_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=fx.Index(max_rows * 4)
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights, max_size=False, num_records_bytes=fx.Index(max_rows * 4)
        )
        expert_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids,
            max_size=False,
            num_records_bytes=fx.Index(max_blocks * 4),
        )

        in_range = arith.cmpi(arith.CmpIPredicate.ult, route_i32, routes_i32)
        range_if = scf.IfOp(in_range)
        with _if_then(range_if):
            expert_i32 = buffer_ops.buffer_load(ids_rsrc, route, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(T.index, expert_i32)
            counts_base = buffer_ops.extract_base_index(expert_counts)
            count_ptr = buffer_ops.create_llvm_ptr(counts_base + expert_idx * fx.Index(4), address_space=1)
            count_ptr_v = count_ptr._value if hasattr(count_ptr, "_value") else count_ptr
            rank_i32 = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                count_ptr_v,
                arith.constant(1, type=T.i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            rank_idx = arith.index_cast(T.index, rank_i32)

            block_local = rank_idx // fx.Index(tile_m)
            row_local = rank_idx % fx.Index(tile_m)
            block = expert_idx * fx.Index(tiles_per_expert) + block_local
            out_row = block * fx.Index(tile_m) + row_local

            token = route // fx.Index(topk)
            slot = route % fx.Index(topk)
            token_i32 = arith.index_cast(T.i32, token)
            slot_i32 = arith.index_cast(T.i32, slot)
            shift24 = arith.constant(24, type=T.i32)
            fused = arith.ori(arith.shli(slot_i32, shift24), token_i32)
            route_weight = buffer_ops.buffer_load(weights_rsrc, route, vec_width=1, dtype=T.f32)

            buffer_ops.buffer_store(fused, sorted_rsrc, out_row)
            buffer_ops.buffer_store(route_weight, sorted_w_rsrc, out_row)
            buffer_ops.buffer_store(expert_i32, expert_rsrc, block)

    @flyc.jit
    def launch_atomic_grouped_moe_route(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        expert_counts: fx.Tensor,
        i32_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        init_elems = fx.Index(max(max_rows, max_blocks, experts))
        init_blocks = (init_elems + fx.Index(block_threads - 1)) // fx.Index(block_threads)
        atomic_grouped_route_init_kernel(
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            expert_counts,
            num_valid_ids,
            i32_tokens,
        ).launch(grid=(init_blocks, 1, 1), block=(block_threads, 1, 1), stream=stream)

        routes = fx.Index(i32_tokens) * fx.Index(topk)
        route_blocks = (routes + fx.Index(block_threads - 1)) // fx.Index(block_threads)
        atomic_grouped_moe_route_kernel(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            expert_counts,
            i32_tokens,
        ).launch(grid=(route_blocks, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch_atomic_grouped_moe_route


@functools.lru_cache(maxsize=32)
def compile_compact_atomic_grouped_moe_route(
    *,
    experts: int,
    topk: int,
    tile_m: int,
    tiles_per_expert: int,
    max_blocks: int | None = None,
    block_threads: int = 256,
):
    """Compile a parallel compact grouped route path.

    This uses route-parallel atomics for count/scatter and a small per-expert
    prefix/fill kernel to emit compact GEMM metadata blocks.
    """

    max_blocks = int(max_blocks) if max_blocks is not None else int(experts) * int(tiles_per_expert)
    max_rows = max_blocks * int(tile_m)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def compact_route_init_count_kernel(
        topk_ids: fx.Tensor,
        expert_counts: fx.Tensor,
        expert_bases: fx.Tensor,
        route_tile_count: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        tid = gpu.thread_id("x")
        tid_i32 = arith.index_cast(T.i32, tid)
        routes_idx = arith.index_cast(T.index, i32_tokens) * fx.Index(topk)
        routes_i32 = arith.index_cast(T.i32, routes_idx)

        ids_rsrc = buffer_ops.create_buffer_resource(
            topk_ids, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        counts_rsrc = buffer_ops.create_buffer_resource(
            expert_counts, max_size=False, num_records_bytes=fx.Index(experts * 4)
        )
        bases_rsrc = buffer_ops.create_buffer_resource(
            expert_bases, max_size=False, num_records_bytes=fx.Index(experts * 4)
        )
        tile_count_rsrc = buffer_ops.create_buffer_resource(
            route_tile_count, max_size=False, num_records_bytes=fx.Index(4)
        )
        num_valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))

        expert_ok = arith.cmpi(arith.CmpIPredicate.ult, tid_i32, arith.constant(experts, type=T.i32))
        expert_if = scf.IfOp(expert_ok)
        with _if_then(expert_if):
            buffer_ops.buffer_store(arith.constant(0, type=T.i32), counts_rsrc, tid)
            buffer_ops.buffer_store(arith.constant(-1, type=T.i32), bases_rsrc, tid)

        first = arith.cmpi(arith.CmpIPredicate.eq, tid_i32, arith.constant(0, type=T.i32))
        first_if = scf.IfOp(first)
        with _if_then(first_if):
            zero = arith.constant(0, type=T.i32)
            buffer_ops.buffer_store(zero, tile_count_rsrc, fx.Index(0))
            buffer_ops.buffer_store(zero, num_valid_rsrc, fx.Index(0))
            buffer_ops.buffer_store(
                i32_tokens * arith.constant(topk, type=T.i32),
                num_valid_rsrc,
                fx.Index(1),
            )

        gpu.barrier()

        in_range = arith.cmpi(arith.CmpIPredicate.ult, tid_i32, routes_i32)
        range_if = scf.IfOp(in_range)
        with _if_then(range_if):
            expert_i32 = buffer_ops.buffer_load(ids_rsrc, tid, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(T.index, expert_i32)
            counts_base = buffer_ops.extract_base_index(expert_counts)
            count_ptr = buffer_ops.create_llvm_ptr(counts_base + expert_idx * fx.Index(4), address_space=1)
            count_ptr_v = count_ptr._value if hasattr(count_ptr, "_value") else count_ptr
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                count_ptr_v,
                arith.constant(1, type=T.i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

    @flyc.kernel
    def compact_route_init_kernel(
        expert_counts: fx.Tensor,
        expert_bases: fx.Tensor,
        route_tile_count: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        linear = bid * fx.Index(block_threads) + tid
        linear_i32 = arith.index_cast(T.i32, linear)

        counts_rsrc = buffer_ops.create_buffer_resource(
            expert_counts, max_size=False, num_records_bytes=fx.Index(experts * 4)
        )
        bases_rsrc = buffer_ops.create_buffer_resource(
            expert_bases, max_size=False, num_records_bytes=fx.Index(experts * 4)
        )
        tile_count_rsrc = buffer_ops.create_buffer_resource(
            route_tile_count, max_size=False, num_records_bytes=fx.Index(4)
        )
        num_valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))

        expert_ok = arith.cmpi(arith.CmpIPredicate.ult, linear_i32, arith.constant(experts, type=T.i32))
        expert_if = scf.IfOp(expert_ok)
        with _if_then(expert_if):
            buffer_ops.buffer_store(arith.constant(0, type=T.i32), counts_rsrc, linear)
            buffer_ops.buffer_store(arith.constant(-1, type=T.i32), bases_rsrc, linear)

        first = arith.cmpi(arith.CmpIPredicate.eq, linear_i32, arith.constant(0, type=T.i32))
        first_if = scf.IfOp(first)
        with _if_then(first_if):
            zero = arith.constant(0, type=T.i32)
            buffer_ops.buffer_store(zero, tile_count_rsrc, fx.Index(0))
            buffer_ops.buffer_store(zero, num_valid_rsrc, fx.Index(0))
            buffer_ops.buffer_store(
                i32_tokens * arith.constant(topk, type=T.i32),
                num_valid_rsrc,
                fx.Index(1),
            )

    @flyc.kernel
    def compact_route_count_kernel(
        topk_ids: fx.Tensor,
        expert_counts: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        route = bid * fx.Index(block_threads) + tid
        route_i32 = arith.index_cast(T.i32, route)
        routes_idx = arith.index_cast(T.index, i32_tokens) * fx.Index(topk)
        routes_i32 = arith.index_cast(T.i32, routes_idx)

        ids_rsrc = buffer_ops.create_buffer_resource(
            topk_ids, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        in_range = arith.cmpi(arith.CmpIPredicate.ult, route_i32, routes_i32)
        range_if = scf.IfOp(in_range)
        with _if_then(range_if):
            expert_i32 = buffer_ops.buffer_load(ids_rsrc, route, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(T.index, expert_i32)
            counts_base = buffer_ops.extract_base_index(expert_counts)
            count_ptr = buffer_ops.create_llvm_ptr(counts_base + expert_idx * fx.Index(4), address_space=1)
            count_ptr_v = count_ptr._value if hasattr(count_ptr, "_value") else count_ptr
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                count_ptr_v,
                arith.constant(1, type=T.i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

    @flyc.kernel
    def compact_route_prefix_fill_kernel(
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        expert_counts: fx.Tensor,
        expert_bases: fx.Tensor,
        route_tile_count: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        expert = gpu.block_id("x")
        expert_i32 = arith.index_cast(T.i32, expert)

        counts_rsrc = buffer_ops.create_buffer_resource(
            expert_counts, max_size=False, num_records_bytes=fx.Index(experts * 4)
        )
        bases_rsrc = buffer_ops.create_buffer_resource(
            expert_bases, max_size=False, num_records_bytes=fx.Index(experts * 4)
        )
        sorted_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=fx.Index(max_rows * 4)
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights, max_size=False, num_records_bytes=fx.Index(max_rows * 4)
        )
        expert_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids,
            max_size=False,
            num_records_bytes=fx.Index(max_blocks * 4),
        )

        count = buffer_ops.buffer_load(counts_rsrc, expert, vec_width=1, dtype=T.i32)
        has_routes = arith.cmpi(arith.CmpIPredicate.ugt, count, arith.constant(0, type=T.i32))
        count_if = scf.IfOp(has_routes)
        with _if_then(count_if):
            tile_m_i32 = arith.constant(tile_m, type=T.i32)
            one_i32 = arith.constant(1, type=T.i32)
            tiles_i32 = (count + tile_m_i32 - one_i32) // tile_m_i32
            tile_count_base = buffer_ops.extract_base_index(route_tile_count)
            tile_count_ptr = buffer_ops.create_llvm_ptr(tile_count_base, address_space=1)
            tile_count_ptr_v = tile_count_ptr._value if hasattr(tile_count_ptr, "_value") else tile_count_ptr
            base_i32 = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                tile_count_ptr_v,
                tiles_i32,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            end_rows_i32 = (base_i32 + tiles_i32) * arith.constant(tile_m, type=T.i32)
            num_valid_base = buffer_ops.extract_base_index(num_valid_ids)
            num_valid_ptr = buffer_ops.create_llvm_ptr(num_valid_base, address_space=1)
            num_valid_ptr_v = num_valid_ptr._value if hasattr(num_valid_ptr, "_value") else num_valid_ptr
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.max,
                num_valid_ptr_v,
                end_rows_i32,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )
            buffer_ops.buffer_store(base_i32, bases_rsrc, expert)

            base_idx = arith.index_cast(T.index, base_i32)
            tiles_idx = arith.index_cast(T.index, tiles_i32)
            topk_i32 = arith.constant(topk, type=T.i32)
            shift24 = arith.constant(24, type=T.i32)
            sentinel = arith.ori(arith.shli(topk_i32, shift24), arith.unwrap(i32_tokens))

            for tile in range(fx.Index(0), tiles_idx, fx.Index(1)):
                tile_idx = arith.index_cast(T.index, tile)
                buffer_ops.buffer_store(expert_i32, expert_rsrc, base_idx + tile_idx)

            padded_rows = tiles_idx * fx.Index(tile_m)
            for row in range(fx.Index(0), padded_rows, fx.Index(1)):
                row_idx = arith.index_cast(T.index, row)
                out_row = base_idx * fx.Index(tile_m) + row_idx
                buffer_ops.buffer_store(sentinel, sorted_rsrc, out_row)
                buffer_ops.buffer_store(arith.constant(0.0, type=T.f32), sorted_w_rsrc, out_row)

            buffer_ops.buffer_store(arith.constant(0, type=T.i32), counts_rsrc, expert)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def compact_route_scatter_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        expert_counts: fx.Tensor,
        expert_bases: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        route = bid * fx.Index(block_threads) + tid
        route_i32 = arith.index_cast(T.i32, route)
        routes_idx = arith.index_cast(T.index, i32_tokens) * fx.Index(topk)
        routes_i32 = arith.index_cast(T.i32, routes_idx)

        ids_rsrc = buffer_ops.create_buffer_resource(
            topk_ids, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        weights_rsrc = buffer_ops.create_buffer_resource(
            topk_weights, max_size=False, num_records_bytes=routes_idx * fx.Index(4)
        )
        sorted_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=fx.Index(max_rows * 4)
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights, max_size=False, num_records_bytes=fx.Index(max_rows * 4)
        )
        bases_rsrc = buffer_ops.create_buffer_resource(
            expert_bases, max_size=False, num_records_bytes=fx.Index(experts * 4)
        )

        in_range = arith.cmpi(arith.CmpIPredicate.ult, route_i32, routes_i32)
        range_if = scf.IfOp(in_range)
        with _if_then(range_if):
            expert_i32 = buffer_ops.buffer_load(ids_rsrc, route, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(T.index, expert_i32)
            counts_base = buffer_ops.extract_base_index(expert_counts)
            count_ptr = buffer_ops.create_llvm_ptr(counts_base + expert_idx * fx.Index(4), address_space=1)
            count_ptr_v = count_ptr._value if hasattr(count_ptr, "_value") else count_ptr
            rank_i32 = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                count_ptr_v,
                arith.constant(1, type=T.i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result
            rank_idx = arith.index_cast(T.index, rank_i32)
            base_i32 = buffer_ops.buffer_load(bases_rsrc, expert_idx, vec_width=1, dtype=T.i32)
            base_idx = arith.index_cast(T.index, base_i32)
            out_row = base_idx * fx.Index(tile_m) + rank_idx

            token = route // fx.Index(topk)
            slot = route % fx.Index(topk)
            token_i32 = arith.index_cast(T.i32, token)
            slot_i32 = arith.index_cast(T.i32, slot)
            shift24 = arith.constant(24, type=T.i32)
            fused = arith.ori(arith.shli(slot_i32, shift24), token_i32)
            route_weight = buffer_ops.buffer_load(weights_rsrc, route, vec_width=1, dtype=T.f32)

            buffer_ops.buffer_store(fused, sorted_rsrc, out_row)
            buffer_ops.buffer_store(route_weight, sorted_w_rsrc, out_row)

    @flyc.kernel
    def compact_route_finalize_kernel(
        route_tile_count: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        tile_count_rsrc = buffer_ops.create_buffer_resource(
            route_tile_count, max_size=False, num_records_bytes=fx.Index(4)
        )
        num_valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=False, num_records_bytes=fx.Index(8))
        tiles = buffer_ops.buffer_load(tile_count_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
        buffer_ops.buffer_store(tiles * arith.constant(tile_m, type=T.i32), num_valid_rsrc, fx.Index(0))
        buffer_ops.buffer_store(i32_tokens * arith.constant(topk, type=T.i32), num_valid_rsrc, fx.Index(1))

    @flyc.jit
    def launch_compact_atomic_grouped_moe_route(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        expert_counts: fx.Tensor,
        expert_bases: fx.Tensor,
        route_tile_count: fx.Tensor,
        i32_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        routes = fx.Index(i32_tokens) * fx.Index(topk)
        route_blocks = (routes + fx.Index(block_threads - 1)) // fx.Index(block_threads)
        if const_expr(block_threads >= experts):
            compact_route_init_count_kernel(
                topk_ids,
                expert_counts,
                expert_bases,
                route_tile_count,
                num_valid_ids,
                i32_tokens,
            ).launch(grid=(1, 1, 1), block=(block_threads, 1, 1), stream=stream)
        else:
            expert_blocks = (fx.Index(experts) + fx.Index(block_threads - 1)) // fx.Index(block_threads)
            compact_route_init_kernel(expert_counts, expert_bases, route_tile_count, num_valid_ids, i32_tokens).launch(
                grid=(expert_blocks, 1, 1), block=(block_threads, 1, 1), stream=stream
            )
            compact_route_count_kernel(topk_ids, expert_counts, i32_tokens).launch(
                grid=(route_blocks, 1, 1), block=(block_threads, 1, 1), stream=stream
            )
        compact_route_prefix_fill_kernel(
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            expert_counts,
            expert_bases,
            route_tile_count,
            num_valid_ids,
            i32_tokens,
        ).launch(grid=(experts, 1, 1), block=(1, 1, 1), stream=stream)
        compact_route_scatter_kernel(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            expert_counts,
            expert_bases,
            i32_tokens,
        ).launch(grid=(route_blocks, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch_compact_atomic_grouped_moe_route


def grouped_moe_route(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    experts: int,
    topk: int,
    tile_m: int,
    stream,
):
    tokens = int(topk_ids.shape[0])
    routes = tokens * int(topk)
    tiles_per_expert = (tokens + int(tile_m) - 1) // int(tile_m)
    dense_max_blocks = int(experts) * int(tiles_per_expert)
    if routes <= int(experts):
        tight_max_blocks = routes
    else:
        tight_max_blocks = int(experts) + ((routes - int(experts)) // int(tile_m))
    tight_max_blocks = max(1, int(tight_max_blocks))
    max_blocks = min(dense_max_blocks, tight_max_blocks)
    max_rows = max_blocks * int(tile_m)
    sorted_ids = torch.empty((max_rows,), dtype=torch.int32, device=topk_ids.device)
    sorted_weights = torch.empty((max_rows,), dtype=torch.float32, device=topk_ids.device)
    sorted_expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_valid_ids = torch.empty((2,), dtype=torch.int32, device=topk_ids.device)
    expert_counts = torch.empty((int(experts),), dtype=torch.int32, device=topk_ids.device)
    expert_bases = torch.empty((int(experts),), dtype=torch.int32, device=topk_ids.device)
    route_tile_count = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    block_threads = 512 if routes <= 512 and int(experts) <= 512 else 256
    launcher = compile_compact_atomic_grouped_moe_route(
        experts=int(experts),
        topk=int(topk),
        tile_m=int(tile_m),
        tiles_per_expert=int(tiles_per_expert),
        max_blocks=int(max_blocks),
        block_threads=int(block_threads),
    )
    _gr_args = (
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        expert_counts,
        expert_bases,
        route_tile_count,
        tokens,
    )
    _launch_cached(_route_cf_cache, id(launcher), launcher, _gr_args, stream)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
