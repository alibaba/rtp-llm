# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable epilogue helpers for MFMA 16x16-based kernels.

This module provides:

- `mfma_epilog(...)`
  A single entrypoint that dispatches to either the default row-epilogue or the
  LDS CShuffle epilogue based on input parameters.

- `default_epilog(...)` (implementation helper)
  A lightweight row-iterator for the common MFMA accumulator-to-output mapping
  (mi in [0,m_repeat), ii in [0,4), row = bx_m + mi*16 + lane_div_16*4 + ii).
  The caller supplies `body_row(...)` that performs the per-row epilogue work
  (e.g. loads scales once, loops over ni, stores).

- `c_shuffle_epilog(...)` (implementation helper)
  A LDS CShuffle epilogue skeleton:
    1) call `write_row_to_lds(...)` for each MFMA output row to populate `lds_out`
       in row-major [tile_m, tile_n] order unless an optional LDS index mapper
       is provided
    2) barrier
    3) remap threads into (MLane, NLane) = (8,32) and read half2 from LDS,
       then call `store_pair(...)` to emit the final global store/atomic.

  When ``lds_out_split`` is provided, the epilogue runs in split-LDS mode:
  waves are partitioned into two groups (group A uses ``lds_out``, group B
  uses ``lds_out_split``), each handling half of the N dimension.

These helpers are intentionally *dialect-agnostic*: callers pass the dialect
modules (`arith`, `vector`, `gpu`) and the `range_constexpr` iterator.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr.typing import T


@contextmanager
def _if_then(if_op, scf=None):
    scf_mod = scf if scf is not None else _scf
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf_mod.YieldOp):
                scf_mod.YieldOp([])


def default_epilog(
    *,
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    body_row: Callable,
):
    """Iterate the standard MFMA 16x16 row mapping and call `body_row(...)`.

    The mapping matches the common MFMA fragment layout used across kernels in this repo.

    Args:
      arith: flydsl arith ext module.
      range_constexpr: compile-time unrolled range helper.
      m_repeat: tile_m // 16 (python int).
      lane_div_16: index Value (0..3).
      bx_m: base row (index Value). For MoE, this is the base sorted-row for the tile.
      body_row: callback invoked as:
        body_row(mi=<int>, ii=<int>, row_in_tile=<index>, row=<index>)
    """
    bx_m_v = bx_m
    lane_div_16_mul4 = lane_div_16 * 4
    ii_idx_list = [fx.Index(ii) for ii in range(4)]

    for mi in range_constexpr(m_repeat):
        mi_base = arith.constant(mi * 16, index=True)
        for ii in range_constexpr(4):
            row_off = lane_div_16_mul4 + ii_idx_list[ii]
            row_in_tile = mi_base + row_off
            row = bx_m_v + row_in_tile
            body_row(mi=mi, ii=ii, row_in_tile=row_in_tile, row=row)


def c_shuffle_epilog(
    *,
    arith,
    vector,
    gpu,
    scf=None,
    range_constexpr,
    # Tile params
    tile_m: int,
    tile_n: int,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    m_repeat: int,
    num_acc_n: int,
    # Thread mapping inputs
    tx,
    lane_div_16,
    lane_mod_16,
    bx_m,
    by_n,
    n_tile_base,
    # LDS buffer (f16 view, row-major [tile_m, tile_n] flattened)
    lds_out,
    # Element type for LDS loads (defaults to f16). Pass bf16 to support bf16 epilogues.
    frag_elem_type: ir.Type | None = None,
    # Callbacks
    write_row_to_lds: Callable,
    precompute_row: Callable | None = None,
    store_pair: Callable,
    lds_index_mapper: Callable | None = None,
    # When LDS overflows, split lds_out across two buffers by wave-group.
    # Pass the second buffer here; first buffer is `lds_out`.
    lds_out_split=None,
    # Row offset in lds_out for 8-wave mode (MLIR index value).
    # Shifts both write and read LDS indices by lds_row_offset * tile_n elements.
    lds_row_offset=None,
):
    """LDS CShuffle epilogue skeleton.

    Call pattern:
      - `write_row_to_lds(...)` is called once per MFMA row produced by this thread.
        It is responsible for writing all ni columns for that row into `lds_out`.
      - `store_pair(...)` is called for each (row_local, col_pair0) half2 after shuffle.

    `store_pair` can implement either global stores or atomics.
    """
    if int(block_size) <= 0 or (int(block_size) % int(cshuffle_nlane)) != 0:
        raise ValueError(f"block_size ({block_size}) must be divisible by cshuffle_nlane ({cshuffle_nlane})")
    cshuffle_mlane = int(block_size) // int(cshuffle_nlane)
    if (int(tile_m) % cshuffle_mlane) != 0:
        raise ValueError(f"tile_m must be divisible by CShuffleMLane ({cshuffle_mlane}), got tile_m={tile_m}")
    if int(e_vec) <= 0:
        raise ValueError(f"e_vec must be positive, got {e_vec}")
    if (int(tile_n) % (int(cshuffle_nlane) * int(e_vec))) != 0:
        raise ValueError(
            f"tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_vec}, got tile_n={tile_n}"
        )

    # ===================== Split-LDS mode (early return) =====================
    # When lds_out_split is provided, waves are divided into two groups:
    #   Group A (waves 0..N/2-1) uses lds_out,  columns [0, tile_n/2)
    #   Group B (waves N/2..N-1) uses lds_out_split, columns [tile_n/2, tile_n)
    # Each group writes/reads independently; same barriers synchronise all waves.
    if lds_out_split is not None:
        if scf is None:
            raise ValueError("scf module is required for split-LDS cshuffle")

        _half_n = int(tile_n) // 2
        _half_threads = int(block_size) // 2
        EVec = int(e_vec)

        CShuffleNLane_s = min(int(cshuffle_nlane), _half_n // EVec)
        if _half_threads % CShuffleNLane_s != 0:
            raise ValueError(f"half_threads={_half_threads} not divisible by CShuffleNLane_split={CShuffleNLane_s}")
        CShuffleMLane_s = _half_threads // CShuffleNLane_s
        if int(tile_m) % CShuffleMLane_s != 0:
            raise ValueError(f"tile_m={tile_m} not divisible by CShuffleMLane_split={CShuffleMLane_s}")
        m_reps_s = int(tile_m) // CShuffleMLane_s
        n_reps_s = _half_n // (CShuffleNLane_s * EVec)

        _half_n_idx = arith.constant(_half_n, index=True)
        _half_thr_idx = arith.constant(_half_threads, index=True)
        _zero_idx = arith.constant(0, index=True)

        _is_group_b = arith.cmpi(CmpIPredicate.uge, tx, _half_thr_idx)

        # -- write phase (all waves, each to its group's LDS buffer) --
        n_tile_base_v = n_tile_base
        col_base_local_a = n_tile_base_v + lane_mod_16
        col_base_local_b = col_base_local_a - _half_n_idx

        def _write_row_split(mi: int, ii: int, row_in_tile, row):
            row_base_lds = row_in_tile * _half_n_idx
            _if_g = scf.IfOp(_is_group_b)
            with ir.InsertionPoint(_if_g.then_block):
                write_row_to_lds(
                    mi=mi,
                    ii=ii,
                    row_in_tile=row_in_tile,
                    row=row,
                    row_base_lds=row_base_lds,
                    col_base_local=col_base_local_b,
                    num_acc_n=num_acc_n,
                    lds_out=lds_out_split,
                )
                scf.YieldOp([])
            with ir.InsertionPoint(_if_g.else_block):
                write_row_to_lds(
                    mi=mi,
                    ii=ii,
                    row_in_tile=row_in_tile,
                    row=row,
                    row_base_lds=row_base_lds,
                    col_base_local=col_base_local_a,
                    num_acc_n=num_acc_n,
                    lds_out=lds_out,
                )
                scf.YieldOp([])

        gpu.barrier()
        default_epilog(
            arith=arith,
            range_constexpr=range_constexpr,
            m_repeat=m_repeat,
            lane_div_16=lane_div_16,
            bx_m=bx_m,
            body_row=_write_row_split,
        )
        gpu.barrier()

        # -- read phase (each group reads from its own LDS buffer) --
        tx_local = tx - arith.select(_is_group_b, _half_thr_idx, _zero_idx)
        c_nlane_s = arith.constant(CShuffleNLane_s, index=True)
        m_lane_s = tx_local / c_nlane_s
        n_lane_s = tx_local % c_nlane_s
        c_evec = arith.constant(EVec, index=True)

        if frag_elem_type is None:
            frag_elem_type = T.f16
        vec_frag = T.vec(EVec, frag_elem_type)
        bx_m_v = bx_m
        by_n_v = by_n

        _precomputed_rows_s = []
        for mr in range_constexpr(m_reps_s):
            row_base_m = arith.constant(mr * CShuffleMLane_s, index=True)
            row_local = row_base_m + m_lane_s
            row = bx_m_v + row_local
            row_ctx_raw = precompute_row(row_local=row_local, row=row) if precompute_row is not None else None
            row_ctx = row_ctx_raw
            row_pred = None
            if scf is not None and row_ctx_raw is not None and isinstance(row_ctx_raw, tuple) and len(row_ctx_raw) == 2:
                row_ctx, row_pred = row_ctx_raw
            _precomputed_rows_s.append((row_local, row, row_ctx, row_pred))

        for mr in range_constexpr(m_reps_s):
            row_local, row, row_ctx, row_pred = _precomputed_rows_s[mr]

            def _do_store_row_split():
                row_base_lds = row_local * _half_n_idx
                for nr in range_constexpr(n_reps_s):
                    col_base_nr = arith.constant(nr * (CShuffleNLane_s * EVec), index=True)
                    col_pair0_local = col_base_nr + (n_lane_s * c_evec)
                    lds_idx = row_base_lds + col_pair0_local

                    _if_ld = scf.IfOp(_is_group_b, [vec_frag])
                    with ir.InsertionPoint(_if_ld.then_block):
                        fb = vector.load_op(vec_frag, lds_out_split, [lds_idx])
                        scf.YieldOp([fb])
                    with ir.InsertionPoint(_if_ld.else_block):
                        fa = vector.load_op(vec_frag, lds_out, [lds_idx])
                        scf.YieldOp([fa])
                    frag = _if_ld.results[0]

                    col_pair0 = col_pair0_local + arith.select(_is_group_b, _half_n_idx, _zero_idx)
                    store_pair(
                        row_local=row_local,
                        row=row,
                        row_ctx=row_ctx,
                        col_pair0=col_pair0,
                        col_g0=by_n_v + col_pair0,
                        frag=frag,
                    )

            if row_pred is not None:
                _if_row = scf.IfOp(row_pred)
                with _if_then(_if_row, scf):
                    _do_store_row_split()
            else:
                _do_store_row_split()

        return  # split path complete

    # ===================== Standard (non-split) path below =====================

    # ---------------- Step 1: write C tile to LDS (row-major, fp16) ----------------
    tile_n_idx = arith.constant(int(tile_n), index=True)
    n_tile_base_v = n_tile_base
    col_base_local = n_tile_base_v + lane_mod_16  # index within [0,tile_n)

    _lds_row_base_offset = lds_row_offset * tile_n_idx if lds_row_offset is not None else None

    def _write_row(mi: int, ii: int, row_in_tile, row):
        row_base_lds = row_in_tile * tile_n_idx
        if _lds_row_base_offset is not None:
            row_base_lds = row_base_lds + _lds_row_base_offset
        write_row_to_lds(
            mi=mi,
            ii=ii,
            row_in_tile=row_in_tile,
            row=row,
            row_base_lds=row_base_lds,
            col_base_local=col_base_local,
            num_acc_n=num_acc_n,
            lds_out=lds_out,
        )

    # Ensure all LDS reads finished before the lds write.
    gpu.barrier()
    default_epilog(
        arith=arith,
        range_constexpr=range_constexpr,
        m_repeat=m_repeat,
        lane_div_16=lane_div_16,
        bx_m=bx_m,
        body_row=_write_row,
    )

    # Ensure all LDS writes are visible before the shuffle-read.
    gpu.barrier()

    # ---------------- Step 2: shuffle mapping + half2 store/atomic ----------------
    CShuffleNLane = int(cshuffle_nlane)
    CShuffleMLane = int(cshuffle_mlane)
    EVec = int(e_vec)

    m_reps_shuffle = int(tile_m) // CShuffleMLane
    n_reps_shuffle = int(tile_n) // (CShuffleNLane * EVec)

    c_nlane = fx.Index(CShuffleNLane)
    m_lane = tx // c_nlane
    n_lane = tx % c_nlane
    c_evec = fx.Index(EVec)

    if frag_elem_type is None:
        frag_elem_type = T.f16
    vec_frag = T.vec(EVec, frag_elem_type)
    bx_m_v = bx_m
    by_n_v = by_n

    # Batch-precompute all row contexts (sorted_idx loads) before the store loop.
    # This issues all buffer_load instructions upfront so the compiler can pipeline
    # them instead of serializing each load with s_waitcnt vmcnt(0).
    _precomputed_rows = []
    for mr in range_constexpr(m_reps_shuffle):
        row_base_m = arith.constant(mr * CShuffleMLane, index=True)
        row_local = row_base_m + m_lane
        row = bx_m_v + row_local

        row_ctx_raw = precompute_row(row_local=row_local, row=row) if precompute_row is not None else None

        # Optional row-level predicate: if `precompute_row` returns `(ctx, pred_i1)` and `scf`
        # is provided, we can skip the entire N-loop for invalid rows (cheaper than per-store checks).
        row_ctx = row_ctx_raw
        row_pred = None
        if scf is not None and row_ctx_raw is not None and isinstance(row_ctx_raw, tuple) and len(row_ctx_raw) == 2:
            row_ctx, row_pred = row_ctx_raw

        _precomputed_rows.append((row_local, row, row_ctx, row_pred))

    # Now perform LDS reads and stores using the pre-fetched row contexts.
    for mr in range_constexpr(m_reps_shuffle):
        row_local, row, row_ctx, row_pred = _precomputed_rows[mr]

        def _do_store_row():
            row_base_lds = row_local * tile_n_idx
            if _lds_row_base_offset is not None:
                row_base_lds = row_base_lds + _lds_row_base_offset
            for nr in range_constexpr(n_reps_shuffle):
                col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                col_pair0 = col_base_nr + (n_lane * c_evec)  # even col within tile

                lds_idx_pair = (
                    lds_index_mapper(
                        row_local=row_local,
                        row_base_lds=row_base_lds,
                        col_local=col_pair0,
                    )
                    if lds_index_mapper is not None
                    else row_base_lds + col_pair0
                )
                frag = vector.load_op(vec_frag, lds_out, [lds_idx_pair])

                store_pair(
                    row_local=row_local,
                    row=row,
                    row_ctx=row_ctx,
                    col_pair0=col_pair0,
                    col_g0=by_n_v + col_pair0,
                    frag=frag,
                )

        if row_pred is not None:
            _if_row = scf.IfOp(row_pred)
            with _if_then(_if_row, scf):
                _do_store_row()
        else:
            _do_store_row()


def mfma_epilog(
    *,
    use_cshuffle: bool,
    # Common (always required)
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    # Default epilog (required when use_cshuffle=False)
    body_row: Callable | None = None,
    # CShuffle epilog (required when use_cshuffle=True)
    vector=None,
    gpu=None,
    scf=None,
    tile_m: int | None = None,
    tile_n: int | None = None,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    num_acc_n: int | None = None,
    tx=None,
    lane_mod_16=None,
    by_n=None,
    n_tile_base=None,
    lds_out=None,
    write_row_to_lds: Callable | None = None,
    precompute_row: Callable | None = None,
    store_pair: Callable | None = None,
    frag_elem_type: ir.Type | None = None,
):
    if not use_cshuffle:
        if body_row is None:
            raise ValueError("mfma_epilog(use_cshuffle=False) requires `body_row`.")
        return default_epilog(
            arith=arith,
            range_constexpr=range_constexpr,
            m_repeat=m_repeat,
            lane_div_16=lane_div_16,
            bx_m=bx_m,
            body_row=body_row,
        )

    return c_shuffle_epilog(
        arith=arith,
        vector=vector,
        gpu=gpu,
        scf=scf,
        range_constexpr=range_constexpr,
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        e_vec=int(e_vec),
        cshuffle_nlane=int(cshuffle_nlane),
        block_size=int(block_size),
        m_repeat=m_repeat,
        num_acc_n=int(num_acc_n),
        tx=tx,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        bx_m=bx_m,
        by_n=by_n,
        n_tile_base=n_tile_base,
        lds_out=lds_out,
        frag_elem_type=frag_elem_type,
        write_row_to_lds=write_row_to_lds,
        precompute_row=precompute_row,
        store_pair=store_pair,
    )
