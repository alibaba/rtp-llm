# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Adapted for rtp-llm: forward-only, supports USE_EXP2 for log2-space gates.

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.autotune_cache import (
    autotune_cache_kwargs,
    cuda_cached_autotune,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    _validate_selected_state_output_alias,
)
from rtp_llm.models_py.triton_kernels.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from rtp_llm.models_py.triton_kernels.fla.op import exp, exp2

_STATE_HEURISTICS = {
    "USE_G": lambda args: args["g"] is not None,
    "USE_GK": lambda args: args["gk"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
}
_STATE_CONFIGS = [
    triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
    for num_warps in [2, 4]
    for num_stages in [2, 3, 4]
    for BV in [32, 64]
]
_STATE_TUNE_KEY = ["H", "HV", "K", "V", "BT", "USE_EXP2", "TRANSPOSE_STATE"]


@triton.jit
def _selected_state_block_ptr(
    state,
    i_v,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    OFFSET_K: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    if TRANSPOSE_STATE:
        return tl.make_block_ptr(
            state, (V, K), (K, 1), (i_v * BV, OFFSET_K), (BV, 64), (1, 0)
        )
    else:
        return tl.make_block_ptr(
            state, (K, V), (V, 1), (OFFSET_K, i_v * BV), (64, BV), (1, 0)
        )


@triton.jit
def _selected_prefix_weighted_k(
    raw_k,
    gk,
    last_gate,
    bos,
    i_h,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    OFFSET_K: tl.constexpr,
):
    # Match recompute_w_u_fwd_kda_kernel's kg calculation, including its
    # low-precision store, but use the requested prefix as the chunk end.
    p_k = tl.make_block_ptr(raw_k, (K, T), (1, H * K), (OFFSET_K, 0), (64, BT), (0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_gk = tl.make_block_ptr(
        gk + (bos * HV + i_h).to(tl.int64) * K,
        (K, T),
        (1, HV * K),
        (OFFSET_K, 0),
        (64, BT),
        (0, 1),
    )
    b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
    b_kg = b_k * tl.where(
        (tl.arange(0, BT) < T)[None, :],
        exp2(last_gate[:, None] - b_gk),
        0,
    )
    return b_kg.to(b_k.dtype)


@triton.jit(do_not_specialize=["T"])
def _prepare_selected_state_capture(
    state_positions,
    capture_slots,
    cu_seqlens,
    chunk_offsets,
    T,
    N: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_n = tl.program_id(0), tl.program_id(1)
    table_index = i_n * NT + i_t
    if IS_VARLEN:
        # Upper-bound search preserves leading/middle empty sequences, whose
        # repeated offsets are not represented by prepare_chunk_indices IDs.
        lo, hi = 0, N
        while lo < hi:
            mid = (lo + hi + 1) // 2
            start = tl.load(chunk_offsets + mid).to(tl.int32)
            before = start <= i_t
            lo = tl.where(before, mid, lo)
            hi = tl.where(before, hi, mid - 1)
        i_n = lo
        i_t -= tl.load(chunk_offsets + i_n).to(tl.int32)
        T = (tl.load(cu_seqlens + i_n + 1) - tl.load(cu_seqlens + i_n)).to(tl.int32)
    slots = tl.arange(0, BLOCK_S)
    prefixes = tl.load(state_positions + i_n * S + slots, mask=slots < S, other=0)
    valid = (slots < S) & (prefixes > 0) & (prefixes <= T)
    chunk = (tl.maximum(prefixes, 1) - 1) // BT
    first_slot = tl.min(tl.where(valid & (chunk == i_t), slots, S), 0)
    tl.store(capture_slots + table_index, tl.where(first_slot < S, first_slot, -1))


@triton.jit(do_not_specialize=["T"])
def _chunk_gated_delta_rule_state_kernel_base(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    raw_k=None,
    state_positions=None,
    selected_states=None,
    NUM_STATE_POSITIONS: tl.constexpr = 0,
    CAPTURE_STATES: tl.constexpr = False,
    SELECT_STATES: tl.constexpr = False,
    capture_slots=None,
    captured_states=None,
    USE_CAPTURE_MAP: tl.constexpr = False,
    REUSE_NEW_VALUE: tl.constexpr = False,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if SELECT_STATES:
        i_s = tl.program_id(2)
        prefix = tl.load(state_positions + i_n * NUM_STATE_POSITIONS + i_s)
        valid_prefix = (prefix >= 0) & (prefix <= T)
        prefix = tl.where(valid_prefix, prefix, 0).to(tl.int32)
        chunk_start = (tl.maximum(prefix, 1) - 1) // BT * BT
        bos += chunk_start
        T = prefix - chunk_start
        NT = tl.cdiv(T, BT)
        selected_state = (
            selected_states
            + ((i_n * NUM_STATE_POSITIONS + i_s) * HV + i_h).to(tl.int64) * K * V
        )
        if USE_CAPTURE_MAP:
            # Duplicate requests read the same immutable captured start. The
            # output slots remain independent, so replay has no alias races.
            captured_slot = tl.load(
                capture_slots + boh + chunk_start // BT, mask=prefix > 0, other=0
            )
            selected_start = (
                captured_states
                + ((i_n * NUM_STATE_POSITIONS + captured_slot) * HV + i_h).to(tl.int64)
                * K
                * V
            )
        else:
            selected_start = selected_state

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SELECT_STATES:
        raw_k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        if SELECT_STATES:
            ht = selected_state
        else:
            ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h0_1_m0 = (p_h0_1_i0 >= 0) & (p_h0_1_i0 < (V))
            p_h0_1_i1 = tl.arange(0, 64).to(tl.int64) + (0)
            p_h0_1_m1 = (p_h0_1_i1 >= 0) & (p_h0_1_i1 < (K))
            p_h0_1 = (h0) + p_h0_1_i0[:, None] * (K) + p_h0_1_i1[None, :] * (1)
        else:
            p_h0_1_i0 = tl.arange(0, 64).to(tl.int64) + (0)
            p_h0_1_m0 = (p_h0_1_i0 >= 0) & (p_h0_1_i0 < (K))
            p_h0_1_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_h0_1_m1 = (p_h0_1_i1 >= 0) & (p_h0_1_i1 < (V))
            p_h0_1 = (h0) + p_h0_1_i0[:, None] * (V) + p_h0_1_i1[None, :] * (1)
        b_h1 += tl.load(
            p_h0_1, mask=p_h0_1_m0[:, None] & p_h0_1_m1[None, :], other=0
        ).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_h0_2_m0 = (p_h0_2_i0 >= 0) & (p_h0_2_i0 < (V))
                p_h0_2_i1 = tl.arange(0, 64).to(tl.int64) + (64)
                p_h0_2_m1 = (p_h0_2_i1 >= 0) & (p_h0_2_i1 < (K))
                p_h0_2 = (h0) + p_h0_2_i0[:, None] * (K) + p_h0_2_i1[None, :] * (1)
            else:
                p_h0_2_i0 = tl.arange(0, 64).to(tl.int64) + (64)
                p_h0_2_m0 = (p_h0_2_i0 >= 0) & (p_h0_2_i0 < (K))
                p_h0_2_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_h0_2_m1 = (p_h0_2_i1 >= 0) & (p_h0_2_i1 < (V))
                p_h0_2 = (h0) + p_h0_2_i0[:, None] * (V) + p_h0_2_i1[None, :] * (1)
            b_h2 += tl.load(
                p_h0_2, mask=p_h0_2_m0[:, None] & p_h0_2_m1[None, :], other=0
            ).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_h0_3_m0 = (p_h0_3_i0 >= 0) & (p_h0_3_i0 < (V))
                p_h0_3_i1 = tl.arange(0, 64).to(tl.int64) + (128)
                p_h0_3_m1 = (p_h0_3_i1 >= 0) & (p_h0_3_i1 < (K))
                p_h0_3 = (h0) + p_h0_3_i0[:, None] * (K) + p_h0_3_i1[None, :] * (1)
            else:
                p_h0_3_i0 = tl.arange(0, 64).to(tl.int64) + (128)
                p_h0_3_m0 = (p_h0_3_i0 >= 0) & (p_h0_3_i0 < (K))
                p_h0_3_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_h0_3_m1 = (p_h0_3_i1 >= 0) & (p_h0_3_i1 < (V))
                p_h0_3 = (h0) + p_h0_3_i0[:, None] * (V) + p_h0_3_i1[None, :] * (1)
            b_h3 += tl.load(
                p_h0_3, mask=p_h0_3_m0[:, None] & p_h0_3_m1[None, :], other=0
            ).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_h0_4_m0 = (p_h0_4_i0 >= 0) & (p_h0_4_i0 < (V))
                p_h0_4_i1 = tl.arange(0, 64).to(tl.int64) + (192)
                p_h0_4_m1 = (p_h0_4_i1 >= 0) & (p_h0_4_i1 < (K))
                p_h0_4 = (h0) + p_h0_4_i0[:, None] * (K) + p_h0_4_i1[None, :] * (1)
            else:
                p_h0_4_i0 = tl.arange(0, 64).to(tl.int64) + (192)
                p_h0_4_m0 = (p_h0_4_i0 >= 0) & (p_h0_4_i0 < (K))
                p_h0_4_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_h0_4_m1 = (p_h0_4_i1 >= 0) & (p_h0_4_i1 < (V))
                p_h0_4 = (h0) + p_h0_4_i0[:, None] * (V) + p_h0_4_i1[None, :] * (1)
            b_h4 += tl.load(
                p_h0_4, mask=p_h0_4_m0[:, None] & p_h0_4_m1[None, :], other=0
            ).to(tl.float32)

    if SELECT_STATES:
        # Main recurrence captures exact FP32 accumulators in these slots;
        # the ordinary h buffer is low precision and cannot be used here.
        if prefix > 0:
            p_selected = _selected_state_block_ptr(
                selected_start, i_v, K, V, BV, 0, TRANSPOSE_STATE
            )
            b_h1 = tl.load(p_selected, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                p_selected = _selected_state_block_ptr(
                    selected_start, i_v, K, V, BV, 64, TRANSPOSE_STATE
                )
                b_h2 = tl.load(p_selected, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                p_selected = _selected_state_block_ptr(
                    selected_start, i_v, K, V, BV, 128, TRANSPOSE_STATE
                )
                b_h3 = tl.load(p_selected, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                p_selected = _selected_state_block_ptr(
                    selected_start, i_v, K, V, BV, 192, TRANSPOSE_STATE
                )
                b_h4 = tl.load(p_selected, boundary_check=(0, 1)).to(tl.float32)
        elif not valid_prefix:
            b_h1 = tl.full(b_h1.shape, 0, tl.float32)
            if K > 64:
                b_h2 = tl.full(b_h2.shape, 0, tl.float32)
            if K > 128:
                b_h3 = tl.full(b_h3.shape, 0, tl.float32)
            if K > 192:
                b_h4 = tl.full(b_h4.shape, 0, tl.float32)

    if CAPTURE_STATES and not USE_CAPTURE_MAP and NUM_STATE_POSITIONS <= 16:
        # Load sparse requests once. Keep ordinary recurrence chunks free
        # of repeated position loads and a runtime inner request loop.
        request_slots = tl.arange(0, triton.next_power_of_2(NUM_STATE_POSITIONS))
        requested_prefixes = tl.load(
            state_positions + i_n * NUM_STATE_POSITIONS + request_slots,
            mask=request_slots < NUM_STATE_POSITIONS,
            other=0,
        )
        requested_chunks = tl.where(
            (requested_prefixes > 0) & (requested_prefixes <= T),
            (tl.maximum(requested_prefixes, 1) - 1) // BT,
            NT,
        ).to(tl.int32)
        next_capture_chunk = tl.min(requested_chunks, 0)

    # main recurrence
    for i_t in range(NT):
        if CAPTURE_STATES:
            if USE_CAPTURE_MAP:
                capture_slot = tl.load(capture_slots + boh + i_t)
                if capture_slot >= 0:
                    captured_state = (
                        captured_states
                        + ((i_n * NUM_STATE_POSITIONS + capture_slot) * HV + i_h).to(
                            tl.int64
                        )
                        * K
                        * V
                    )
                    p_selected = _selected_state_block_ptr(
                        captured_state, i_v, K, V, BV, 0, TRANSPOSE_STATE
                    )
                    tl.store(p_selected, b_h1, boundary_check=(0, 1))
                    if K > 64:
                        p_selected = _selected_state_block_ptr(
                            captured_state, i_v, K, V, BV, 64, TRANSPOSE_STATE
                        )
                        tl.store(p_selected, b_h2, boundary_check=(0, 1))
                    if K > 128:
                        p_selected = _selected_state_block_ptr(
                            captured_state, i_v, K, V, BV, 128, TRANSPOSE_STATE
                        )
                        tl.store(p_selected, b_h3, boundary_check=(0, 1))
                    if K > 192:
                        p_selected = _selected_state_block_ptr(
                            captured_state, i_v, K, V, BV, 192, TRANSPOSE_STATE
                        )
                        tl.store(p_selected, b_h4, boundary_check=(0, 1))
            elif NUM_STATE_POSITIONS <= 16:
                if i_t == next_capture_chunk:
                    for i_s in tl.static_range(NUM_STATE_POSITIONS):
                        request_chunk = tl.sum(
                            tl.where(request_slots == i_s, requested_chunks, 0), 0
                        )
                        if request_chunk == i_t:
                            selected_state = (
                                selected_states
                                + ((i_n * NUM_STATE_POSITIONS + i_s) * HV + i_h).to(
                                    tl.int64
                                )
                                * K
                                * V
                            )
                            p_selected = _selected_state_block_ptr(
                                selected_state, i_v, K, V, BV, 0, TRANSPOSE_STATE
                            )
                            tl.store(p_selected, b_h1, boundary_check=(0, 1))
                            if K > 64:
                                p_selected = _selected_state_block_ptr(
                                    selected_state, i_v, K, V, BV, 64, TRANSPOSE_STATE
                                )
                                tl.store(p_selected, b_h2, boundary_check=(0, 1))
                            if K > 128:
                                p_selected = _selected_state_block_ptr(
                                    selected_state, i_v, K, V, BV, 128, TRANSPOSE_STATE
                                )
                                tl.store(p_selected, b_h3, boundary_check=(0, 1))
                            if K > 192:
                                p_selected = _selected_state_block_ptr(
                                    selected_state, i_v, K, V, BV, 192, TRANSPOSE_STATE
                                )
                                tl.store(p_selected, b_h4, boundary_check=(0, 1))
                    # All duplicates at this boundary have been captured.
                    next_capture_chunk = tl.min(
                        tl.where(requested_chunks > i_t, requested_chunks, NT), 0
                    )
            else:
                for i_s in range(NUM_STATE_POSITIONS):
                    prefix = tl.load(state_positions + i_n * NUM_STATE_POSITIONS + i_s)
                    if (prefix > 0) & (prefix <= T) & ((prefix - 1) // BT == i_t):
                        selected_state = (
                            selected_states
                            + ((i_n * NUM_STATE_POSITIONS + i_s) * HV + i_h).to(
                                tl.int64
                            )
                            * K
                            * V
                        )
                        p_selected = _selected_state_block_ptr(
                            selected_state, i_v, K, V, BV, 0, TRANSPOSE_STATE
                        )
                        tl.store(p_selected, b_h1, boundary_check=(0, 1))
                        if K > 64:
                            p_selected = _selected_state_block_ptr(
                                selected_state, i_v, K, V, BV, 64, TRANSPOSE_STATE
                            )
                            tl.store(p_selected, b_h2, boundary_check=(0, 1))
                        if K > 128:
                            p_selected = _selected_state_block_ptr(
                                selected_state, i_v, K, V, BV, 128, TRANSPOSE_STATE
                            )
                            tl.store(p_selected, b_h3, boundary_check=(0, 1))
                        if K > 192:
                            p_selected = _selected_state_block_ptr(
                                selected_state, i_v, K, V, BV, 192, TRANSPOSE_STATE
                            )
                            tl.store(p_selected, b_h4, boundary_check=(0, 1))

        if not SELECT_STATES:
            i_t_int64 = i_t.to(tl.int64)
            if TRANSPOSE_STATE:
                p_h1_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_h1_m0 = (p_h1_i0 >= 0) & (p_h1_i0 < (V))
                p_h1_i1 = tl.arange(0, 64).to(tl.int64) + (0)
                p_h1_m1 = (p_h1_i1 >= 0) & (p_h1_i1 < (K))
                p_h1 = (
                    (h + i_t_int64 * HV * K * V)
                    + p_h1_i0[:, None] * (K)
                    + p_h1_i1[None, :] * (1)
                )
            else:
                p_h1_i0 = tl.arange(0, 64).to(tl.int64) + (0)
                p_h1_m0 = (p_h1_i0 >= 0) & (p_h1_i0 < (K))
                p_h1_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_h1_m1 = (p_h1_i1 >= 0) & (p_h1_i1 < (V))
                p_h1 = (
                    (h + i_t_int64 * HV * K * V)
                    + p_h1_i0[:, None] * (V)
                    + p_h1_i1[None, :] * (1)
                )
            tl.store(
                p_h1,
                b_h1.to(p_h1.dtype.element_ty),
                mask=p_h1_m0[:, None] & p_h1_m1[None, :],
            )
            if K > 64:
                if TRANSPOSE_STATE:
                    p_h2_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                    p_h2_m0 = (p_h2_i0 >= 0) & (p_h2_i0 < (V))
                    p_h2_i1 = tl.arange(0, 64).to(tl.int64) + (64)
                    p_h2_m1 = (p_h2_i1 >= 0) & (p_h2_i1 < (K))
                    p_h2 = (
                        (h + i_t_int64 * HV * K * V)
                        + p_h2_i0[:, None] * (K)
                        + p_h2_i1[None, :] * (1)
                    )
                else:
                    p_h2_i0 = tl.arange(0, 64).to(tl.int64) + (64)
                    p_h2_m0 = (p_h2_i0 >= 0) & (p_h2_i0 < (K))
                    p_h2_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                    p_h2_m1 = (p_h2_i1 >= 0) & (p_h2_i1 < (V))
                    p_h2 = (
                        (h + i_t_int64 * HV * K * V)
                        + p_h2_i0[:, None] * (V)
                        + p_h2_i1[None, :] * (1)
                    )
                tl.store(
                    p_h2,
                    b_h2.to(p_h2.dtype.element_ty),
                    mask=p_h2_m0[:, None] & p_h2_m1[None, :],
                )
            if K > 128:
                if TRANSPOSE_STATE:
                    p_h3_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                    p_h3_m0 = (p_h3_i0 >= 0) & (p_h3_i0 < (V))
                    p_h3_i1 = tl.arange(0, 64).to(tl.int64) + (128)
                    p_h3_m1 = (p_h3_i1 >= 0) & (p_h3_i1 < (K))
                    p_h3 = (
                        (h + i_t_int64 * HV * K * V)
                        + p_h3_i0[:, None] * (K)
                        + p_h3_i1[None, :] * (1)
                    )
                else:
                    p_h3_i0 = tl.arange(0, 64).to(tl.int64) + (128)
                    p_h3_m0 = (p_h3_i0 >= 0) & (p_h3_i0 < (K))
                    p_h3_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                    p_h3_m1 = (p_h3_i1 >= 0) & (p_h3_i1 < (V))
                    p_h3 = (
                        (h + i_t_int64 * HV * K * V)
                        + p_h3_i0[:, None] * (V)
                        + p_h3_i1[None, :] * (1)
                    )
                tl.store(
                    p_h3,
                    b_h3.to(p_h3.dtype.element_ty),
                    mask=p_h3_m0[:, None] & p_h3_m1[None, :],
                )
            if K > 192:
                if TRANSPOSE_STATE:
                    p_h4_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                    p_h4_m0 = (p_h4_i0 >= 0) & (p_h4_i0 < (V))
                    p_h4_i1 = tl.arange(0, 64).to(tl.int64) + (192)
                    p_h4_m1 = (p_h4_i1 >= 0) & (p_h4_i1 < (K))
                    p_h4 = (
                        (h + i_t_int64 * HV * K * V)
                        + p_h4_i0[:, None] * (K)
                        + p_h4_i1[None, :] * (1)
                    )
                else:
                    p_h4_i0 = tl.arange(0, 64).to(tl.int64) + (192)
                    p_h4_m0 = (p_h4_i0 >= 0) & (p_h4_i0 < (K))
                    p_h4_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                    p_h4_m1 = (p_h4_i1 >= 0) & (p_h4_i1 < (V))
                    p_h4 = (
                        (h + i_t_int64 * HV * K * V)
                        + p_h4_i0[:, None] * (V)
                        + p_h4_i1[None, :] * (1)
                    )
                tl.store(
                    p_h4,
                    b_h4.to(p_h4.dtype.element_ty),
                    mask=p_h4_m0[:, None] & p_h4_m1[None, :],
                )

        if SELECT_STATES and REUSE_NEW_VALUE:
            # g=None: the ordinary recurrence casts u - w@h to k.dtype
            # immediately before its update. v_new stores that same cast.
            p_v = tl.make_block_ptr(
                v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))
        else:
            p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
            p_w_i1 = tl.arange(0, 64).to(tl.int64) + (0)
            p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
            p_w = (w) + p_w_i0[:, None] * (HV * K) + p_w_i1[None, :] * (1)
            b_w = tl.load(p_w, mask=p_w_m0[:, None] & p_w_m1[None, :], other=0)
            if TRANSPOSE_STATE:
                b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
            else:
                b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
                p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
                p_w_i1 = tl.arange(0, 64).to(tl.int64) + (64)
                p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
                p_w = (w) + p_w_i0[:, None] * (HV * K) + p_w_i1[None, :] * (1)
                b_w = tl.load(p_w, mask=p_w_m0[:, None] & p_w_m1[None, :], other=0)
                if TRANSPOSE_STATE:
                    b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
                else:
                    b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
                p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
                p_w_i1 = tl.arange(0, 64).to(tl.int64) + (128)
                p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
                p_w = (w) + p_w_i0[:, None] * (HV * K) + p_w_i1[None, :] * (1)
                b_w = tl.load(p_w, mask=p_w_m0[:, None] & p_w_m1[None, :], other=0)
                if TRANSPOSE_STATE:
                    b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
                else:
                    b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
                p_w_m0 = (p_w_i0 >= 0) & (p_w_i0 < (T))
                p_w_i1 = tl.arange(0, 64).to(tl.int64) + (192)
                p_w_m1 = (p_w_i1 >= 0) & (p_w_i1 < (K))
                p_w = (w) + p_w_i0[:, None] * (HV * K) + p_w_i1[None, :] * (1)
                b_w = tl.load(p_w, mask=p_w_m0[:, None] & p_w_m1[None, :], other=0)
                if TRANSPOSE_STATE:
                    b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
                else:
                    b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
            p_v_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_v_m0 = (p_v_i0 >= 0) & (p_v_i0 < (T))
            p_v_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_v_m1 = (p_v_i1 >= 0) & (p_v_i1 < (V))
            p_v = (v) + p_v_i0[:, None] * (HV * V) + p_v_i1[None, :] * (1)
            b_v = tl.load(p_v, mask=p_v_m0[:, None] & p_v_m1[None, :], other=0) - b_v

        if SAVE_NEW_VALUE:
            p_v_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_v_m0 = (p_v_i0 >= 0) & (p_v_i0 < (T))
            p_v_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_v_m1 = (p_v_i1 >= 0) & (p_v_i1 < (V))
            p_v = (v_new) + p_v_i0[:, None] * (HV * V) + p_v_i1[None, :] * (1)
            tl.store(
                p_v,
                b_v.to(p_v.dtype.element_ty),
                mask=p_v_m0[:, None] & p_v_m1[None, :],
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(
                tl.float32
            )
            p_g_i0 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_g_m0 = (p_g_i0 >= 0) & (p_g_i0 < (T))
            p_g = (g + (bos * HV + i_h).to(tl.int64)) + p_g_i0 * (HV)
            b_g = tl.load(p_g, mask=p_g_m0, other=0).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * HV * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            ).to(tl.float32)
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_h2 *= exp(b_gk_last2)[None, :]
                else:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_h3 *= exp(b_gk_last3)[None, :]
                else:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_h4 *= exp(b_gk_last4)[None, :]
                else:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        if SELECT_STATES:
            b_k = _selected_prefix_weighted_k(
                raw_k, gk, b_gk_last1, bos, i_h, T, H, HV, K, BT, 0
            )
        else:
            p_k_i0 = tl.arange(0, 64).to(tl.int64) + (0)
            p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (K))
            p_k_i1 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
            p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (T))
            p_k = (k) + p_k_i0[:, None] * (1) + p_k_i1[None, :] * (H * K)
            b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            if SELECT_STATES:
                b_k = _selected_prefix_weighted_k(
                    raw_k, gk, b_gk_last2, bos, i_h, T, H, HV, K, BT, 64
                )
            else:
                p_k_i0 = tl.arange(0, 64).to(tl.int64) + (64)
                p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (K))
                p_k_i1 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
                p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (T))
                p_k = (k) + p_k_i0[:, None] * (1) + p_k_i1[None, :] * (H * K)
                b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            if SELECT_STATES:
                b_k = _selected_prefix_weighted_k(
                    raw_k, gk, b_gk_last3, bos, i_h, T, H, HV, K, BT, 128
                )
            else:
                p_k_i0 = tl.arange(0, 64).to(tl.int64) + (128)
                p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (K))
                p_k_i1 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
                p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (T))
                p_k = (k) + p_k_i0[:, None] * (1) + p_k_i1[None, :] * (H * K)
                b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            if SELECT_STATES:
                b_k = _selected_prefix_weighted_k(
                    raw_k, gk, b_gk_last4, bos, i_h, T, H, HV, K, BT, 192
                )
            else:
                p_k_i0 = tl.arange(0, 64).to(tl.int64) + (192)
                p_k_m0 = (p_k_i0 >= 0) & (p_k_i0 < (K))
                p_k_i1 = tl.arange(0, BT).to(tl.int64) + (i_t * BT)
                p_k_m1 = (p_k_i1 >= 0) & (p_k_i1 < (T))
                p_k = (k) + p_k_i0[:, None] * (1) + p_k_i1[None, :] * (H * K)
                b_k = tl.load(p_k, mask=p_k_m0[:, None] & p_k_m1[None, :], other=0)
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (V))
            p_ht_i1 = tl.arange(0, 64).to(tl.int64) + (0)
            p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (K))
            p_ht = (ht) + p_ht_i0[:, None] * (K) + p_ht_i1[None, :] * (1)
        else:
            p_ht_i0 = tl.arange(0, 64).to(tl.int64) + (0)
            p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (K))
            p_ht_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
            p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (V))
            p_ht = (ht) + p_ht_i0[:, None] * (V) + p_ht_i1[None, :] * (1)
        tl.store(
            p_ht,
            b_h1.to(p_ht.dtype.element_ty),
            mask=p_ht_m0[:, None] & p_ht_m1[None, :],
        )
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (V))
                p_ht_i1 = tl.arange(0, 64).to(tl.int64) + (64)
                p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (K))
                p_ht = (ht) + p_ht_i0[:, None] * (K) + p_ht_i1[None, :] * (1)
            else:
                p_ht_i0 = tl.arange(0, 64).to(tl.int64) + (64)
                p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (K))
                p_ht_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (V))
                p_ht = (ht) + p_ht_i0[:, None] * (V) + p_ht_i1[None, :] * (1)
            tl.store(
                p_ht,
                b_h2.to(p_ht.dtype.element_ty),
                mask=p_ht_m0[:, None] & p_ht_m1[None, :],
            )
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (V))
                p_ht_i1 = tl.arange(0, 64).to(tl.int64) + (128)
                p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (K))
                p_ht = (ht) + p_ht_i0[:, None] * (K) + p_ht_i1[None, :] * (1)
            else:
                p_ht_i0 = tl.arange(0, 64).to(tl.int64) + (128)
                p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (K))
                p_ht_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (V))
                p_ht = (ht) + p_ht_i0[:, None] * (V) + p_ht_i1[None, :] * (1)
            tl.store(
                p_ht,
                b_h3.to(p_ht.dtype.element_ty),
                mask=p_ht_m0[:, None] & p_ht_m1[None, :],
            )
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht_i0 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (V))
                p_ht_i1 = tl.arange(0, 64).to(tl.int64) + (192)
                p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (K))
                p_ht = (ht) + p_ht_i0[:, None] * (K) + p_ht_i1[None, :] * (1)
            else:
                p_ht_i0 = tl.arange(0, 64).to(tl.int64) + (192)
                p_ht_m0 = (p_ht_i0 >= 0) & (p_ht_i0 < (K))
                p_ht_i1 = tl.arange(0, BV).to(tl.int64) + (i_v * BV)
                p_ht_m1 = (p_ht_i1 >= 0) & (p_ht_i1 < (V))
                p_ht = (ht) + p_ht_i0[:, None] * (V) + p_ht_i1[None, :] * (1)
            tl.store(
                p_ht,
                b_h4.to(p_ht.dtype.element_ty),
                mask=p_ht_m0[:, None] & p_ht_m1[None, :],
            )


# Replay reads and overwrites selected_states. It must never run through an
# autotuner, whose candidate trials would consume already-updated snapshots.
_chunk_gated_delta_rule_selected_kernel = triton.heuristics(_STATE_HEURISTICS)(
    _chunk_gated_delta_rule_state_kernel_base
)
# Capture uses the same base kernel as the ordinary autotuned path. A separate
# non-autotuned wrapper avoids candidate trials while preserving code structure.
_chunk_gated_delta_rule_capture_kernel = triton.heuristics(_STATE_HEURISTICS)(
    _chunk_gated_delta_rule_state_kernel_base
)
chunk_gated_delta_rule_fwd_kernel_h_blockdim64 = triton.heuristics(_STATE_HEURISTICS)(
    cuda_cached_autotune(
        configs=_STATE_CONFIGS,
        key=_STATE_TUNE_KEY,
        **autotune_cache_kwargs,
    )(_chunk_gated_delta_rule_state_kernel_base)
)
# Exact launch configurations are cached per Triton autotune key.
_CAPTURE_LAUNCH_CONFIGS = {}


def _ordinary_autotuner():
    obj = chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    while obj is not None and not hasattr(obj, "best_config"):
        obj = getattr(obj, "fn", None)
    return obj


def _ordinary_launch_config():
    """Read the launch configuration chosen by the ordinary autotuned kernel."""
    obj = _ordinary_autotuner()
    config = getattr(obj, "best_config", None)
    if config is None:
        return None
    return {
        "BV": config.kwargs.get("BV", 32),
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def _ordinary_autotune_key(**kwargs):
    """Reconstruct the exact Triton autotune key for an ordinary launch."""
    obj = _ordinary_autotuner()
    if obj is None:
        return None
    arg_names = set(getattr(obj, "arg_names", ()))
    tune_names = list(getattr(obj, "keys", ()))
    filtered = {name: value for name, value in kwargs.items() if name in arg_names}
    key = [filtered[name] for name in tune_names if name in filtered]
    for value in filtered.values():
        if hasattr(value, "dtype"):
            key.append(str(value.dtype))
    return tuple(key)


def _validate_selected_states(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_size: int,
    use_exp2: bool,
    transpose_state_layout: bool,
    state_positions: torch.Tensor,
    selected_states: torch.Tensor,
    raw_k: torch.Tensor | None,
) -> None:
    """Check support from metadata without reading device positions on CPU."""
    if k.device.type != "cuda" or torch.version.hip is not None:
        raise NotImplementedError("selected KDA states currently require NVIDIA CUDA")
    if g is not None or gk is None or not use_exp2:
        raise NotImplementedError("selected KDA states require log2 gk and no scalar g")
    if raw_k is None:
        raise ValueError("selected KDA states require original normalized raw_k")
    if chunk_size != 64:
        raise ValueError("selected KDA states require chunk_size=64")
    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4:
        raise ValueError("k, w and u must have rank 4")
    B, T, H, K = k.shape
    HV, V = u.shape[-2:]
    if min(B, H, HV, K, V) <= 0 or K > 256:
        raise ValueError("invalid KDA batch or head dimensions")
    if H != HV:
        raise NotImplementedError(
            "selected KDA states require equal key/value head counts"
        )
    if u.shape[:2] != (B, T) or w.shape != (B, T, HV, K):
        raise ValueError("k, w and u have incompatible shapes")
    if k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("k must have floating-point dtype supported by KDA")
    if w.dtype != k.dtype or u.dtype != k.dtype:
        raise ValueError("k, w and u must have the same dtype")
    if raw_k.shape != k.shape or raw_k.dtype != k.dtype:
        raise ValueError("raw_k must have the same shape and dtype as k")
    if gk.shape != (B, T, HV, K) or not gk.is_floating_point():
        raise ValueError("gk must be a floating-point tensor of shape [B, T, HV, K]")
    if cu_seqlens is None:
        N = B
    else:
        if (
            B != 1
            or cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("cu_seqlens must be an integer vector for a packed batch")
        N = cu_seqlens.numel() - 1
    if chunk_indices is not None and (
        chunk_indices.ndim != 2
        or chunk_indices.shape[-1] != 2
        or chunk_indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("chunk_indices must be an integer tensor of shape [NT, 2]")
    if state_positions.ndim != 2 or state_positions.shape[0] != N:
        raise ValueError("state_positions must have shape [N, S]")
    if state_positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("state_positions must have int32 or int64 dtype")
    S = state_positions.shape[1]
    if S > 65535:
        raise ValueError("at most 65535 state positions per sequence are supported")
    state_shape = (V, K) if transpose_state_layout else (K, V)
    if selected_states.shape != (N, S, HV, *state_shape):
        raise ValueError("selected_states has an incompatible shape or state layout")
    if selected_states.dtype != torch.float32:
        raise ValueError("selected_states must have FP32 dtype")
    if initial_state is not None and (
        initial_state.shape != (N, HV, *state_shape)
        or initial_state.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError(
            "initial_state has an incompatible shape, dtype or state layout"
        )
    named_inputs = dict(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        state_positions=state_positions,
        raw_k=raw_k,
    )
    for name, tensor in (*named_inputs.items(), ("selected_states", selected_states)):
        if tensor is not None:
            if tensor.device != k.device:
                raise ValueError(f"{name} must be on the same device as k")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
    _validate_selected_state_output_alias(selected_states, **named_inputs)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
    state_positions: torch.Tensor | None = None,
    selected_states: torch.Tensor | None = None,
    raw_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute KDA chunk states, optionally writing arbitrary prefix states.

    state_positions is a contiguous int32/int64 CUDA [N, S] tensor of processed
    token counts local to each sequence. Zero selects the initial state;
    negative or overlength positions write zeros. Duplicates/unsorted positions
    are allowed. selected_states is a caller-owned contiguous FP32 buffer with
    shape [N, S, HV, K, V] (or [N, S, HV, V, K] with transposed state layout).
    Both optional arguments must be supplied together, with raw_k containing
    the original normalized keys used to construct the full-chunk weighted k.
    Snapshot support currently requires NVIDIA CUDA and log2 gk gates, with
    no scalar g. Original return values and ordinary h dtype remain unchanged.
    """
    if (state_positions is None) != (selected_states is None):
        raise ValueError(
            "state_positions and selected_states must be provided together"
        )
    if state_positions is not None:
        _validate_selected_states(
            k,
            w,
            u,
            g,
            gk,
            initial_state,
            cu_seqlens,
            chunk_indices,
            chunk_size,
            use_exp2,
            transpose_state_layout,
            state_positions,
            selected_states,
            raw_k,
        )
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if transpose_state_layout:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = (
            k.new_zeros(N, HV, V, K, dtype=torch.float32)
            if output_final_state
            else None
        )
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = (
            k.new_zeros(N, HV, K, V, dtype=torch.float32)
            if output_final_state
            else None
        )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * HV)

    capture_states = state_positions is not None and state_positions.shape[1] > 0
    # Sparse metadata avoids keeping many request-dependent pointers live in
    # the recurrent kernel. Scratch scales with requested states, not chunks.
    use_capture_map = capture_states and 4 < state_positions.shape[1] <= 32
    capture_slots = None
    captured_states = None
    if use_capture_map:
        capture_slots = k.new_empty(B * NT, dtype=torch.int32)
        captured_states = torch.empty_like(selected_states)
        if NT > 0:
            _prepare_selected_state_capture[(NT, B)](
                state_positions=state_positions,
                capture_slots=capture_slots,
                cu_seqlens=cu_seqlens,
                chunk_offsets=chunk_offsets,
                T=T,
                N=N,
                NT=NT,
                BT=BT,
                S=state_positions.shape[1],
                BLOCK_S=triton.next_power_of_2(state_positions.shape[1]),
                IS_VARLEN=cu_seqlens is not None,
                num_warps=1,
            )
    capture_launch = None
    replay_launch = None
    if capture_states:
        base_launch_kwargs = dict(
            k=k,
            v=u,
            w=w,
            g=g,
            gk=gk,
            h=h,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            USE_EXP2=use_exp2,
            TRANSPOSE_STATE=transpose_state_layout,
        )
        # Capture must preserve the caller's ordinary outputs, including the
        # save_new_value=False autotune specialization. Replay must match the
        # prefix oracle, which always materializes v_new before recurrence.
        capture_kwargs = dict(base_launch_kwargs, v_new=v_new)
        capture_key = _ordinary_autotune_key(**capture_kwargs)
        if capture_key not in _CAPTURE_LAUNCH_CONFIGS:
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](**capture_kwargs)
            _CAPTURE_LAUNCH_CONFIGS[capture_key] = _ordinary_launch_config() or {
                "BV": 32,
                "num_warps": 4,
                "num_stages": 3,
            }
        capture_launch = _CAPTURE_LAUNCH_CONFIGS[capture_key]

        replay_v_new = v_new if v_new is not None else torch.empty_like(u)
        replay_kwargs = dict(base_launch_kwargs, v_new=replay_v_new)
        replay_key = _ordinary_autotune_key(**replay_kwargs)
        if replay_key not in _CAPTURE_LAUNCH_CONFIGS:
            chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](**replay_kwargs)
            _CAPTURE_LAUNCH_CONFIGS[replay_key] = _ordinary_launch_config() or {
                "BV": 32,
                "num_warps": 4,
                "num_stages": 3,
            }
        replay_launch = _CAPTURE_LAUNCH_CONFIGS[replay_key]

    state_kernel = (
        _chunk_gated_delta_rule_capture_kernel
        if capture_states
        else chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    )
    snapshot_kwargs = (
        dict(
            state_positions=state_positions,
            selected_states=selected_states,
            NUM_STATE_POSITIONS=state_positions.shape[1],
            CAPTURE_STATES=True,
            capture_slots=capture_slots,
            captured_states=captured_states,
            USE_CAPTURE_MAP=use_capture_map,
        )
        if capture_states
        else {}
    )
    state_kernel[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
        **snapshot_kwargs,
        **(
            dict(
                BV=capture_launch["BV"],
                num_warps=capture_launch["num_warps"],
                num_stages=capture_launch["num_stages"],
            )
            if capture_states
            else {}
        ),
    )
    if capture_states:
        launch_config = replay_launch
        _chunk_gated_delta_rule_selected_kernel[
            (triton.cdiv(V, launch_config["BV"]), N * HV, state_positions.shape[1])
        ](
            k=k,
            v=v_new if v_new is not None else u,
            w=w,
            v_new=None,
            g=None,
            gk=gk,
            h=h,
            h0=initial_state,
            ht=selected_states,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BV=launch_config["BV"],
            USE_EXP2=True,
            TRANSPOSE_STATE=transpose_state_layout,
            raw_k=raw_k,
            state_positions=state_positions,
            selected_states=selected_states,
            NUM_STATE_POSITIONS=state_positions.shape[1],
            SELECT_STATES=True,
            capture_slots=capture_slots,
            captured_states=captured_states,
            USE_CAPTURE_MAP=use_capture_map,
            REUSE_NEW_VALUE=v_new is not None,
            num_warps=launch_config["num_warps"],
            num_stages=launch_config["num_stages"],
        )
    return h, v_new, final_state
