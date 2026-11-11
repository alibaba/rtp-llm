"""Prepare main-model decode RoPE and undo rows in one read-only launch.

The following writer remains a separate launch so every request snapshots
shared sentinel rows before any request can change them.
"""

import torch
import triton
import triton.language as tl

from .decode_snapshot_triton import _snapshot_decode, prepare_decode_snapshot
from .draft_decode_positions_triton import (
    _allocate_decode_positions,
    _decode_positions_config,
    _draft_decode_positions,
)
from .indexer_norm_rope import _norm_rope, _norm_rope_supported


@triton.jit
def _prepare_decode(
    SEQ,
    POS,
    INV,
    CC,
    CS,
    BC,
    BS,
    VISIBLE,
    COMPRESSED,
    KV,
    KT,
    STATE,
    ST,
    SS,
    OS,
    KS,
    OK,
    ROWS,
    KTW,
    STW,
    KB: tl.constexpr,
    SB: tl.constexpr,
    INV_SCALE: tl.constexpr,
    H_SECTION: tl.constexpr,
    W_SECTION: tl.constexpr,
    Q=None,
    QG=None,
    QO=None,
    Q_STRIDE: tl.constexpr = 0,
    EPS: tl.constexpr = 1e-6,
    FUSE_Q: tl.constexpr = False,
):
    _draft_decode_positions(
        SEQ,
        POS,
        INV,
        CC,
        CS,
        BC,
        BS,
        VISIBLE,
        COMPRESSED,
        ROWS,
        FACTOR=3,
        INV_SCALE=INV_SCALE,
        BLOCK=1,
        MROPE=True,
        H_SECTION=H_SECTION,
        W_SECTION=W_SECTION,
    )
    _snapshot_decode(SEQ, KV, KT, STATE, ST, SS, OS, KS, OK, KTW, STW, KB, SB)
    if FUSE_Q:
        # The four query heads consume this request's freshly written RoPE.
        tl.debug_barrier()
        _norm_rope(Q, QG, CC, CS, QO, 4, Q_STRIDE, EPS, HEAD_BLOCK=4)


def try_prepare_decode(
    sequence,
    positions,
    frequency,
    config,
    kv,
    kt,
    state,
    st,
    *,
    query=None,
    query_gamma=None,
    norm_eps=1e-6,
):
    settings = _decode_positions_config(
        sequence, positions, frequency, config, target_mrope=True
    )
    if settings is None:
        return None
    undo = prepare_decode_snapshot(sequence, kv, kt, state, st)
    if undo is None:
        return None
    sections, scale = settings
    metadata = _allocate_decode_positions(sequence)
    fuse_q = (
        query is not None
        and query_gamma is not None
        and query.shape == (sequence.numel(), 4, 128)
        and _norm_rope_supported(query, metadata[0], metadata[1], query_gamma)
    )
    rotated_q = (
        torch.empty_like(query, memory_format=torch.contiguous_format)
        if fuse_q
        else None
    )
    _prepare_decode[(sequence.numel(),)](
        sequence,
        positions,
        frequency,
        *metadata,
        kv,
        kt,
        state,
        st,
        undo.state_slots,
        undo.original_state,
        undo.kv_slots,
        undo.original_kv,
        sequence.numel(),
        kt.shape[1],
        st.shape[1],
        kv.shape[0],
        state.shape[0],
        INV_SCALE=1.0 / scale,
        H_SECTION=sections[1],
        W_SECTION=sections[2],
        Q=query if fuse_q else None,
        QG=query_gamma if fuse_q else None,
        QO=rotated_q,
        Q_STRIDE=query.stride(0) if fuse_q else 0,
        EPS=norm_eps,
        FUSE_Q=fuse_q,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return metadata, undo, rotated_q
