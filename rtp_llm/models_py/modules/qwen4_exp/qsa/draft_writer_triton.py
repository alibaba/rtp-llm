"""Ragged draft variant of the qualified two-launch snapshot/write contract."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import IndexerCacheUndo


@triton.jit
def _snapshot_draft_window(
    PREFIX,
    LENGTH,
    KV,
    KT,
    STATE,
    ST,
    STATE_SLOTS,
    OLD_STATE,
    KV_SLOTS,
    OLD_KV,
    Q: tl.constexpr,
    KTW: tl.constexpr,
    STW: tl.constexpr,
    KB: tl.constexpr,
    SB: tl.constexpr,
    BQ: tl.constexpr,
):
    request = tl.program_id(0)
    step = tl.arange(0, BQ)
    d = tl.arange(0, 128)
    prefix = tl.load(PREFIX + request)
    length = tl.load(LENGTH + request)
    positions = prefix + step
    column = tl.minimum(tl.maximum(positions >> 7, 0), STW - 1)
    state_id = tl.load(ST + request * STW + column, step < Q, other=0).to(tl.int64)
    slot = tl.minimum(tl.maximum(state_id * 8 + (positions & 7), 0), SB * 8 - 1)
    slot = tl.where(step < length, slot, 0)
    tl.store(STATE_SLOTS + request * Q + step, slot, step < Q)
    old = tl.load(
        STATE + slot[:, None] * 128 + d[None, :], step[:, None] < Q, other=0.0
    )
    tl.store(
        OLD_STATE + (request * Q + step[:, None]) * 128 + d[None, :],
        old,
        step[:, None] < Q,
    )
    completion = prefix + 3 - (prefix & 3)
    kv_column = tl.minimum(tl.maximum(completion >> 7, 0), KTW - 1)
    kv_id = tl.load(KT + request * KTW + kv_column).to(tl.int64)
    kv_slot = tl.where(
        completion < prefix + length, kv_id * 32 + ((completion >> 2) & 31), 0
    )
    kv_slot = tl.minimum(tl.maximum(kv_slot, 0), KB * 32 - 1)
    tl.store(KV_SLOTS + request, kv_slot)
    old_kv = tl.load(KV + kv_slot * 128 + d)
    tl.store(OLD_KV + request * 128 + d, old_kv)


@triton.jit
def _write_draft_window(
    RAW,
    CU,
    PREFIX,
    LENGTH,
    COS,
    SIN,
    GAMMA,
    KV,
    KT,
    STATE,
    ST,
    Q: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    KTW: tl.constexpr,
    STW: tl.constexpr,
    KB: tl.constexpr,
    SB: tl.constexpr,
    EPS: tl.constexpr,
    BQ: tl.constexpr,
):
    request = tl.program_id(0)
    d = tl.arange(0, 128)
    prefix = tl.load(PREFIX + request)
    length = tl.load(LENGTH + request)
    completion = prefix + 3 - (prefix & 3)
    begin = tl.load(CU + request)
    if completion < prefix + length:
        group = tl.arange(0, 4)
        source_positions = completion - 3 + group
        source_columns = source_positions >> 7
        source_ids = tl.load(
            ST + request * STW + source_columns,
            (source_positions >= 0) & (source_columns < STW),
            other=0,
        )
        source_valid = (source_ids > 0) & (source_ids < SB)
        source_slots = source_ids * 8 + (source_positions & 7)
        history = tl.load(
            STATE + source_slots[:, None] * 128 + d[None, :],
            source_valid[:, None],
            other=0.0,
        )
        source_step = source_positions - prefix
        # All candidate keys are read directly, before any ring write. This
        # preserves the ordered writer without depending on a global fence.
        current = (source_step >= 0) & (source_step < length)
        candidate = tl.load(
            RAW + (begin + source_step[:, None]) * RAW_STRIDE + d[None, :],
            current[:, None],
            other=0.0,
        ).to(tl.float32)
        history = tl.where(
            (current & (source_valid | (group == 3)))[:, None], candidate, history
        )
        pooled = (tl.sum(history, axis=0) * 0.25).to(tl.bfloat16).to(tl.float32)
        gain = 1.0 + tl.load(GAMMA + d).to(tl.float32)
        rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / 128 + EPS)
        normalized = (pooled * rms * gain).to(tl.bfloat16)
        partner = tl.where(d < 64, tl.where(d < 32, d + 32, d - 32), d)
        rotated = tl.gather(normalized, partner, axis=0)
        rope_offset = (begin + completion - prefix) * 64 + d
        cosine = tl.load(COS + rope_offset, d < 64, other=1.0)
        sine = tl.load(SIN + rope_offset, d < 64, other=0.0)
        left = (normalized.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
        right = (
            rotated.to(tl.float32) * sine.to(tl.float32) * tl.where(d < 32, -1.0, 1.0)
        ).to(tl.bfloat16)
        value = tl.where(
            d < 64,
            (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
            normalized,
        )
        column = completion >> 7
        kv_id = tl.load(
            KT + request * KTW + column, (completion >= 0) & (column < KTW), other=0
        )
        valid = (kv_id > 0) & (kv_id < KB)
        tl.store(KV + (kv_id * 32 + ((completion >> 2) & 31)) * 128 + d, value, valid)
    step = tl.arange(0, BQ)
    positions = prefix + step
    columns = positions >> 7
    ids = tl.load(
        ST + request * STW + columns,
        (step < length) & (positions >= 0) & (columns < STW),
        other=0,
    )
    valid = (step < length) & (ids > 0) & (ids < SB)
    raw = tl.load(
        RAW + (begin + step[:, None]) * RAW_STRIDE + d[None, :],
        step[:, None] < length,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        STATE + (ids[:, None] * 8 + (positions[:, None] & 7)) * 128 + d[None, :],
        raw,
        valid[:, None],
    )


def try_write_draft_window(
    raw, cu, prefixes, lengths, cos, sin, gamma, kv, kt, state, st, *, norm_eps
):
    # Public writer validates all original metadata/layout contracts first.
    if not (
        raw.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(raw.device) == (12, 0)
        and kv.shape[1:] == (32, 128)
        and 1 <= raw.shape[0] <= 32
        and 1 <= prefixes.numel() <= 8
        and raw.stride(1) == 1
        and raw.stride(0) >= 128
    ):
        return None
    batch = prefixes.numel()
    state_slots = torch.empty(batch * 4, dtype=torch.int64, device=raw.device)
    old_state = torch.empty((batch * 4, 128), dtype=state.dtype, device=raw.device)
    kv_slots = torch.empty(batch, dtype=torch.int64, device=raw.device)
    old_kv = torch.empty((batch, 128), dtype=kv.dtype, device=raw.device)
    parameters = dict(
        Q=4,
        KTW=kt.shape[1],
        STW=st.shape[1],
        KB=kv.shape[0],
        SB=state.shape[0],
        BQ=4,
        num_warps=4,
    )
    _snapshot_draft_window[(batch,)](
        prefixes,
        lengths,
        kv,
        kt,
        state,
        st,
        state_slots,
        old_state,
        kv_slots,
        old_kv,
        **parameters
    )
    _write_draft_window[(batch,)](
        raw,
        cu,
        prefixes,
        lengths,
        cos,
        sin,
        gamma,
        kv,
        kt,
        state,
        st,
        RAW_STRIDE=raw.stride(0),
        EPS=norm_eps,
        **parameters
    )
    return IndexerCacheUndo(
        state_pool=state,
        state_slots=state_slots,
        original_state=old_state,
        kv_pool=kv,
        kv_slots=kv_slots,
        original_kv=old_kv,
    )
