"""Ragged draft variant of the qualified two-launch snapshot/write contract."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qsa.draft_writer import (
    _snapshot_draft_window,
    _write_draft_window,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import IndexerCacheUndo


def try_write_draft_window(
    raw, cu, prefixes, lengths, cos, sin, gamma, kv, kt, state, st, *, norm_eps
):
    # Public writer validates all original metadata/layout contracts first.
    if not (
        raw.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(raw.device) == (12, 0)
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
        **parameters,
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
        **parameters,
    )
    return IndexerCacheUndo(
        state_pool=state,
        state_slots=state_slots,
        original_state=old_state,
        kv_pool=kv,
        kv_slots=kv_slots,
        original_kv=old_kv,
    )
