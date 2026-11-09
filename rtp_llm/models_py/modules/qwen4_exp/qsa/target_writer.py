"""Bounded QSA target writes with a complete snapshot in a preceding launch."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qsa.target_writer import (
    _snapshot_target_window,
    _write_target_window,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import IndexerCacheUndo


def try_write_target_window(
    raw, prefixes, cos, sin, gamma, kv, kt, state, st, *, norm_eps
):
    """Called only after the existing writer's full layout validation."""
    if (
        torch.cuda.get_device_capability(raw.device) != (12, 0)
        or min(kt.shape[1], st.shape[1], kv.shape[0], state.shape[0]) <= 0
        or raw.stride(1) != 1
        or raw.stride(0) <= 0
        or not cos.is_contiguous()
        or not sin.is_contiguous()
        or any(t.requires_grad for t in (raw, cos, sin, gamma, kv, state))
    ):
        return None
    batch = prefixes.numel()
    query = raw.shape[0] // batch
    state_slots = torch.empty(batch * query, dtype=torch.int64, device=raw.device)
    old_state = torch.empty((batch * query, 128), dtype=state.dtype, device=raw.device)
    kv_slots = torch.empty(batch, dtype=torch.int64, device=raw.device)
    old_kv = torch.empty((batch, 128), dtype=kv.dtype, device=raw.device)
    parameters = dict(
        Q=query,
        KTW=kt.shape[1],
        STW=st.shape[1],
        KB=kv.shape[0],
        SB=state.shape[0],
        BQ=triton.next_power_of_2(query),
        num_warps=4,
    )
    # A separate launch preserves snapshot-before-write even across requests.
    _snapshot_target_window[(batch,)](
        prefixes,
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
    _write_target_window[(batch,)](
        raw,
        prefixes,
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
        padded_slots=True,
    )
