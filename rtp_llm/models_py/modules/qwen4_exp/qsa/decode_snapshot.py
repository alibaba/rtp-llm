"""Snapshot ordinary decode destinations before the original cache writer.

Fuse the integer metadata and gathers as in the target snapshot path. A
separate launch finishes every read before any request mutates a cache row,
including reserved sentinel rows shared by inactive completion slots.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qsa.decode_snapshot import _snapshot_decode
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import IndexerCacheUndo


def try_snapshot_decode(positions, kv, kt, state, st):
    """Return None without writes outside the explicit bounded SM120 path."""
    if not (
        positions.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(positions.device) == (12, 0)
        and positions.ndim == 1
        and 1 <= positions.numel() <= 32
        and positions.dtype == torch.int32
        and kv.ndim == state.ndim == 3
        and kv.shape[1:] == (32, 128)
        and state.shape[1:] == (8, 128)
        and kv.dtype == torch.bfloat16
        and state.dtype == torch.float32
        and kt.ndim == st.ndim == 2
        and kt.shape[0] == st.shape[0] == positions.numel()
        and kt.dtype == st.dtype == torch.int32
        and min(kv.shape[0], state.shape[0], kt.shape[1], st.shape[1]) > 0
        and all(
            t.device == positions.device and t.is_contiguous()
            for t in (positions, kv, kt, state, st)
        )
        and not (kv.requires_grad or state.requires_grad)
    ):
        return None
    batch = positions.numel()
    ss = torch.empty(batch, device=positions.device, dtype=torch.int64)
    ks = torch.empty_like(ss)
    old_state = torch.empty((batch, 128), device=positions.device, dtype=state.dtype)
    old_kv = torch.empty((batch, 128), device=positions.device, dtype=kv.dtype)
    _snapshot_decode[(batch,)](
        positions,
        kv,
        kt,
        state,
        st,
        ss,
        old_state,
        ks,
        old_kv,
        kt.shape[1],
        st.shape[1],
        kv.shape[0],
        state.shape[0],
        num_warps=4,
    )
    return IndexerCacheUndo(
        state_pool=state,
        state_slots=ss,
        original_state=old_state,
        kv_pool=kv,
        kv_slots=ks,
        original_kv=old_kv,
        padded_slots=True,
    )
