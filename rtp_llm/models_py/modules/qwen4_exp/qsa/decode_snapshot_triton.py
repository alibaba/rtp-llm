"""Snapshot ordinary decode destinations before the original cache writer.

Fuse the integer metadata and gathers as in the target snapshot path. A
separate launch finishes every read before any request mutates a cache row,
including reserved sentinel rows shared by inactive completion slots.
"""

import torch
import triton
import triton.language as tl

from .indexer_compressor import IndexerCacheUndo


@triton.jit
def _snapshot_decode(
    POS,
    KV,
    KT,
    STATE,
    ST,
    SS,
    OS,
    KS,
    OK,
    KTW: tl.constexpr,
    STW: tl.constexpr,
    KB: tl.constexpr,
    SB: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 128)
    # Match the reference's int64 conversion before arithmetic, even for
    # malformed int32 positions inspected by standalone validation.
    pos = tl.load(POS + row).to(tl.int64)
    sc = tl.minimum(tl.maximum(pos >> 7, 0), STW - 1)
    kc = tl.minimum(tl.maximum(pos >> 7, 0), KTW - 1)
    sid = tl.load(ST + row * STW + sc).to(tl.int64)
    kid = tl.load(KT + row * KTW + kc).to(tl.int64)
    ss = tl.minimum(tl.maximum(sid * 8 + (pos & 7), 0), SB * 8 - 1)
    ks = tl.where(((pos + 1) & 3) == 0, kid * 32 + ((pos >> 2) & 31), 0)
    ks = tl.minimum(tl.maximum(ks, 0), KB * 32 - 1)
    tl.store(SS + row, ss)
    tl.store(KS + row, ks)
    old_state = tl.load(STATE + ss * 128 + d)
    old_kv = tl.load(KV + ks * 128 + d)
    tl.store(OS + row * 128 + d, old_state)
    tl.store(OK + row * 128 + d, old_kv)


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
    )
