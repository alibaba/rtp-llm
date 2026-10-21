"""SM90 paged candidate scorer for the guarded decode-indexer fast path.

Only the dot-product domain is sparse. Full-width FP32 logits preserve the
existing native selector's dispatch, short-row and exact-tie behavior. The
68-byte planar index cache and all BF16 rounding points are unchanged.
"""

from __future__ import annotations

import torch

from ._v41_sm90_indexer import _score
from ._v41_sparse_prefill_indexer import _plan_supported, prepare_plan


def is_supported(
    q, q_scale, pool, weights, table, lengths, candidates, capacity, logical_entries
):
    """Shape-only gate: do not inspect live device contents or hide launch errors."""
    if not (
        isinstance(q, torch.Tensor)
        and q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (9, 0)
        and q.dtype == torch.int8
        and q.ndim == 4
        and q.shape[0] > 0
        and q.shape[1] > 0
        and q.shape[2:] == (32, 64)
        and q.is_contiguous()
        and type(capacity) is int
        and capacity > 0
        and type(logical_entries) is int
    ):
        return False
    b, s = q.shape[:2]
    tensors = (q_scale, pool, weights, table, lengths, candidates)
    if not all(
        isinstance(value, torch.Tensor) and value.device == q.device
        for value in tensors
    ):
        return False
    return (
        q_scale.dtype == torch.int32
        and q_scale.shape == (b, s, 32)
        and q_scale.is_contiguous()
        and pool.dtype == torch.uint8
        and pool.ndim == 3
        and pool.shape[0] > 0
        and pool.shape[1] in (64, 128)
        and pool.shape[2] == 68
        and pool.is_contiguous()
        and logical_entries in (pool.shape[1], pool.shape[1] // 2)
        and weights.dtype in (torch.float32, torch.bfloat16)
        and weights.shape in ((b, s, 32), (b * s, 32))
        and weights.is_contiguous()
        and table.dtype == torch.int32
        and table.ndim == 2
        and table.shape[0] == b
        and table.stride(0) > 0
        and table.stride(1) > 0
        and capacity <= table.shape[1] * logical_entries
        and lengths.ndim == 1
        and lengths.shape[0] == b * s
        and _plan_supported(candidates, lengths, capacity, 8, sm90=True)
    )


def try_score(
    q, q_scale, pool, weights, table, lengths, candidates, *, capacity, logical_entries
):
    """Return full-width masked logits, or None for unsupported input metadata.

    q/q_scale are already RoPE'd and FP4-quantized, shaped [B,S,32,64]/
    [B,S,32]. lengths are logical causal key counts, not physical page extents.
    candidates are request-local block-8 IDs, one row per B*S token. Duplicate
    and out-of-causal-range IDs have set/mask semantics via the existing plan.
    Every call rebuilds device metadata; no graph-live pointer cache is used.
    Nonpositive table entries retain the existing visible zero-key semantics.
    Callers must supply valid positive physical page IDs, as for paged_score.
    """
    if not is_supported(
        q, q_scale, pool, weights, table, lengths, candidates, capacity, logical_entries
    ):
        return None
    plan = prepare_plan(candidates, lengths, capacity, sm90=True)
    if plan is None:
        return None
    return _score(
        q,
        q_scale,
        pool,
        pool,
        weights,
        plan.row_ks,
        plan.row_ke,
        capacity,
        table=table,
        logical_entries=logical_entries,
        candidate_plan=plan,
    )
