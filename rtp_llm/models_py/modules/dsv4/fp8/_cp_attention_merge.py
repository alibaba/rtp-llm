"""CP-sharded attention output merge helpers.

When KV is partitioned across CP ranks, each rank can attend the same query
rows against only its local KV shard and produce a local output plus the
softmax log-sum-exp. The exact global attention result is recovered by
logsumexp-merging those partial states, as long as the local KV shards form a
partition of the same key set the full attention would have used.
"""

from __future__ import annotations

from typing import Tuple

import torch


def merge_lse_output(
    local_outs: torch.Tensor,
    local_lse: torch.Tensor,
    dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge partial attention outputs using their per-shard LSE.

    Args:
        local_outs: Partial outputs, usually ``[R, ..., D]``.
        local_lse: Per-shard log-sum-exp, usually ``[R, ...]``.
        dim: Shard/rank dimension shared by both tensors.

    Returns:
        ``(merged_out, merged_lse)`` where ``merged_out`` has ``dim`` removed
        from ``local_outs`` and ``merged_lse`` has ``dim`` removed from
        ``local_lse``.

    The helper accumulates in fp32 and casts the merged output back to
    ``local_outs.dtype``. ``-inf`` LSE rows are treated as empty shards; if all
    shards are empty for a row, the merged output is zero and merged LSE stays
    ``-inf``.
    """
    if local_outs.dim() < 1:
        raise ValueError("local_outs must have at least one dimension")
    if local_lse.dim() < 1:
        raise ValueError("local_lse must have at least one dimension")
    if local_outs.size(dim) != local_lse.size(dim):
        raise ValueError(
            f"rank dimension mismatch: local_outs.size({dim})="
            f"{local_outs.size(dim)} != local_lse.size({dim})={local_lse.size(dim)}"
        )

    dim = dim % local_outs.dim()
    lse_dim = dim % local_lse.dim()
    if dim != lse_dim:
        raise ValueError(
            "dim must refer to the same positive axis in local_outs and local_lse"
        )

    lse_f = local_lse.float()
    out_f = local_outs.float()
    merged_lse = torch.logsumexp(lse_f, dim=dim)

    weights = torch.exp(lse_f - merged_lse.unsqueeze(dim))
    # All-empty rows produce exp(-inf - -inf) = NaN. They should contribute
    # zero output while preserving merged_lse = -inf.
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    while weights.dim() < out_f.dim():
        weights = weights.unsqueeze(-1)

    merged_out = torch.sum(weights * out_f, dim=dim)
    return merged_out.to(dtype=local_outs.dtype), merged_lse
