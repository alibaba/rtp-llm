"""Tensor-parallel normalization helpers for hidden-dimension shards."""

from __future__ import annotations
import torch
import torch.nn as nn


def tp_rms_norm(
    norm: nn.Module,
    x: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    global_dim: int | None = None,
) -> torch.Tensor:
    """Apply RMSNorm to one contiguous hidden-dimension TP shard."""

    tp_size = int(tp_size)
    tp_rank = int(tp_rank)
    local_dim = int(x.shape[-1])
    weight = norm.weight.data
    weight_numel = int(weight.numel())
    declared_dim = None if global_dim is None else int(global_dim)
    if declared_dim is not None and declared_dim < local_dim:
        raise ValueError(
            f"RMSNorm local hidden size={local_dim} exceeds declared global "
            f"size={declared_dim}"
        )
    if tp_size == 1:
        return norm(x)
    if declared_dim is None and weight_numel == local_dim:
        # Legacy call sites do not declare the global width, so a local-width
        # weight is indistinguishable from a replicated one.
        return norm(x)
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid RMSNorm TP geometry: size={tp_size}, rank={tp_rank}")

    global_dim = declared_dim if declared_dim is not None else local_dim * tp_size
    if local_dim == global_dim and weight_numel == global_dim:
        return norm(x)
    if weight_numel == global_dim:
        start = tp_rank * local_dim
        weight_shard = weight[start : start + local_dim]
    elif weight_numel == local_dim:
        # Both the activation and the affine weight are TP-sharded.
        weight_shard = weight
    else:
        raise ValueError(
            f"RMSNorm weight size={weight_numel} does not match global hidden "
            f"size={global_dim} or local hidden size={local_dim}"
        )

    square_sum = x.float().square().sum(-1, keepdim=True)
    from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

    square_sum = all_reduce(square_sum, Group.TP)
    rsqrt = torch.rsqrt(square_sum / global_dim + norm.variance_epsilon)
    return (x.float() * rsqrt * weight_shard.float()).to(x.dtype)


def tp_gather_hidden(x: torch.Tensor, *, tp_size: int) -> torch.Tensor:
    """Gather contiguous hidden shards and concatenate them on the last axis."""

    tp_size = int(tp_size)
    if tp_size == 1:
        return x
    if tp_size <= 0:
        raise ValueError(f"invalid hidden gather tp_size={tp_size}")
    from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

    local_dim = x.shape[-1]
    leading = x.shape[:-1]
    shard_major = x.reshape(-1, local_dim).transpose(0, 1).contiguous()
    gathered = all_gather(shard_major, Group.TP)
    return gathered.transpose(0, 1).reshape(*leading, local_dim * tp_size)
