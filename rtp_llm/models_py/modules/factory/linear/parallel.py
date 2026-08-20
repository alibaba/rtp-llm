"""Tensor- and sequence-parallel linear projection orchestration."""

from __future__ import annotations

from typing import Dict

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    all_reduce,
    reduce_scatter,
    reduce_scatter_padded,
)
from rtp_llm.models_py.distributed.sequence_parallel import (
    TokenShardLayout,
    shard_tokens,
    shard_tokens_with_padding,
    token_shard_layout,
)


def _replicate_column_weight(
    local_weight: torch.Tensor,
    group: Group,
) -> torch.Tensor:
    return (
        all_gather(local_weight.transpose(0, 1).contiguous(), group=group)
        .transpose(0, 1)
        .contiguous()
    )


def _replicate_row_weight(
    local_weight: torch.Tensor,
    group: Group,
) -> torch.Tensor:
    return all_gather(local_weight.contiguous(), group=group)


def sequence_parallel_column_weight(
    weights: Dict[str, torch.Tensor],
    weight_name: str,
    world_size: int,
    rank: int,
    cache: dict[str, torch.Tensor],
    cache_key: str,
    *,
    sequence_parallel: bool,
    group: Group = Group.TP,
) -> torch.Tensor:
    """Select a local column shard or lazily materialize its full SP weight."""

    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid parallel layout: world_size={world_size}, rank={rank}"
        )
    if world_size <= 1:
        return weights[weight_name]
    full_weight = cache.get(cache_key)
    if sequence_parallel:
        if full_weight is None:
            full_weight = _replicate_column_weight(weights[weight_name], group)
            cache[cache_key] = full_weight
            weights[weight_name] = full_weight
        return full_weight
    if full_weight is None:
        return weights[weight_name]
    if full_weight.shape[1] % world_size:
        raise ValueError(
            "column-parallel weight width must divide world_size: "
            f"shape={tuple(full_weight.shape)}, world_size={world_size}"
        )
    local_width = full_weight.shape[1] // world_size
    begin = rank * local_width
    return full_weight[:, begin : begin + local_width]


def sequence_parallel_row_weight(
    weights: Dict[str, torch.Tensor],
    weight_name: str,
    world_size: int,
    rank: int,
    cache: dict[str, torch.Tensor],
    cache_key: str,
    *,
    sequence_parallel: bool,
    group: Group = Group.TP,
) -> torch.Tensor:
    """Select a local row shard or lazily materialize its full SP weight."""

    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(
            f"invalid parallel layout: world_size={world_size}, rank={rank}"
        )
    if world_size <= 1:
        return weights[weight_name]
    full_weight = cache.get(cache_key)
    if sequence_parallel:
        if full_weight is None:
            full_weight = _replicate_row_weight(weights[weight_name], group)
            cache[cache_key] = full_weight
            weights[weight_name] = full_weight
        return full_weight
    if full_weight is None:
        return weights[weight_name]
    if full_weight.shape[0] % world_size:
        raise ValueError(
            "row-parallel weight height must divide world_size: "
            f"shape={tuple(full_weight.shape)}, world_size={world_size}"
        )
    local_height = full_weight.shape[0] // world_size
    begin = rank * local_height
    return full_weight[begin : begin + local_height]


def row_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    world_size: int,
    *,
    reduce_scatter_tokens: bool = False,
    pad_reduce_scatter_tokens: bool = False,
    use_input_dtype_reduce_scatter: bool = False,
    group: Group = Group.TP,
) -> torch.Tensor:
    """Apply a row-parallel projection followed by all-reduce or token RS."""

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if pad_reduce_scatter_tokens and not reduce_scatter_tokens:
        raise ValueError(
            "pad_reduce_scatter_tokens requires reduce_scatter_tokens=True"
        )
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"linear input width {x.shape[-1]} does not match weight "
            f"shape {tuple(weight.shape)}"
        )
    if world_size <= 1:
        return torch.matmul(x, weight)
    if (
        x.is_cuda
        and x.ndim == 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
    ):
        if reduce_scatter_tokens and use_input_dtype_reduce_scatter:
            partial = (
                _matmul_with_padded_rows(x, weight, world_size, x.dtype)
                if pad_reduce_scatter_tokens
                else torch.mm(x, weight)
            )
            return reduce_scatter(partial, group=group)
        output = torch.mm(x, weight, out_dtype=torch.float32)
        if reduce_scatter_tokens:
            output = (
                reduce_scatter_padded(output, group=group)
                if pad_reduce_scatter_tokens
                else reduce_scatter(output, group=group)
            )
        else:
            output = all_reduce(output, group=group)
        return output.to(dtype=x.dtype)
    output = torch.matmul(x, weight)
    if reduce_scatter_tokens:
        return (
            reduce_scatter_padded(output, group=group)
            if pad_reduce_scatter_tokens
            else reduce_scatter(output, group=group)
        )
    return all_reduce(output, group=group)


def _matmul_with_padded_rows(
    x: torch.Tensor,
    weight: torch.Tensor,
    world_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    padded_rows = ((int(x.shape[0]) + world_size - 1) // world_size) * world_size
    output = torch.empty(
        (padded_rows, weight.shape[1]),
        dtype=output_dtype,
        device=x.device,
    )
    valid_output = output.narrow(0, 0, x.shape[0])
    if output_dtype == x.dtype:
        torch.mm(x, weight, out=valid_output)
    else:
        torch.mm(x, weight, out_dtype=output_dtype, out=valid_output)
    if padded_rows != x.shape[0]:
        output.narrow(0, x.shape[0], padded_rows - x.shape[0]).zero_()
    return output


__all__ = [
    "TokenShardLayout",
    "row_parallel_linear",
    "sequence_parallel_column_weight",
    "sequence_parallel_row_weight",
    "shard_tokens",
    "shard_tokens_with_padding",
    "token_shard_layout",
]
