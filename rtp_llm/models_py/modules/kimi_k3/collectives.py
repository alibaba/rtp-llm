"""K3 sequence-parallel reduction with a fixed FP32 accumulation order."""

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    reduce_scatter as default_reduce_scatter,
)


def reduce_scatter(input_tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Sum BF16 TP contributions in rank order, then return the local token shard.

    NCCL reduction order can depend on the physical token shape and output
    owner. Even FP32 accumulation can then round identical BF16 contributions
    differently at a BF16 midpoint when a cached prefix changes SP ownership.
    Exchange contributions without arithmetic and sum in a fixed rank order.
    Other dtypes retain RTP's existing reduction implementation.
    """
    if input_tensor.dtype != torch.bfloat16:
        return default_reduce_scatter(input_tensor, group)
    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)
    if input_tensor.ndim == 0 or input_tensor.shape[0] % world_size:
        raise ValueError("K3 SP token dimension must be divisible by the TP size")
    if world_size == 1:
        return input_tensor
    chunk_size = input_tensor.shape[0] // world_size
    shape = (chunk_size, *input_tensor.shape[1:])
    if input_tensor.numel() == 0:
        return input_tensor.new_empty(shape)
    source = input_tensor.contiguous()
    received = torch.empty_like(source)
    torch.distributed.all_to_all_single(received, source, group=process_group)
    contributions = received.view(world_size, *shape)
    accumulated = contributions[0].float()
    for rank in range(1, world_size):
        accumulated = accumulated + contributions[rank].float()
    return accumulated.to(input_tensor.dtype)
