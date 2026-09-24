"""K3 sequence-parallel reduction with a fixed BF16 accumulation order."""

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    reduce_scatter as default_reduce_scatter,
)


def reduce_scatter(input_tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Sum BF16 TP contributions in rank order, then return the local token shard.

    Exchange contributions without arithmetic, then add in rank order with
    BF16 rounding after each addition, matching the native K3 reference.
    Keeping the order independent of token count and output owner also makes
    cached-prefix reuse invariant to a change in SP ownership. FP32
    accumulation changes K3's numerical behavior. Other dtypes retain RTP's
    existing reduction implementation.
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
    accumulated = contributions[0]
    for rank in range(1, world_size):
        accumulated = accumulated + contributions[rank]
    return accumulated
