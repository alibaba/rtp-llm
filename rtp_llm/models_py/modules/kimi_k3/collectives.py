"""K3 SP reductions use BF16 NCCL for every input size."""

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    reduce_scatter as default_reduce_scatter,
)

def reduce_scatter(input_tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Keep BF16 communication without size-dependent widening or extra casts.

    Other dtypes retain RTP's existing collective behavior.
    """
    if input_tensor.dtype != torch.bfloat16:
        return default_reduce_scatter(input_tensor, group)
    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)
    if input_tensor.ndim == 0 or input_tensor.shape[0] % world_size:
        raise ValueError("K3 SP token dimension must be divisible by the TP size")
    if world_size == 1:
        return input_tensor
    if input_tensor.numel() == 0:
        return input_tensor.new_empty(
            (input_tensor.shape[0] // world_size, *input_tensor.shape[1:])
        )
    source = input_tensor.contiguous()
    return default_reduce_scatter(source, group)
