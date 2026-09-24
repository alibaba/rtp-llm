"""K3 SP reduction precision matching the pinned native deployment."""

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    reduce_scatter as default_reduce_scatter,
)

# vLLM c3b48446349569512749db7f6e2164aa8a33437d, CustomAllreduce:
# MNNVL accumulates small reductions in FP32; larger inputs use BF16 NCCL.
_NATIVE_MNNVL_REDUCE_SCATTER_MAX_BYTES = 16 * 1024 * 1024


def reduce_scatter(input_tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Preserve native K3 accumulation precision through RTP's collective.

    BF16 rank-ordered manual sums differ from both native paths: MNNVL uses
    FP32 accumulation, while NCCL's BF16 reduction order depends on the owner.
    Keep this policy local to K3; other dtypes and models retain RTP's behavior.
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
    if source.nbytes <= _NATIVE_MNNVL_REDUCE_SCATTER_MAX_BYTES:
        return default_reduce_scatter(source.float(), group).to(input_tensor.dtype)
    return default_reduce_scatter(source, group)
