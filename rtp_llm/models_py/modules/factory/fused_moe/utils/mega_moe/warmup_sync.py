"""CUDA-graph warmup helpers for distributed MoE executors."""

from __future__ import annotations

import torch

from rtp_llm.models_py.utils.cuda_graph_warmup import (
    cuda_graph_warmup_forward_enabled,
)


def sync_cuda_graph_warmup_ranks(
    _phase: str, device: torch.device | None = None
) -> None:
    if not cuda_graph_warmup_forward_enabled():
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return

    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()

    from rtp_llm.models_py.distributed import collective_torch

    collective_torch.barrier(collective_torch.Group.STAGE)
