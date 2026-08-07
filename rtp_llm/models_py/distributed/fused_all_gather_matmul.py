"""Thin wrappers for PyTorch symmetric-memory AllGather/GEMM fusion."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as torch_symm_mem


def reserve_fused_all_gather_matmul_workspace(
    group: dist.ProcessGroup,
    min_size_bytes: int,
) -> None:
    """Allocate and rendezvous the process-lifetime P2P workspace."""

    torch_symm_mem.get_symm_mem_workspace(
        group.group_name,
        min_size=min_size_bytes,
    )


def fused_all_gather_matmul(
    local_a: torch.Tensor,
    weights: Sequence[torch.Tensor],
    group: dist.ProcessGroup,
    *,
    return_gathered: bool,
) -> tuple[Optional[torch.Tensor], list[torch.Tensor]]:
    """Execute PyTorch's dim-0 symmetric-memory AllGather/GEMM operator."""

    gathered, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
        local_a,
        list(weights),
        0,
        group.group_name,
        return_A=return_gathered,
    )
    return gathered, list(outputs)
