from __future__ import annotations

import gc
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters

__all__ = [
    "init_distributed_environment",
    "distributed_environment_initialized",
    "destroy_distributed_environment",
]


def distributed_environment_initialized() -> bool:
    return torch.distributed.is_initialized()


def init_distributed_environment(
    params: GptInitModelParameters,
    backend: str = "nccl",
    timeout: Optional[int] = None,
):
    assert backend in ["nccl"], "backend current only supports nccl"
    ip = params.nccl_ip
    port = params.th_nccl_port
    rank = params.dp_rank * params.tp_size + params.tp_rank
    world_size = params.world_size
    local_rank = params.local_rank
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
    if not torch.distributed.is_initialized():
        print(
            f"[rank: {rank}] initialize process_group: {ip}:{port}, rank: {rank}, world_size: {world_size}, "
            f"local_rank: {local_rank}, backend: {backend}, timeout: {timeout}",
            flush=True,
        )
        if timeout is not None:
            assert isinstance(timeout, (int)), "timeout must be a number"
            assert timeout > 0, "timeout must be positive"
            timeout = timedelta(
                seconds=timeout
            )  # pyright: ignore[reportAssignmentType]
        torch.distributed.init_process_group(
            backend=backend,
            init_method=f"tcp://{ip}:{port}",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timeout,  # pyright: ignore[reportArgumentType]
        )


def destroy_distributed_environment():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    gc.collect()
