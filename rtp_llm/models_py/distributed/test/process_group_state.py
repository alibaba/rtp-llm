from __future__ import annotations

import gc
import os
from datetime import timedelta
from typing import Optional
import logging

import torch
import torch.distributed

from rtp_llm.ops import ParallelismConfig

__all__ = [
    "init_distributed_environment",
    "distributed_environment_initialized",
    "destroy_distributed_environment",
]


def distributed_environment_initialized() -> bool:
    return torch.distributed.is_initialized()


def init_distributed_environment(
    parallelism_config: ParallelismConfig,
    backend: str = "nccl",
    timeout: Optional[int] = None,
):
    assert backend in ["nccl"], "backend current only supports nccl"
    ip = parallelism_config.nccl_ip
    port = parallelism_config.tp_nccl_port
    rank = parallelism_config.dp_rank * parallelism_config.tp_size + parallelism_config.tp_rank
    world_size = parallelism_config.world_size
    local_rank = parallelism_config.local_rank
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
    if not torch.distributed.is_initialized():
        logging.info(
            f"[rank: {rank}] initialize process_group: {ip}:{port}, rank: {rank}, world_size: {world_size}, "
            f"local_rank: {local_rank}, backend: {backend}, timeout: {timeout}"
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
