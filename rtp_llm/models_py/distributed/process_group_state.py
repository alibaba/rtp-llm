from __future__ import annotations

import gc
import os
import weakref
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters

__all__ = [
    "ProcessGroupState",
    "get_ep_group",
    "init_distributed_environment",
    "destroy_distributed_environment",
]


@dataclass
class ProcessGroupState:
    group_name: str  # name of the group
    rank: int  # global rank
    world_size: int  # size of the world
    ranks: List[int]  # global ranks in the group
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    group_size: int  # size of the group
    device_group: Optional[ProcessGroup]  # group for device communication
    device: Optional[torch.device]  # device used to assign devices


_group_name_counter: Dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: Dict[str, Callable[[], Optional[ProcessGroupState]]] = {}


def _register_group(group: ProcessGroupState) -> None:
    _groups[group.group_name] = weakref.ref(group)


def _init_process_group_state(
    group_ranks: List[List[int]],
    local_rank: int,
    torch_distributed_backend: Union[str, Backend],
    group_name: Optional[str] = None,
) -> ProcessGroupState:
    group_name = group_name or "anonymous"
    unique_name = _get_unique_name(group_name)
    rank = torch.distributed.get_rank()
    device_group = None
    world_size = None
    current_ranks = None
    group_size = None
    rank_in_group = None

    for ranks in group_ranks:
        # create a new group for the ranks
        device_group = torch.distributed.new_group(
            ranks,
            backend=torch_distributed_backend,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        if rank in ranks:
            # set the world size, ranks, group size, rank in group
            world_size = torch.distributed.get_world_size()
            current_ranks = ranks
            group_size = len(ranks)
            rank_in_group = ranks.index(rank)

    assert (
        device_group is not None
        and world_size is not None
        and current_ranks is not None
        and group_size is not None
        and rank_in_group is not None
    ), "device_group, world_size, current_ranks, group_size, rank_in_group are not initialized"
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        raise RuntimeError("GPU is not available")

    process_group_state = ProcessGroupState(
        group_name=unique_name,
        rank=rank,
        world_size=world_size,
        ranks=current_ranks,
        local_rank=local_rank,
        rank_in_group=rank_in_group,
        group_size=group_size,
        device_group=device_group,  # pyright: ignore[reportUnknownArgumentType]
        device=device,
    )
    _register_group(process_group_state)
    return process_group_state


def _destroy_process_group_state(process_group_state: ProcessGroupState) -> None:
    if process_group_state.device_group is not None:
        torch.distributed.destroy_process_group(process_group_state.device_group)
        process_group_state.device_group = None
    process_group_state.device = None


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
        initialize_expert_parallel(params, backend)


_EP: Optional[ProcessGroupState] = None


def get_ep_group() -> ProcessGroupState:
    assert _EP is not None, "expert parallel group is not initialized"
    return _EP


def initialize_expert_parallel(
    params: GptInitModelParameters,
    backend: str,
) -> None:
    """
    Initialize expert parallel groups.
    """
    global _EP
    assert _EP is None, "expert parallel group is already initialized"
    group_ranks = list(range(params.ep_size))

    _EP = _init_process_group_state(  # pyright: ignore[reportConstantRedefinition]
        group_ranks=[group_ranks],
        local_rank=params.local_rank,
        torch_distributed_backend=backend,
        group_name="ep",
    )


def destroy_ep_parallel():
    """Set the groups to none and destroy them."""
    global _EP
    if _EP:
        _destroy_process_group_state(_EP)
    _EP = None  # pyright: ignore[reportConstantRedefinition]


def destroy_distributed_environment():
    destroy_ep_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    gc.collect()
