from __future__ import annotations

import ctypes
import gc
import logging
import os
from datetime import timedelta
from enum import Enum
from typing import Dict, Optional, Union

import torch
import torch.distributed

from rtp_llm.models_py.distributed.symm_mem import (
    get_symm_mem_communicator,
    init_symm_mem_communicator,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig


class Group(Enum):
    """Process group types for collective operations"""

    DP = "DP"
    TP = "TP"
    DP_AND_TP = "DP_AND_TP"


# Global process group storage
# Key can be Group enum or string (for multiple DP/TP groups)
_group_map: Dict[Union[Group, str], torch.distributed.ProcessGroup] = {}
_parallelism_config: Optional[ParallelismConfig] = None
_initialized: bool = False  # Track if we've initialized (to prevent double init)

# ---------------------------------------------------------------------------
# Direct RCCL ctypes wrapper for HIP Graph capture mode.
#
# During HIP Graph capture, torch.distributed's ProcessGroupNCCL watchdog
# thread queries HIP events recorded on the capturing stream via
# hipEventQuery(), which is illegal and crashes.  By calling RCCL directly
# with the existing C++ communicator handle we avoid creating any WorkNCCL
# objects or events that the watchdog can see.
# ---------------------------------------------------------------------------

# RCCL constants
_NCCL_SUCCESS = 0
_NCCL_SUM = 0

# ncclDataType_t
_NCCL_DTYPE_MAP = {
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.bfloat16: 9,
}

_rccl_lib: Optional[ctypes.CDLL] = None


def _get_rccl_lib() -> Optional[ctypes.CDLL]:
    """Lazy-load librccl / libnccl shared library."""
    global _rccl_lib
    if _rccl_lib is not None:
        return _rccl_lib
    for name in ["librccl.so.1", "librccl.so", "libnccl.so.2", "libnccl.so"]:
        try:
            _rccl_lib = ctypes.CDLL(name)
            logging.info(f"Loaded RCCL library: {name}")
            return _rccl_lib
        except OSError:
            continue
    logging.warning("Failed to load RCCL/NCCL library.")
    return None


# Graph capture mode state
_in_graph_capture: bool = False
_rccl_comm: Optional[ctypes.c_void_p] = None  # ncclComm_t handle from C++
_rccl_world_size: int = 1
_rccl_rank: int = 0


def init_distributed_environment(
    parallelism_config: ParallelismConfig,
    nccl_comm_config: NcclCommConfig,
    nccl_init_port: int,
    backend: str = "nccl",
    timeout: Optional[int] = None,
):
    """Initialize distributed environment and create process groups.

    This function creates DP, TP, and DP_AND_TP process groups using torch.distributed.
    It can only be called once unless destroy_distributed_environment() has been called.

    Args:
        parallelism_config: Configuration for parallelism setup (sizes, ranks, etc.)
        nccl_comm_config: NCCL config with nccl_ip (and other ports for C++ init).
        nccl_init_port: Port for torch.distributed init_process_group (tcp://ip:port).
        backend: Distributed backend (default: "nccl")
        timeout: Timeout in seconds for process group initialization

    Raises:
        RuntimeError: If already initialized and not destroyed
    """
    global _group_map, _parallelism_config, _initialized

    # Check if already initialized (and not destroyed)
    if _initialized and torch.distributed.is_initialized():
        logging.warning(
            "Distributed environment already initialized, skipping initialization"
        )
        # Still need to create groups if they don't exist
        if not _group_map:
            _create_process_groups(
                parallelism_config, backend, timedelta(seconds=timeout)
            )
        return

    assert backend in ["nccl"], "backend current only supports nccl"
    ip = nccl_comm_config.nccl_ip
    port = nccl_init_port
    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    local_rank = parallelism_config.local_rank

    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"

    # If torch.distributed is already initialized (e.g., by external code),
    # we still need to create our process groups
    if torch.distributed.is_initialized():
        logging.info("torch.distributed already initialized, creating process groups")
        _create_process_groups(parallelism_config, backend, timedelta(seconds=timeout))
        _parallelism_config = parallelism_config
        _initialized = True
        return

    logging.info(
        f"[rank: {world_rank}] initialize process_group: {ip}:{port}, rank: {world_rank}, world_size: {world_size}, "
        f"local_rank: {local_rank}, backend: {backend}, timeout: {timeout}",
    )

    if timeout is not None:
        assert isinstance(timeout, (int)), "timeout must be a number"
        assert timeout > 0, "timeout must be positive"
        timeout = timedelta(seconds=timeout)  # pyright: ignore[reportAssignmentType]

    # DP_AND_TP (global group) - initialized via init_process_group
    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=world_rank,
        # device_id=torch.device(f"cuda:{local_rank}"), # https://github.com/pytorch/pytorch/pull/149144
        timeout=timeout,  # pyright: ignore[reportArgumentType]
    )
    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    _group_map[Group.DP_AND_TP] = torch.distributed.group.WORLD
    logging.info(
        f"[rank: {world_rank}] Created DP_AND_TP group {torch.distributed.group.WORLD} with ranks: {list(range(world_size))}"
    )

    # Create DP and TP groups
    _create_process_groups(parallelism_config, backend, timeout)

    _parallelism_config = parallelism_config
    _initialized = True


def _create_process_groups(
    parallelism_config: ParallelismConfig,
    backend: str,
    timeout: Optional[timedelta],
):
    """Create DP and TP process groups.

    Args:
        parallelism_config: Configuration for parallelism setup
        backend: Distributed backend
        timeout: Timeout for process group creation
    """
    global _group_map

    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    tp_size = parallelism_config.tp_size
    dp_size = parallelism_config.dp_size

    if dp_size > 1 and world_size != dp_size:
        # Create all DP groups - all ranks must participate in creating all DP groups
        # DP group: ranks with the same tp_rank (i.e., world_rank % tp_size)
        # There are tp_size DP groups (one for each tp_rank value)
        for tp_rank_val in range(tp_size):
            dp_ranks = [r for r in range(world_size) if r % tp_size == tp_rank_val]
            if len(dp_ranks) > 0:
                logging.info(
                    f"[rank: {world_rank}] Creating DP group for tp_rank {tp_rank_val} with ranks: {dp_ranks}"
                )
                dp_group = torch.distributed.new_group(
                    ranks=dp_ranks,
                    backend=backend,
                    timeout=timeout,  # pyright: ignore[reportArgumentType]
                )
                # Only store the group if this rank is part of it
                if world_rank in dp_ranks:
                    group_key = Group.DP.name + str(tp_rank_val)
                    _group_map[group_key] = dp_group
                    logging.info(
                        f"[rank: {world_rank}] Stored DP group with key: {group_key} {dp_group} with ranks: {dp_ranks}"
                    )
                # All ranks must wait for group creation to complete
                torch.distributed.barrier()

    if tp_size > 1 and world_size != tp_size:
        # Create all TP groups - all ranks must participate in creating all TP groups
        # TP group: ranks with the same dp_rank (i.e., world_rank // tp_size)
        # There are dp_size TP groups (one for each dp_rank value)
        for dp_rank_val in range(dp_size):
            tp_ranks = [r for r in range(world_size) if r // tp_size == dp_rank_val]
            if len(tp_ranks) > 0:
                logging.info(
                    f"[rank: {world_rank}] Creating TP group for dp_rank {dp_rank_val} with ranks: {tp_ranks}"
                )
                tp_group = torch.distributed.new_group(
                    ranks=tp_ranks,
                    backend=backend,
                    timeout=timeout,  # pyright: ignore[reportArgumentType]
                )
                # Only store the group if this rank is part of it
                if world_rank in tp_ranks:
                    group_key = Group.TP.name + str(dp_rank_val)
                    _group_map[group_key] = tp_group
                    logging.info(
                        f"[rank: {world_rank}] Stored TP group with key: {group_key} {tp_group} with ranks: {tp_ranks}"
                    )

                init_symm_mem_communicator(tp_group)

                # All ranks must wait for group creation to complete
                torch.distributed.barrier()
    elif tp_size > 1 and world_size == tp_size:
        # Single TP group: WORLD is the TP group, init symm_mem for it
        init_symm_mem_communicator(torch.distributed.group.WORLD)


def _setup_rccl_signatures(lib: ctypes.CDLL) -> None:
    """Set up ctypes function signatures for ncclAllReduce / ncclAllGather."""
    lib.ncclAllReduce.restype = ctypes.c_int
    lib.ncclAllReduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.ncclAllGather.restype = ctypes.c_int
    lib.ncclAllGather.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]


def _rccl_all_reduce(tensor: torch.Tensor) -> None:
    """In-place all-reduce via direct RCCL call on the current HIP stream."""
    lib = _get_rccl_lib()
    assert lib is not None and _rccl_comm is not None
    nccl_dtype = _NCCL_DTYPE_MAP[tensor.dtype]
    stream = torch.cuda.current_stream().cuda_stream
    result = lib.ncclAllReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        nccl_dtype,
        _NCCL_SUM,
        _rccl_comm,
        stream,
    )
    if result != _NCCL_SUCCESS:
        raise RuntimeError(f"ncclAllReduce failed with error code {result}")


def _rccl_all_gather(output: torch.Tensor, input: torch.Tensor) -> None:
    """All-gather via direct RCCL call on the current HIP stream."""
    lib = _get_rccl_lib()
    assert lib is not None and _rccl_comm is not None
    nccl_dtype = _NCCL_DTYPE_MAP[input.dtype]
    stream = torch.cuda.current_stream().cuda_stream
    result = lib.ncclAllGather(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        nccl_dtype,
        _rccl_comm,
        stream,
    )
    if result != _NCCL_SUCCESS:
        raise RuntimeError(f"ncclAllGather failed with error code {result}")


def enter_graph_capture_mode(nccl_comm_handle: int, world_size: int, rank: int) -> None:
    """Enter HIP Graph capture mode.

    Stores the existing C++ ncclComm_t handle so that subsequent all_reduce /
    all_gather calls go through direct RCCL ctypes instead of torch.distributed.

    Called from C++ HipGraphRunner before graph.capture_begin().

    Args:
        nccl_comm_handle: The ncclComm_t pointer from C++ ROCmDevice (as int64)
        world_size: TP world size
        rank: TP rank
    """
    global _in_graph_capture, _rccl_comm, _rccl_world_size, _rccl_rank

    lib = _get_rccl_lib()
    if lib is None:
        logging.warning("enter_graph_capture_mode: RCCL library not available")
        return

    _setup_rccl_signatures(lib)
    _rccl_comm = ctypes.c_void_p(nccl_comm_handle)
    _rccl_world_size = world_size
    _rccl_rank = rank
    _in_graph_capture = True
    logging.info(
        f"Entered HIP Graph capture mode - using C++ NCCL comm "
        f"(rank={rank}, world_size={world_size}, handle=0x{nccl_comm_handle:x})"
    )


def exit_graph_capture_mode() -> None:
    """Exit HIP Graph capture mode.

    Restores collective operations to use torch.distributed.
    Called from C++ HipGraphRunner after graph.capture_end().
    """
    global _in_graph_capture
    _in_graph_capture = False
    logging.info(
        "Exited HIP Graph capture mode - collective ops restored to torch.distributed"
    )


def distributed_environment_initialized() -> bool:
    """Check if distributed environment is initialized.

    Returns:
        True if distributed environment is initialized, False otherwise
    """
    return torch.distributed.is_initialized()


def destroy_distributed_environment():
    """Destroy distributed environment and clean up process groups.

    After calling this function, init_distributed_environment() can be called again
    to reinitialize the distributed environment.
    """
    global _group_map, _parallelism_config, _initialized

    rank = torch.distributed.get_rank()
    logging.info(f"[rank: {rank}] Destroying distributed environment")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    _group_map.clear()
    logging.info(f"[rank: {rank}] Distributed environment destroyed")
    _parallelism_config = None
    _initialized = False
    gc.collect()


def _get_group(group: Group) -> torch.distributed.ProcessGroup:
    """Get process group for the specified group type.

    This function checks if the distributed environment is initialized.
    If not initialized and _parallelism_config is available, it will attempt to initialize.

    Args:
        group: Group type (DP, TP, or DP_AND_TP)

    Returns:
        Process group for the specified group type

    Raises:
        RuntimeError: If distributed environment is not initialized and cannot be auto-initialized
        ValueError: If group type is invalid
    """
    global _parallelism_config, _initialized

    # Check if we need to initialize
    if not torch.distributed.is_initialized() or not _initialized:
        if _parallelism_config is not None:
            # Auto-initialize if we have the config
            logging.info(
                "Auto-initializing distributed environment from stored parallelism_config"
            )
            init_distributed_environment(_parallelism_config)
        else:
            raise RuntimeError(
                "Distributed environment is not initialized. "
                "Call init_distributed_environment(parallelism_config) first, "
                "or ensure parallelism_config is available for auto-initialization."
            )

    # Determine the actual key to use in _group_map
    group_key = group
    tp_size = _parallelism_config.tp_size
    dp_size = _parallelism_config.dp_size
    world_size = _parallelism_config.world_size
    if group == Group.DP and dp_size > 1 and world_size != dp_size:
        tp_rank = torch.distributed.get_rank() % tp_size
        group_key = Group.DP.name + str(tp_rank)
    elif group == Group.TP and tp_size > 1 and world_size != tp_size:
        dp_rank = torch.distributed.get_rank() // tp_size
        group_key = Group.TP.name + str(dp_rank)
    else:
        # DP_AND_TP always uses Group.DP_AND_TP as key
        group_key = Group.DP_AND_TP

    if group_key not in _group_map:
        raise ValueError(
            f"Process group {group_key} not found. Make sure init_distributed_environment() was called."
        )

    return _group_map[group_key]


# 需要注意：调用 send/recv 时如果某些 rank 没有操作，就没有对应的 ncclgroupstart/ncclgroupend
# 这样直接使用 torch 的 send/recv 是错误的。
def send(tensor: torch.Tensor, dst: int, group: Group) -> None:
    """Send a tensor to a destination rank.

    Args:
        tensor: Tensor to send
        dst: Destination global rank
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.send(tensor, dst, group=process_group)


def recv(tensor: torch.Tensor, src: int, group: Group) -> torch.Tensor:
    """Receive a tensor from a source rank.

    Args:
        tensor: Tensor to receive into
        src: Source global rank
        group: Process group to use

    Returns:
        Received tensor (same as input tensor)
    """
    process_group = _get_group(group)
    torch.distributed.recv(tensor, src, group=process_group)
    return tensor


def broadcast(tensor: torch.Tensor, src: int, group: Group) -> None:
    """Broadcast a tensor from source rank to all ranks in the group.

    Args:
        tensor: Tensor to broadcast (will be modified on non-source ranks)
        src: Source global rank
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.broadcast(tensor, src, group=process_group)


def all_reduce(tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """All-reduce a tensor across all ranks in the group.

    Args:
        tensor: Tensor to all-reduce (will be modified in-place)
        group: Process group to use

    Returns:
        All-reduced tensor (same as input tensor)
    """
    # In HIP Graph capture mode, bypass torch.distributed for TP collective ops
    # to avoid the NCCL watchdog thread querying events on the capturing stream.
    if _in_graph_capture and _rccl_comm is not None and group == Group.TP:
        _rccl_all_reduce(tensor)
        return tensor

    if group == Group.TP:
        symm_mem_comm = get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allreduce(
            tensor
        ):
            return symm_mem_comm.all_reduce(tensor)

    process_group = _get_group(group)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.SUM, group=process_group
    )
    return tensor


def all_gather(tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Gather tensors from all ranks in the group.

    Args:
        tensor: Tensor to gather from this rank
        group: Process group to use

    Returns:
        Concatenated tensor containing all gathered tensors
        (shape: [world_size * tensor.shape[0]] + list(tensor.shape)[1:])
    """
    # In HIP Graph capture mode, bypass torch.distributed for TP collective ops
    if _in_graph_capture and _rccl_comm is not None and group == Group.TP:
        tensor_list = torch.zeros(
            [_rccl_world_size * tensor.shape[0]] + list(tensor.shape)[1:],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        _rccl_all_gather(tensor_list, tensor)
        return tensor_list

    if group == Group.TP:
        symm_mem_comm = get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allgather(
            tensor
        ):
            gathered = symm_mem_comm.all_gather(tensor)
            if gathered is not None:
                world_size = gathered.shape[0]
                return gathered.view(
                    [world_size * tensor.shape[0]] + list(tensor.shape)[1:]
                )

    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)

    tensor_list = torch.zeros(
        [world_size * tensor.shape[0]] + list(tensor.shape)[1:],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    torch.distributed.all_gather_into_tensor(tensor_list, tensor, group=process_group)
    return tensor_list

    # reference old implementation
    # tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    # torch.distributed.all_gather(tensor_list, tensor, group=process_group)
    # return torch.cat(tensor_list, dim=0)


def barrier(group: Group) -> None:
    """Barrier all ranks in the group.

    Args:
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.barrier(group=process_group)


__all__ = [
    "Group",
    "init_distributed_environment",
    "distributed_environment_initialized",
    "destroy_distributed_environment",
    "enter_graph_capture_mode",
    "exit_graph_capture_mode",
    "send",
    "recv",
    "broadcast",
    "all_reduce",
    "all_gather",
    "barrier",
]
