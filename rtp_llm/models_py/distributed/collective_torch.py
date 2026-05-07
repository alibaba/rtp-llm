from __future__ import annotations

import gc
import logging
import os
import re
from datetime import timedelta
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
import torch.distributed

from rtp_llm.models_py.distributed import rocm_rccl
from rtp_llm.models_py.distributed.symm_mem import (
    get_symm_mem_communicator,
    init_symm_mem_communicator,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig

# ParallelMode enum values matching C++ rtp_llm::ParallelMode in OpData.h
_CPP_PARALLEL_MODE_TP = 0
_CPP_PARALLEL_MODE_DP = 1
_CPP_PARALLEL_MODE_DP_AND_TP = 2
_UDS_SUN_PATH_LIMIT = 108


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
_cpu_tp_broadcaster_base_path: Optional[str] = None


def _make_cpu_tp_broadcaster_base_path(
    parallelism_config: ParallelismConfig,
    nccl_init_port: int,
) -> str:
    session_id = os.environ.get("RTP_LLM_CPU_TP_BROADCASTER_ID")
    if not session_id:
        session_id = f"ppid{os.getppid()}_port{nccl_init_port}"
    session_id = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)

    base_dir = os.environ.get("RTP_LLM_CPU_TP_BROADCASTER_DIR")
    if not base_dir:
        base_dir = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"rtp_llm_{os.getuid()}"
        )
    os.makedirs(base_dir, mode=0o700, exist_ok=True)
    base_path = os.path.join(
        base_dir, f"rtp_llm_tp_{session_id}_dp{parallelism_config.dp_rank}"
    )
    rank0_path = f"{base_path}_0.sock"
    if len(os.fsencode(rank0_path)) >= _UDS_SUN_PATH_LIMIT:
        raise ValueError(
            f"CpuTpBroadcaster UDS path too long ({len(os.fsencode(rank0_path))} "
            f"bytes, limit {_UDS_SUN_PATH_LIMIT - 1}): {rank0_path}"
        )
    return base_path


def _normalize_parallelism_ranks(parallelism_config: ParallelismConfig) -> None:
    # Process-group construction below uses this world-rank layout. Keep the
    # explicit config fields in sync for callsites that only fill sizes/ranks.
    if parallelism_config.tp_size > 0:
        old_tp_rank = parallelism_config.tp_rank
        old_dp_rank = parallelism_config.dp_rank
        tp_rank = parallelism_config.world_rank % parallelism_config.tp_size
        dp_rank = parallelism_config.world_rank // parallelism_config.tp_size
        if (old_tp_rank, old_dp_rank) != (tp_rank, dp_rank):
            logging.warning(
                "Normalize ParallelismConfig ranks from tp_rank=%s, dp_rank=%s "
                "to tp_rank=%s, dp_rank=%s for world_rank=%s, tp_size=%s",
                old_tp_rank,
                old_dp_rank,
                tp_rank,
                dp_rank,
                parallelism_config.world_rank,
                parallelism_config.tp_size,
            )
        parallelism_config.tp_rank = tp_rank
        parallelism_config.dp_rank = dp_rank


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
    global _group_map, _parallelism_config, _initialized, _cpu_tp_broadcaster_base_path

    # Check if already initialized (and not destroyed)
    if _initialized and torch.distributed.is_initialized():
        logging.warning(
            "Distributed environment already initialized, skipping initialization"
        )
        # Still need to create groups if they don't exist
        if not _group_map:
            _normalize_parallelism_ranks(parallelism_config)
            _cpu_tp_broadcaster_base_path = _make_cpu_tp_broadcaster_base_path(
                parallelism_config, nccl_init_port
            )
            _create_process_groups(parallelism_config, backend, timedelta(days=36500))
            _register_process_groups_to_cpp()
        if rocm_rccl.is_available_runtime() and parallelism_config.tp_size > 1:
            rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
        return

    _normalize_parallelism_ranks(parallelism_config)
    _cpu_tp_broadcaster_base_path = _make_cpu_tp_broadcaster_base_path(
        parallelism_config, nccl_init_port
    )

    assert backend in ["nccl"], "backend current only supports nccl"
    ip = nccl_comm_config.nccl_ip
    port = nccl_init_port
    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    local_rank = parallelism_config.local_rank

    rocm_rccl.configure_process_groups(parallelism_config)
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"

    # If torch.distributed is already initialized (e.g., by external code),
    # we still need to create our process groups
    if torch.distributed.is_initialized():
        logging.info("torch.distributed already initialized, creating process groups")
        _create_process_groups(parallelism_config, backend, timedelta(days=36500))
        _parallelism_config = parallelism_config
        _initialized = True
        _register_process_groups_to_cpp()
        if rocm_rccl.is_available_runtime() and parallelism_config.tp_size > 1:
            rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
        return

    logging.info(
        f"[rank: {world_rank}] initialize process_group: {ip}:{port}, rank: {world_rank}, world_size: {world_size}, "
        f"local_rank: {local_rank}, backend: {backend}, timeout: {timeout}",
    )

    # Use a very large timeout for NCCL so that workers simply block
    # until rank 0 has real work, instead of crashing with a timeout.
    # Note: timedelta.max overflows in PyTorch's C++ TCP store, so use 100 years instead.
    infinite_timeout = timedelta(days=36500)

    # DP_AND_TP (global group) - initialized via init_process_group
    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=world_rank,
        # device_id=torch.device(f"cuda:{local_rank}"), # https://github.com/pytorch/pytorch/pull/149144
        timeout=infinite_timeout,
    )
    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    _group_map[Group.DP_AND_TP] = torch.distributed.group.WORLD
    logging.info(
        f"[rank: {world_rank}] Created DP_AND_TP group {torch.distributed.group.WORLD} with ranks: {list(range(world_size))}"
    )

    # Create DP and TP groups
    _create_process_groups(parallelism_config, backend, timedelta(days=36500))
    _parallelism_config = parallelism_config
    _initialized = True
    _register_process_groups_to_cpp()
    if rocm_rccl.is_available_runtime() and parallelism_config.tp_size > 1:
        rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
    init_user_buffers_environment(parallelism_config)


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
                    timeout=timedelta(days=36500),
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
                    timeout=timedelta(days=36500),
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


def _register_process_groups_to_cpp():
    """Register Python comm op callbacks for C++ to call back into."""
    try:
        import librtp_compute_ops

        if not hasattr(librtp_compute_ops, "register_comm_ops"):
            logging.debug(
                "register_comm_ops not available, skip C++ comm ops registration"
            )
            return
    except ImportError:
        logging.debug(
            "librtp_compute_ops not available, skip C++ comm ops registration"
        )
        return

    # Build mode -> process_group mapping (int mode -> ProcessGroup)
    mode_to_group: Dict[int, torch.distributed.ProcessGroup] = {}
    registered_modes: set = set()

    for group_key, pg in _group_map.items():
        if group_key == Group.DP_AND_TP:
            if _CPP_PARALLEL_MODE_DP_AND_TP not in registered_modes:
                mode_to_group[_CPP_PARALLEL_MODE_DP_AND_TP] = pg
                registered_modes.add(_CPP_PARALLEL_MODE_DP_AND_TP)
        elif isinstance(group_key, str):
            if group_key.startswith(Group.TP.name):
                if _parallelism_config is not None:
                    dp_rank = (
                        torch.distributed.get_rank() // _parallelism_config.tp_size
                    )
                    expected_key = Group.TP.name + str(dp_rank)
                    if (
                        group_key == expected_key
                        and _CPP_PARALLEL_MODE_TP not in registered_modes
                    ):
                        mode_to_group[_CPP_PARALLEL_MODE_TP] = pg
                        registered_modes.add(_CPP_PARALLEL_MODE_TP)
            elif group_key.startswith(Group.DP.name):
                if _parallelism_config is not None:
                    tp_rank = torch.distributed.get_rank() % _parallelism_config.tp_size
                    expected_key = Group.DP.name + str(tp_rank)
                    if (
                        group_key == expected_key
                        and _CPP_PARALLEL_MODE_DP not in registered_modes
                    ):
                        mode_to_group[_CPP_PARALLEL_MODE_DP] = pg
                        registered_modes.add(_CPP_PARALLEL_MODE_DP)

    # If world_size == tp_size, WORLD is also TP group.
    if (
        _parallelism_config is not None
        and _parallelism_config.tp_size > 1
        and _parallelism_config.world_size == _parallelism_config.tp_size
        and _CPP_PARALLEL_MODE_TP not in registered_modes
    ):
        pg_world = _group_map.get(Group.DP_AND_TP)
        if pg_world is not None:
            mode_to_group[_CPP_PARALLEL_MODE_TP] = pg_world

    # NOTE: These callbacks are NOT thin wrappers around the module-level broadcast()/
    # all_reduce()/all_gather() because the C++ calling convention differs significantly:
    #   - C++ uses int mode (ParallelMode enum ordinal) instead of Group enum
    #   - execBroadcast passes multiple tensors + CPU tensors needing GPU promotion
    #   - execAllReduce supports dest tensor + multiple ReduceOp types
    #   - execAllGather writes into pre-allocated recv_buffers with inplace mode
    # The module-level functions have different signatures and semantics (e.g. all_gather
    # allocates a new tensor), so we implement the C++ contract directly here.

    def _ensure_cuda(t: torch.Tensor, device_id: int):
        """Move CPU tensor to CUDA if needed (NCCL requires CUDA tensors)."""
        if t.is_cuda:
            return t, False
        return t.to(torch.device("cuda", device_id)), True

    def cpp_broadcast(tensors: List[torch.Tensor], root: int, mode: int) -> None:
        """Broadcast tensors from root rank to all ranks in the group.

        Args:
            tensors: Tensors to broadcast, each is broadcast in-place from root.
            root: Source rank that holds the data.
            mode: ParallelMode int (0=TP, 1=DP, 2=DP_AND_TP) selecting process group.
        """
        pg = mode_to_group.get(mode)
        if pg is None or pg.size() < 2:
            return
        global_root = torch.distributed.get_global_rank(pg, root)
        device_id = torch.cuda.current_device()
        for t in tensors:
            gpu_t, was_cpu = _ensure_cuda(t, device_id)
            torch.distributed.broadcast(gpu_t, global_root, group=pg)
            if was_cpu:
                t.copy_(gpu_t)

    _REDUCE_OPS = {
        0: torch.distributed.ReduceOp.SUM,
        1: torch.distributed.ReduceOp.PRODUCT,
        2: torch.distributed.ReduceOp.MAX,
        3: torch.distributed.ReduceOp.MIN,
        4: torch.distributed.ReduceOp.AVG,
    }

    def cpp_allreduce(
        tensor: torch.Tensor, op: int, mode: int, dest: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """All-reduce a tensor across ranks in the group.

        Args:
            tensor: Input tensor to reduce.
            op: ReduceOp int (0=SUM, 1=PROD, 2=MAX, 3=MIN, 4=AVG).
            mode: ParallelMode int (0=TP, 1=DP, 2=DP_AND_TP) selecting process group.
            dest: If not None, result is written here instead of reducing in-place on tensor.
        Returns:
            The reduced tensor (dest if provided, otherwise tensor).
        """
        pg = mode_to_group.get(mode)
        if pg is None or pg.size() < 2:
            return tensor if dest is None else tensor
        target = dest if dest is not None else tensor
        if dest is not None:
            target.copy_(tensor)
        device_id = torch.cuda.current_device()
        gpu_t, was_cpu = _ensure_cuda(target, device_id)
        torch.distributed.all_reduce(
            gpu_t, op=_REDUCE_OPS.get(op, torch.distributed.ReduceOp.SUM), group=pg
        )
        if was_cpu:
            target.copy_(gpu_t)
        return target

    def cpp_allgather(
        recv_buffers: List[torch.Tensor],
        mode: int,
        send_buffers: List[torch.Tensor],
        inplace: bool,
    ) -> None:
        """All-gather tensors from all ranks into recv_buffers.

        Args:
            recv_buffers: Output tensors, each of size [world_size * per_rank_numel].
            mode: ParallelMode int (0=TP, 1=DP, 2=DP_AND_TP) selecting process group.
            send_buffers: Per-rank input tensors (used when inplace=False).
            inplace: If True, each rank's send data is extracted from its slice in recv_buffers;
                     if False, send data comes from send_buffers.
        """
        pg = mode_to_group.get(mode)
        if pg is None or pg.size() < 2:
            return
        world_size = pg.size()
        device_id: Optional[int] = None
        rank = pg.rank() if inplace else 0
        for i, recv_buf in enumerate(recv_buffers):
            recv_on_cpu = not recv_buf.is_cuda
            data_num = recv_buf.numel() // world_size
            if not inplace:
                send_tensor = send_buffers[i]
                if (
                    not recv_on_cpu
                    and send_tensor.is_cuda
                    and recv_buf.is_contiguous()
                    and send_tensor.is_contiguous()
                ):
                    # Fast path for C++ explicit-send allgather: keep the 2D
                    # output shape so c10d can launch directly without local
                    # rank-slice packing or Python-side CUDA promotion.
                    torch.distributed.all_gather_into_tensor(
                        recv_buf, send_tensor, group=pg
                    )
                    continue

            if device_id is None:
                device_id = torch.cuda.current_device()
            gpu_recv = (
                recv_buf.to(torch.device("cuda", device_id))
                if recv_on_cpu
                else recv_buf
            )
            gpu_recv_flat = gpu_recv.reshape(-1)
            if inplace:
                send_tensor = gpu_recv_flat.narrow(
                    0, rank * data_num, data_num
                ).contiguous()
            else:
                send_tensor, _ = _ensure_cuda(send_tensor, device_id)
            torch.distributed.all_gather_into_tensor(
                gpu_recv_flat, send_tensor, group=pg
            )
            if recv_on_cpu:
                recv_buf.copy_(gpu_recv)

    librtp_compute_ops.register_comm_ops(cpp_broadcast, cpp_allreduce, cpp_allgather)
    logging.info(
        f"Registered C++ comm ops callbacks (modes: {list(mode_to_group.keys())})"
    )

    # Bootstrap the UDS-backed intra-node TP broadcaster right after new_group.
    # Lazy C++ init can race if a peer reaches tpSyncModelInputs before rank 0
    # binds; cross-node TP keeps the NCCL fallback.
    if (
        _parallelism_config is not None
        and _parallelism_config.tp_size > 1
        and _parallelism_config.tp_size <= _parallelism_config.local_world_size
        and hasattr(librtp_compute_ops, "init_cpu_tp_broadcaster")
    ):
        # Parent PID plus NCCL init port gives peers a shared per-init UDS path.
        # dp_rank disambiguates DP groups on the same node.
        base_path = _cpu_tp_broadcaster_base_path
        assert base_path is not None
        librtp_compute_ops.init_cpu_tp_broadcaster(
            _parallelism_config.tp_rank,
            _parallelism_config.tp_size,
            base_path,
        )
        logging.info(
            f"Initialized CpuTpBroadcaster (tp_rank={_parallelism_config.tp_rank}, "
            f"tp_size={_parallelism_config.tp_size}, base_path={base_path})"
        )


def distributed_environment_initialized() -> bool:
    """Check if distributed environment is initialized.

    Returns:
        True if distributed environment is initialized, False otherwise
    """
    return torch.distributed.is_initialized()


def init_user_buffers_environment(parallelism_config: ParallelismConfig):
    """Initialize user buffers communicator for context parallelism."""
    from rtp_llm.models_py.utils.arch import is_cuda

    if parallelism_config.use_ub_comm and is_cuda():

        from rtp_llm.models_py.distributed.user_buffers import (
            init_user_buffers_communicator,
        )

        local_rank = parallelism_config.local_rank
        world_size = parallelism_config.world_size

        buffer_size = parallelism_config.prefill_cp_config.comm_buffer_size

        logging.info(
            f"[rank: {parallelism_config.world_rank}] Initializing user buffers communicator "
            f"with buffer_size: {buffer_size}, local_rank: {local_rank}, world_size: {world_size}"
        )
        init_user_buffers_communicator(
            _get_group(Group.TP), local_rank, world_size, buffer_size
        )


def destroy_distributed_environment():
    """Destroy distributed environment and clean up process groups.

    After calling this function, init_distributed_environment() can be called again
    to reinitialize the distributed environment.
    """
    global _group_map, _parallelism_config, _initialized, _cpu_tp_broadcaster_base_path

    rank = torch.distributed.get_rank()
    logging.info(f"[rank: {rank}] Destroying distributed environment")

    from rtp_llm.models_py.utils.arch import is_cuda

    if is_cuda():
        from rtp_llm.models_py.distributed.user_buffers import (
            destroy_user_buffers_communicator,
        )

        destroy_user_buffers_communicator()

    try:
        import librtp_compute_ops

        if hasattr(librtp_compute_ops, "clear_comm_ops"):
            librtp_compute_ops.clear_comm_ops()
        if hasattr(librtp_compute_ops, "destroy_cpu_tp_broadcaster"):
            librtp_compute_ops.destroy_cpu_tp_broadcaster()
    except ImportError:
        pass

    # Clean up ROCm RCCL capture comm before destroying process groups,
    # so that re-init will bootstrap a fresh communicator instead of
    # reusing the stale one from the destroyed environment.
    if rocm_rccl.is_available_runtime():
        rocm_rccl.destroy_capture_comm()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    _group_map.clear()
    logging.info(f"[rank: {rank}] Distributed environment destroyed")
    _parallelism_config = None
    _cpu_tp_broadcaster_base_path = None
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
    rocm_rccl.ensure_capture_comm_ready(group == Group.TP)
    if rocm_rccl.should_use_capture_collectives(group == Group.TP):
        return rocm_rccl.capture_all_reduce(tensor, _get_group(group))

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
    rocm_rccl.ensure_capture_comm_ready(group == Group.TP)
    if rocm_rccl.should_use_capture_collectives(group == Group.TP):
        return rocm_rccl.capture_all_gather(tensor)

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
    "init_user_buffers_environment",
    "distributed_environment_initialized",
    "destroy_distributed_environment",
    "send",
    "recv",
    "broadcast",
    "all_reduce",
    "all_gather",
    "barrier",
]
