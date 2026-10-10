from __future__ import annotations

import ctypes
import gc
import logging
import os
import re
import socket
from bisect import bisect_left
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist

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
_rocm_rccl = None
_symm_mem = None
_flashinfer_allreduce = None


def _get_rocm_rccl():
    """Lazily import ROCm RCCL helpers.

    The module self-gates on non-ROCm runtimes via its internal
    ``_is_rocm_runtime`` checks (same semantics as the eager import on main),
    so callers may invoke its wrappers unconditionally.
    """
    global _rocm_rccl
    if _rocm_rccl is None:
        from rtp_llm.models_py.distributed import rocm_rccl

        _rocm_rccl = rocm_rccl
    return _rocm_rccl


def _get_symm_mem():
    global _symm_mem
    if _symm_mem is None:
        from rtp_llm.models_py.distributed import symm_mem

        _symm_mem = symm_mem
    return _symm_mem


def _get_flashinfer_allreduce():
    global _flashinfer_allreduce
    if _flashinfer_allreduce is None:
        from rtp_llm.models_py.distributed import flashinfer_all_reduce

        _flashinfer_allreduce = flashinfer_all_reduce
    return _flashinfer_allreduce


def _init_flashinfer_allreduce(
    parallelism_config: ParallelismConfig,
    disable_custom_all_reduce: Optional[bool] = None,
) -> None:
    if parallelism_config.tp_size <= 1:
        return
    _get_flashinfer_allreduce().init_flashinfer_allreduce(
        _get_group(Group.TP),
        torch.device("cuda", parallelism_config.local_rank),
        single_node=parallelism_config.tp_size <= parallelism_config.local_world_size,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )


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
    disable_custom_all_reduce: Optional[bool] = None,
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
        disable_custom_all_reduce: Explicit server-wide custom AllReduce
            override. ``None`` preserves the implementation-specific default.

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
            _create_process_groups(
                parallelism_config,
                backend,
                timedelta(days=36500),
                disable_custom_all_reduce,
            )
            _register_process_groups_to_cpp()
        rocm_rccl = _get_rocm_rccl()
        if rocm_rccl is not None and parallelism_config.tp_size > 1:
            rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
        _init_flashinfer_allreduce(parallelism_config, disable_custom_all_reduce)
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

    rocm_rccl = _get_rocm_rccl()
    if rocm_rccl is not None:
        rocm_rccl.configure_process_groups(parallelism_config)
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"

    # If torch.distributed is already initialized (e.g., by external code),
    # we still need to create our process groups
    if torch.distributed.is_initialized():
        logging.info("torch.distributed already initialized, creating process groups")
        _create_process_groups(
            parallelism_config,
            backend,
            timedelta(days=36500),
            disable_custom_all_reduce,
        )
        _parallelism_config = parallelism_config
        _initialized = True
        _register_process_groups_to_cpp()
        if rocm_rccl is not None and parallelism_config.tp_size > 1:
            rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
        _init_flashinfer_allreduce(parallelism_config, disable_custom_all_reduce)
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
    _create_process_groups(
        parallelism_config,
        backend,
        timedelta(days=36500),
        disable_custom_all_reduce,
    )
    _parallelism_config = parallelism_config
    _initialized = True
    _register_process_groups_to_cpp()
    if rocm_rccl is not None and parallelism_config.tp_size > 1:
        rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
    init_user_buffers_environment(parallelism_config)
    _init_flashinfer_allreduce(parallelism_config, disable_custom_all_reduce)


def _create_process_groups(
    parallelism_config: ParallelismConfig,
    backend: str,
    timeout: Optional[timedelta],
    disable_custom_all_reduce: Optional[bool] = None,
):
    """Create DP and TP process groups.

    Args:
        parallelism_config: Configuration for parallelism setup
        backend: Distributed backend
        timeout: Timeout for process group creation
        disable_custom_all_reduce: Explicit custom AllReduce override.
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

                _get_symm_mem().init_symm_mem_communicator(
                    tp_group,
                    disable_custom_all_reduce=disable_custom_all_reduce,
                )

                # All ranks must wait for group creation to complete
                torch.distributed.barrier()
    elif tp_size > 1 and world_size == tp_size:
        # Single TP group: WORLD is the TP group, init symm_mem for it
        _get_symm_mem().init_symm_mem_communicator(
            torch.distributed.group.WORLD,
            disable_custom_all_reduce=disable_custom_all_reduce,
        )


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

    # If world_size == dp_size, WORLD is also the only DP group.
    if (
        _parallelism_config is not None
        and _parallelism_config.dp_size > 1
        and _parallelism_config.world_size == _parallelism_config.dp_size
        and _CPP_PARALLEL_MODE_DP not in registered_modes
    ):
        pg_world = _group_map.get(Group.DP_AND_TP)
        if pg_world is not None:
            mode_to_group[_CPP_PARALLEL_MODE_DP] = pg_world

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
            if dest is None:
                return tensor
            dest.copy_(tensor)
            return dest
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
            if not inplace:
                for i, recv_buf in enumerate(recv_buffers):
                    recv_buf.copy_(send_buffers[i].reshape(recv_buf.shape))
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
        _get_flashinfer_allreduce().destroy_flashinfer_allreduce()

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
    rocm_rccl = _get_rocm_rccl()
    if rocm_rccl is not None:
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


def get_process_group(group: Group) -> torch.distributed.ProcessGroup:
    """Return RTP's initialized process group for collective operators."""

    return _get_group(group)


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


def all_reduce(
    tensor: torch.Tensor, group: Group, *, inplace: bool = True
) -> torch.Tensor:
    """All-reduce a tensor across all ranks in the group.

    Args:
        tensor: Tensor to all-reduce.
        group: Process group to use
        inplace: If true, write fast-path results back to ``tensor``. Defaults to
            true to preserve the in-place contract of torch.distributed.all_reduce.

    Returns:
        All-reduced tensor.
    """
    rocm_rccl = _get_rocm_rccl()
    if rocm_rccl is not None:
        rocm_rccl.ensure_capture_comm_ready(group == Group.TP)
        if rocm_rccl.should_use_capture_collectives(group == Group.TP):
            target = tensor if inplace else tensor.clone()
            return rocm_rccl.capture_all_reduce(target, _get_group(group))

    if group == Group.TP:
        flashinfer_ar = _get_flashinfer_allreduce().get_flashinfer_allreduce()
        if flashinfer_ar is not None and flashinfer_ar.should_use(tensor):
            result = flashinfer_ar.all_reduce(tensor)
            if inplace:
                tensor.copy_(result)
                return tensor
            return result

        symm_mem_comm = _get_symm_mem().get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allreduce(
            tensor
        ):
            return symm_mem_comm.all_reduce(tensor, out=tensor if inplace else None)

    target = tensor if inplace else tensor.clone()
    process_group = _get_group(group)
    torch.distributed.all_reduce(
        target, op=torch.distributed.ReduceOp.SUM, group=process_group
    )
    return target


def all_gather(tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Gather tensors from all ranks in the group.

    Args:
        tensor: Tensor to gather from this rank
        group: Process group to use

    Returns:
        Concatenated tensor containing all gathered tensors
        (shape: [world_size * tensor.shape[0]] + list(tensor.shape)[1:])
    """
    rocm_rccl = _get_rocm_rccl()
    if rocm_rccl is not None:
        rocm_rccl.ensure_capture_comm_ready(group == Group.TP)
        if rocm_rccl.should_use_capture_collectives(group == Group.TP):
            process_group = _get_group(group)
            return rocm_rccl.capture_all_gather(tensor, process_group)

    if group == Group.TP:
        symm_mem_comm = _get_symm_mem().get_symm_mem_communicator()
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

    # all_gather_into_tensor writes every element before returning. Avoid
    # initializing the output only to overwrite it during the collective.
    tensor_list = torch.empty(
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


def all_to_all_single(
    tensor: torch.Tensor, group: Group, *, output: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Exchange equal contiguous dim-0 chunks across the process group."""

    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)
    if world_size <= 1:
        return tensor
    if tensor.ndim == 0 or tensor.shape[0] % world_size:
        raise ValueError(
            "all_to_all_single requires dim0 divisible by group size: "
            f"shape={tuple(tensor.shape)}, world_size={world_size}"
        )
    send = tensor.contiguous()
    if output is None:
        output = torch.empty_like(send)
    elif (
        output.shape != send.shape
        or output.dtype != send.dtype
        or output.device != send.device
        or not output.is_contiguous()
        or output.untyped_storage().data_ptr() == send.untyped_storage().data_ptr()
    ):
        raise ValueError("AllToAll output must match input and use separate contiguous storage")
    torch.distributed.all_to_all_single(output, send, group=process_group)
    return output


def all_gather_into(
    tensor: torch.Tensor,
    output: torch.Tensor,
    group: Group,
) -> torch.Tensor:
    """All-gather into a caller-owned contiguous output buffer."""

    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)
    expected_shape = [world_size * tensor.shape[0]] + list(tensor.shape[1:])
    if (
        list(output.shape) != expected_shape
        or output.dtype != tensor.dtype
        or output.device != tensor.device
        or not tensor.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError(
            "all_gather_into input/output mismatch: "
            f"input={tuple(tensor.shape)}, output={tuple(output.shape)}, "
            f"expected={tuple(expected_shape)}"
        )
    if world_size <= 1:
        output.copy_(tensor)
        return output
    torch.distributed.all_gather_into_tensor(output, tensor, group=process_group)
    return output


def reduce_scatter(input_tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Reduce-scatter a tensor across all ranks in the group.

    Reduces (sums) the input tensor across all ranks and scatters the result
    so that each rank receives a 1/world_size chunk of the reduced tensor.

    Args:
        input_tensor: Full-size tensor to reduce-scatter
            (shape: [world_size * chunk_size] + remaining_dims)
        group: Process group to use

    Returns:
        Scattered chunk of the reduced tensor for this rank
        (shape: [chunk_size] + remaining_dims)
    """
    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)
    assert input_tensor.shape[0] % world_size == 0, (
        f"reduce_scatter: input dim 0 ({input_tensor.shape[0]}) "
        f"must be divisible by world_size ({world_size})"
    )
    chunk_size = input_tensor.shape[0] // world_size
    output_tensor = torch.empty(
        [chunk_size] + list(input_tensor.shape[1:]),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.distributed.reduce_scatter_tensor(
        output_tensor,
        input_tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )
    return output_tensor


def barrier(group: Group) -> None:
    """Barrier all ranks in the group.

    Args:
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.barrier(group=process_group)


# BF16 projection reduction

def reduce_scatter_bf16(input_tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Keep BF16 communication without size-dependent widening or extra casts.

    Other dtypes retain RTP's existing collective behavior.
    """
    if input_tensor.dtype != torch.bfloat16:
        return reduce_scatter(input_tensor, group)
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
    return reduce_scatter(source, group)


# Push ReduceScatter and shared NVLink workspace setup
# BF16 partials accumulate in FP32 in rank order and round once to BF16.
# This need not be bitwise identical to NCCL's BF16 reduction tree.

# Measured TP8 / BF16 / hidden=7168 launch configurations. Intermediate M uses
# the next measured bucket; decode above 1024 keeps the largest configuration.
_ROWS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
_LAUNCHES = (
    (128, 128),
    (96, 128),
    (96, 128),
    (128, 128),
    (64, 128),
    (96, 128),
    (96, 128),
    (128, 128),
    (128, 256),
    (128, 256),
    (96, 512),
    (128, 256),
    (128, 256),
    (128, 512),
)


def _load_push_rs_kernel():
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops.push_reduce_scatter


def _nvlink_fabric(uuid):
    """Return the local GPU's usable fabric identity; old drivers may lack it."""
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + uuid)
        info = pynvml.c_nvmlGpuFabricInfoV_t()
        pynvml.nvmlDeviceGetGpuFabricInfoV(handle, ctypes.byref(info))
        # Older NVML bindings decode c_char arrays as text and truncate at NUL.
        field = type(info).clusterUuid
        cluster = ctypes.string_at(ctypes.addressof(info) + field.offset, field.size)
        if (
            info.state == pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
            and info.status == pynvml.NVML_SUCCESS
            and any(cluster)
        ):
            return cluster, int(info.cliqueId)
    except (AttributeError, pynvml.NVMLError):
        pass
    finally:
        pynvml.nvmlShutdown()
    return None


def _is_fabric_allocation(tensor):
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops.is_cuda_fabric_allocation(tensor)


def _nvlink_peers(uuids: list[str], *, hostnames=None, fabrics=None) -> bool:
    if hostnames is not None and len(set(hostnames)) > 1:
        # Remote UUIDs cannot be looked up using this host's NVML. A nonzero
        # cluster UUID plus clique ID identifies a shared multi-node NVLink domain.
        return (
            fabrics is not None
            and len(fabrics) == len(uuids)
            and fabrics[0] is not None
            and all(f == fabrics[0] for f in fabrics)
        )
    # UUIDs remain unambiguous with CUDA_VISIBLE_DEVICES remapping. NVML also
    # handles NVSwitch paths, unlike testing only direct NVLink remote ports.
    import pynvml

    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByUUID("GPU-" + u) for u in uuids]
        return all(
            pynvml.nvmlDeviceGetP2PStatus(a, b, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
            == pynvml.NVML_P2P_STATUS_OK
            for i, a in enumerate(handles)
            for j, b in enumerate(handles)
            if i != j
        )
    finally:
        pynvml.nvmlShutdown()


def is_same_host_group(group):
    """Agree on allocation ordering before any prefill workspace rendezvous."""
    hosts = [None] * group.size()
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    return len(set(hosts)) == 1


def _all_ready(ready: bool, group, device) -> bool:
    flag = torch.tensor([int(ready)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def create_push_reduce_scatter(group, device, *, max_m: int, n: int):
    """Return a same-host NVLink workspace for BF16 Decode ReduceScatter.

    No allocation, rendezvous, topology query or backend change occurs during
    forward or CUDA Graph capture.
    """
    world_size = group.size()
    if world_size not in (2, 4, 8, 16) or n != 7168:
        return None
    if max_m <= 0 or max_m % world_size:
        raise ValueError("Push RS capacity must be positive and divisible by TP size")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Push RS must be configured before CUDA Graph capture")
    failure = ""
    try:
        import torch.distributed._symmetric_memory as symm

        kernel = _load_push_rs_kernel()
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) not in ((10, 0), (10, 3)):
            raise RuntimeError("requires SM100/SM103")
        if symm.get_backend(device) != "CUDA":
            raise RuntimeError("requires the CUDA symmetric-memory backend")
        metadata = (
            socket.gethostname(),
            str(props.uuid),
            max_m,
            n,
            props.multi_processor_count,
            _nvlink_fabric(str(props.uuid)),
        )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info("[PUSH_RS] NCCL fallback: %s", failure or "unsupported peer")
        return None

    peers_metadata = [None] * world_size
    dist.all_gather_object(peers_metadata, metadata, group=group)
    try:
        uuids = [item[1] for item in peers_metadata]
        hostnames = [item[0] for item in peers_metadata]
        cross_host = len(set(hostnames)) > 1
        if cross_host:
            raise RuntimeError("Decode push ReduceScatter requires one host")
        if (
            any(item[2:-1] != metadata[2:-1] for item in peers_metadata)
            or len(set(uuids)) != world_size
            or not _nvlink_peers(
                uuids,
                hostnames=hostnames,
                fabrics=[item[-1] for item in peers_metadata],
            )
        ):
            raise RuntimeError(
                "requires distinct GPUs in one NVLink domain with matching capacities and SM counts"
            )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info(
            "[PUSH_RS] NCCL fallback: %s", failure or "unsupported peer topology"
        )
        return None

    # Two phases, each with one output shard slot per producer = 2 full inputs.
    # All ranks must finish allocating before any of them enters rendezvous.
    storage = counters = None
    try:
        storage = symm.empty(2 * max_m * n * 2, dtype=torch.uint8, device=device)
        counters = torch.zeros(
            props.multi_processor_count, dtype=torch.int32, device=device
        )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.warning(
            "[PUSH_RS] NCCL fallback allocating workspace: %s",
            failure or "peer allocation failed",
        )
        return None
    handle = None
    try:
        handle = symm.rendezvous(storage, group=group)
        buffers = [
            handle.get_buffer(i, [storage.numel()], torch.uint8)
            for i in range(world_size)
        ]
        storage.zero_()
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.warning(
            "[PUSH_RS] NCCL fallback mapping workspace: %s",
            failure or "peer mapping failed",
        )
        return None
    # No rank may poll or write a peer whose initialization has not completed.
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    state = PushReduceScatter(kernel, storage, handle, buffers, counters, group.rank())
    logging.info(
        "[PUSH_RS] enabled TP%d max_m=%d n=%d workspace=%.3f MiB",
        world_size,
        max_m,
        n,
        storage.numel() / (1 << 20),
    )
    return state


class PushReduceScatter:
    """Shared workspace with the same ordering contract as the native kernel.

    The caller must serialize eager calls and graph replays using this state.
    RTP's graph runner synchronizes the device between warmup and capture and
    runs inference on its compute stream. Do not insert stream waits here:
    replay bypasses Python, so Python cannot track the last execution stream.
    """

    def __init__(self, kernel, storage, handle, peers, counters, rank):
        self.kernel = kernel
        # Own all mappings for the entire model/graph lifetime.
        self.storage = storage
        self.handle = handle
        self.peers = peers
        self.counters = counters
        self.rank = rank

    def reduce_scatter(self, partial: torch.Tensor, output: torch.Tensor) -> None:
        # Use the existing TP8 / hidden=7168 buckets as the initial launch profile
        # for each supported TP size. The bumper
        # CTA keeps phases consistent when either blocks or threads changes.
        index = min(bisect_left(_ROWS, partial.shape[0]), len(_ROWS) - 1)
        blocks, threads = _LAUNCHES[index]
        blocks = min(blocks, self.counters.numel() - 1)
        self.kernel(
            partial, output, self.peers, self.counters, self.rank, blocks, threads
        )


# BF16 / FP8 AllGather workspaces
# FP8 values and packed UE8M0 scales move together without requantization.

# Kernel launch tuning uses the gathered (global) row count.
_STAGING_ROWS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
_BF16_STAGING = (
    (96, 128),
    (64, 256),
    (128, 256),
    (128, 256),
    (96, 512),
    (128, 512),
    (96, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
)
_FP8_STAGING = (
    (128, 256),
    (64, 256),
    (64, 256),
    (64, 256),
    (96, 256),
    (128, 256),
    (128, 512),
    (128, 512),
    (96, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
    (128, 512),
)


def _load_custom_ag_kernel(fp8):
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    name = "custom_all_gather_fp8_staging" if fp8 else "custom_all_gather_staging"
    return getattr(rtp_llm_ops, name)


def create_custom_all_gather(
    group, device, *, max_m: int, k: int, fp8: bool
):
    """Return a workspace or collectively retain the existing AG backend.

    All capability, allocation and multicast mapping checks finish here. Forward
    must never rendezvous or make a rank-local choice to fall back to NCCL.
    max_m is the gathered output capacity (TP times the per-rank input rows).
    """
    world_size = group.size()
    if world_size not in (2, 4, 8, 16) or k != 7168:
        return None
    if max_m <= 0 or max_m % world_size:
        raise ValueError("custom AG capacity must be positive and divisible by TP size")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Custom AG must be configured before CUDA Graph capture")
    failure = ""
    try:
        import torch.distributed._symmetric_memory as symm

        kernel = _load_custom_ag_kernel(fp8)
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) not in ((10, 0), (10, 3)):
            raise RuntimeError("requires SM100/SM103")
        if symm.get_backend(device) != "CUDA":
            raise RuntimeError("requires the CUDA symmetric-memory backend")
        metadata = (
            socket.gethostname(),
            str(props.uuid),
            max_m,
            k,
            fp8,
            props.multi_processor_count,
            _nvlink_fabric(str(props.uuid)),
        )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info(
            "[CUSTOM_AG] existing backend retained: %s", failure or "unsupported peer"
        )
        return None

    peers_metadata = [None] * world_size
    dist.all_gather_object(peers_metadata, metadata, group=group)
    try:
        uuids = [item[1] for item in peers_metadata]
        hostnames = [item[0] for item in peers_metadata]
        cross_host = len(set(hostnames)) > 1
        if cross_host:
            raise RuntimeError("Decode FP8 staging AllGather requires one host")
        if (
            any(item[2:-1] != metadata[2:-1] for item in peers_metadata)
            or len(set(uuids)) != world_size
            or not _nvlink_peers(
                uuids,
                hostnames=hostnames,
                fabrics=[item[-1] for item in peers_metadata],
            )
        ):
            raise RuntimeError(
                "requires distinct NVLink peers and matching capacities/precision/SM counts"
            )
    except Exception as exc:
        failure = str(exc)
    if not _all_ready(not failure, group, device):
        logging.info(
            "[CUSTOM_AG] existing backend retained: %s",
            failure or "unsupported topology",
        )
        return None

    # Reserve staging for the configured Decode/MTP token capacity.
    local_staging_m = max_m // world_size
    groups = (k + 511) // 512
    if fp8:
        # Four payload vectors plus one vector containing their original sign
        # bits and ready markers. Each phase owns one slot per source rank.
        vectors = local_staging_m * k // 16 + groups * ((local_staging_m + 3) // 4)
        slot_bytes = (vectors + 3) // 4 * 80
    else:
        slot_bytes = local_staging_m * k * 2
    storage = {}
    try:
        storage["staging"] = symm.empty(
            2 * world_size * slot_bytes, dtype=torch.uint8, device=device
        )
        storage["values"] = symm.empty(
            (max_m, k), dtype=torch.uint8 if fp8 else torch.bfloat16, device=device
        )
        if fp8:
            storage["scales"] = symm.empty(
                ((max_m + 3) // 4 * 4) * groups, dtype=torch.int32, device=device
            )
        if cross_host and not all(_is_fabric_allocation(t) for t in storage.values()):
            raise RuntimeError("cross-host NVLink requires CUDA FABRIC allocations")
        counters = torch.zeros(
            props.multi_processor_count, dtype=torch.int32, device=device
        )
    except Exception as exc:
        failure = str(exc)
    # Agree before entering any rendezvous, including when only one rank OOMs.
    if not _all_ready(not failure, group, device):
        logging.warning(
            "[CUSTOM_AG] workspace allocation failed: %s",
            failure or "peer allocation failed",
        )
        return None
    handles = {}
    for name, tensor in storage.items():
        try:
            handle = symm.rendezvous(tensor, group=group)
            if not handle.multicast_ptr:
                raise RuntimeError(f"{name} has no multicast mapping")
            handles[name] = handle
        except Exception as exc:
            failure = str(exc)
        if not _all_ready(not failure, group, device):
            logging.warning(
                "[CUSTOM_AG] multicast mapping failed: %s",
                failure or "peer mapping failed",
            )
            return None
    storage["staging"].zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    workspace = CustomAllGather(
        kernel, storage, handles, counters, group.rank(), fp8, world_size
    )
    logging.info(
        "[CUSTOM_AG] enabled TP%d max_m=%d k=%d fp8=%s workspace=%.3f MiB",
        world_size,
        max_m,
        k,
        fp8,
        sum(t.numel() * t.element_size() for t in storage.values()) / (1 << 20),
    )
    return workspace


class CustomAllGather:
    """Same-order, serialized collective calls, including graph replay.

    Results alias reusable workspace and must be consumed before the next AG
    with this state. RTP's AG/GEMM caller completes all GEMMs on the same stream.
    Own the symmetric allocations and handles for the entire model/graph lifetime.
    """

    def __init__(self, kernel, storage, handles, counters, rank, fp8, world_size):
        self.staging_kernel = kernel
        self.storage = storage
        self.handles = handles
        self.counters = counters
        self.rank = rank
        self.fp8 = fp8
        self.world_size = world_size
        self.pointers = {
            name: int(handle.multicast_ptr) for name, handle in handles.items()
        }

    def _launch(self, m):
        configs = _FP8_STAGING if self.fp8 else _BF16_STAGING
        blocks, threads = configs[min(bisect_left(_STAGING_ROWS, m), len(_STAGING_ROWS) - 1)]
        return min(blocks, self.counters.numel() - 1), threads

    @staticmethod
    def _aligned(tensor):
        # A view may be contiguous with a rank-local, unaligned storage offset.
        # Copy it without changing the group's selected communication backend.
        return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()

    def all_gather(self, local_input):
        if self.fp8:
            raise TypeError("BF16 AG requires a BF16 workspace")
        m = local_input.shape[0] * self.world_size
        if not 0 < m <= self.storage["values"].shape[0]:
            raise ValueError("BF16 AG exceeds configured capacity")
        local_input = self._aligned(local_input)
        output = self.storage["values"][:m]
        blocks, threads = self._launch(m)
        self.staging_kernel(
            local_input,
            output,
            self.storage["staging"],
            self.counters,
            self.pointers["staging"],
            self.rank,
            blocks,
            threads,
        )
        return output

    def all_gather_fp8(self, values, wire):
        if not self.fp8:
            raise TypeError("FP8 AG requires an FP8 workspace")
        m, k = values.shape
        m *= self.world_size
        if not 0 < m <= self.storage["values"].shape[0]:
            raise ValueError("FP8 AG exceeds configured capacity")
        values = self._aligned(values.view(torch.uint8))
        wire = self._aligned(wire)
        output = self.storage["values"][:m]
        scale_rows = (m + 3) // 4 * 4
        scales = self.storage["scales"][: ((k + 511) // 512) * scale_rows].view(
            -1, scale_rows
        )
        blocks, threads = self._launch(m)
        self.staging_kernel(
            values,
            wire,
            output,
            scales,
            self.storage["staging"],
            self.counters,
            self.pointers["staging"],
            self.rank,
            blocks,
            threads,
        )
        return output.view(torch.float8_e4m3fn), scales[:, :m].T


# FP8 activation wire format

@dataclass(frozen=True)
class Fp8Activation:
    """Group128 E4M3 rows with TMA-aligned packed UE8M0 scale storage."""

    values: torch.Tensor
    scale_wire: torch.Tensor

    def __post_init__(self):
        if (
            self.values.ndim != 2
            or self.values.dtype != torch.float8_e4m3fn
            or not self.values.is_contiguous()
        ):
            raise ValueError("FP8 activation values must be contiguous E4M3 [M,K]")
        m, k = self.values.shape
        if (
            k % 128
            or self.scale_wire.shape != ((k + 511) // 512, (m + 3) // 4 * 4)
            or self.scale_wire.dtype != torch.int32
            or not self.scale_wire.is_contiguous()
            or self.scale_wire.device != self.values.device
        ):
            raise ValueError("FP8 activation scales must be packed UE8M0 group128")

    @property
    def shape(self):
        return self.values.shape

    @property
    def device(self):
        return self.values.device

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def is_cuda(self):
        return self.values.is_cuda


# BF16 MTP workspace ownership

class KimiK3MtpBf16Collectives:
    """Initialize private staging and Push RS workspaces before Graph capture.

    Native MTP runs on the model compute stream. Its Q1 proposal and fixed-Q
    update share these workspaces serially, as in feat/k3_dev. Real long prompt
    initialization outside the configured Decode capacity keeps the existing
    collective path. Target projections never use this instance.
    """

    def __init__(self, device, *, max_tokens, hidden_size):
        group = _get_group(Group.TP)
        self.tp_size = group.size()
        self.max_tokens = int(max_tokens)
        self.hidden_size = int(hidden_size)
        if self.max_tokens <= 0 or self.max_tokens % self.tp_size:
            raise ValueError("MTP BF16 collective capacity must be divisible by TP")
        # Fixed feat uses BF16 multimem for embedding and final SP output.
        # Give MTP its own buffer: the target's FP8 projections and the global
        # custom-AllReduce policy must not select this model-local backend.
        self.modeling_gather = _get_symm_mem().TorchSymmMemCommunicator(group, device)
        self.gather = create_custom_all_gather(
            group,
            device,
            max_m=self.max_tokens,
            k=self.hidden_size,
            fp8=False,
        )
        self.scatter = create_push_reduce_scatter(
            group,
            device,
            max_m=self.max_tokens,
            n=self.hidden_size,
        )
        self.output = (
            torch.empty(
                (self.max_tokens // self.tp_size, self.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            if self.scatter is not None
            else None
        )
        logging.info(
            "[K3_MTP_BF16_TP] TP%d max_tokens=%d hidden=%d AG=%s RS=%s modeling_AG=%s",
            self.tp_size,
            self.max_tokens,
            self.hidden_size,
            "custom_staging" if self.gather is not None else "nccl",
            "push" if self.scatter is not None else "nccl",
            "nccl" if self.modeling_gather.disabled else "bf16_multimem",
        )

    def all_gather_modeling_boundary(self, local_input):
        if self.modeling_gather.should_torch_symm_mem_allgather(local_input):
            gathered = self.modeling_gather.all_gather(local_input)
            if gathered is not None:
                return gathered.view(
                    self.tp_size * local_input.shape[0], *local_input.shape[1:]
                )
        # Long prompt initialization and unsupported multicast keep the
        # existing collective. The physical token/request layout is retained.
        return all_gather(local_input, Group.TP)

    def all_gather(self, local_input):
        physical_tokens = local_input.shape[0] * self.tp_size
        if self.gather is None or physical_tokens > self.max_tokens:
            return all_gather(local_input, Group.TP)
        if (
            local_input.dtype != torch.bfloat16
            or local_input.shape[1] != self.hidden_size
        ):
            raise ValueError("MTP staging AG requires BF16 hidden states")
        return self.gather.all_gather(local_input)

    def reduce_scatter(self, partial):
        if self.scatter is None or partial.shape[0] > self.max_tokens:
            return reduce_scatter_bf16(partial, Group.TP)
        if (
            partial.dtype != torch.bfloat16
            or partial.shape[1] != self.hidden_size
            or partial.shape[0] % self.tp_size
        ):
            raise ValueError("MTP Push RS requires request-padded BF16 projections")
        output = self.output[: partial.shape[0] // self.tp_size]
        self.scatter.reduce_scatter(partial.contiguous(), output)
        return output


# MLA Decode context-parallel communication

@lru_cache(maxsize=1)
def _get_mla_dcp_kernels():
    from rtp_llm.models_py.triton_kernels.mla.dcp import _pack_a2a, _combine_a2a

    return _pack_a2a, _combine_a2a


class MlaDcpCommunicator:
    """Communication state shared by serial MLA layers in one TP group.

    Q is contiguous [local_heads,T,D]; partial O/LSE are [T,all_heads,L]
    and [T,all_heads]. combine returns BF16 [local_heads,T,L]. LSE is base-2.
    The local MLA adapter supplies O=0/LSE=-inf for empty shards.
    """

    def __init__(self, local_heads, q_dim, latent_dim, dtype, device):
        self.group = get_process_group(Group.TP)
        self.size = torch.distributed.get_world_size(self.group)
        self.rank = torch.distributed.get_rank(self.group)
        self.local_heads = local_heads
        self.heads = local_heads * self.size
        self.q_dim = q_dim
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.device = torch.device(device)
        contract = (local_heads, q_dim, latent_dim, str(dtype))
        contracts = [None] * self.size
        torch.distributed.all_gather_object(contracts, contract, group=self.group)
        if any(value != contract for value in contracts):
            raise ValueError(f"MLA DCP initialization differs across TP ranks: {contracts}")
        if (
            dtype not in (torch.bfloat16, torch.float8_e4m3fn) or latent_dim % 2
            or min(local_heads, q_dim, latent_dim) <= 0
        ):
            raise ValueError("MLA DCP requires BF16/E4M3 Q and an even latent dimension")
        self.backend = "a2a"
        self._pack_a2a, self._combine_a2a = _get_mla_dcp_kernels()
        logging.info(
            "[MLA_DCP] backend=a2a tp=%d rank=%d world_rank=%d dp_rank=%d",
            self.size, self.rank, torch.distributed.get_rank(),
            torch.distributed.get_rank() // self.size,
        )

    def query_gather(self, local_q):
        _, tokens, _ = local_q.shape
        if (
            local_q.shape != (self.local_heads, tokens, self.q_dim)
            or not local_q.is_contiguous() or local_q.dtype != self.dtype
            or local_q.device != self.device
        ):
            raise ValueError("MLA DCP Q must be contiguous [local_heads,T,D] on the configured device")
        output = local_q.new_empty(self.heads, tokens, self.q_dim)
        # NCCL copies FP8 codes without needing FP8 reduction support.
        wire_dtype = torch.uint8 if self.dtype == torch.float8_e4m3fn else self.dtype
        all_gather_into(local_q.view(wire_dtype), output.view(wire_dtype), Group.TP)
        return output

    def combine(self, partial_o, partial_lse, local_seq_lens):
        import triton

        tokens, heads, dim = partial_o.shape
        if (
            heads != self.heads or dim != self.latent_dim
            or partial_o.dtype != torch.bfloat16 or not partial_o.is_contiguous()
            or partial_lse.shape != (tokens, heads) or partial_lse.dtype != torch.float32
            or not partial_lse.is_contiguous() or local_seq_lens.numel() != tokens
            or not local_seq_lens.is_contiguous() or local_seq_lens.dtype != torch.int32
            or not (partial_o.device == partial_lse.device == local_seq_lens.device == self.device)
        ):
            raise ValueError("MLA DCP combine requires BF16 O[T,H,L], FP32 LSE[T,H] and int32 lengths[T]")
        shape = (self.size, tokens, self.local_heads, dim + 2)
        packed = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        self._pack_a2a[(tokens, heads)](
            partial_o, partial_lse, local_seq_lens, packed, packed.view(torch.float32),
            tokens, heads=heads, local_heads=self.local_heads, dim=dim,
            block=triton.next_power_of_2(dim),
        )
        received = all_to_all_single(packed, Group.TP)
        output = partial_o.new_empty(self.local_heads, tokens, dim)
        self._combine_a2a[(tokens, self.local_heads)](
            received, received.view(torch.float32), output, tokens,
            local_heads=self.local_heads, dim=dim, cp_size=self.size,
            block=triton.next_power_of_2(dim),
        )
        return output


_communicators = {}


def _communicator_key(config, device, dtype):
    return (get_process_group(Group.TP), device, config.head_num,
            config.kv_lora_rank, config.rope_head_dim, dtype)


def get_mla_dcp(config, device, dtype):
    key = _communicator_key(config, device, dtype)
    if key not in _communicators:
        _communicators[key] = MlaDcpCommunicator(
            config.head_num, config.kv_lora_rank + config.rope_head_dim,
            config.kv_lora_rank, dtype, device,
        )
    return _communicators[key]


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
    "all_gather_into",
    "all_to_all_single",
    "get_process_group",
    "reduce_scatter",
    "barrier",
    "reduce_scatter_bf16",
    "create_custom_all_gather",
    "CustomAllGather",
    "create_push_reduce_scatter",
    "PushReduceScatter",
    "is_same_host_group",
    "Fp8Activation",
    "MlaDcpCommunicator",
    "get_mla_dcp",
    "KimiK3MtpBf16Collectives",
]
