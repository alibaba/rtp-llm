from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed

from rtp_llm.models_py.distributed import rocm_rccl
from rtp_llm.models_py.distributed.symm_mem import (
    destroy_symm_mem_communicator,
    get_symm_mem_communicator,
    init_symm_mem_communicator,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig

# ParallelMode enum values matching C++ rtp_llm::ParallelMode in OpData.h
_CPP_PARALLEL_MODE_TP = 0
_CPP_PARALLEL_MODE_DP = 1
_CPP_PARALLEL_MODE_DP_AND_TP = 2
GroupPurpose = Literal["world", "tp", "dp", "graph_control"]


class Group(Enum):
    """Process group types for collective operations"""

    DP = "DP"
    TP = "TP"
    DP_AND_TP = "DP_AND_TP"


@dataclass(frozen=True)
class GroupRecord:
    process_group: torch.distributed.ProcessGroup
    ranks: Tuple[int, ...]
    backend: str
    device_index: Optional[int]
    owned_by_rtp: bool
    purpose: GroupPurpose
    generation: int


# Canonical process-group registry. Keys can be Group enums or strings for
# topology-specific groups; the ProcessGroup is always read from its record.
_group_records: Dict[Union[Group, str], GroupRecord] = {}
_parallelism_config: Optional[ParallelismConfig] = None
_initialized: bool = False  # Track if we've initialized (to prevent double init)
_distributed_generation: int = 1
_world_owned_by_rtp: bool = False
_owned_group_creation_order: List[torch.distributed.ProcessGroup] = []
_teardown_failed: bool = False
_data_group_timeout = timedelta(days=36500)
_DEFAULT_GRAPH_CONTROL_TIMEOUT = timedelta(seconds=300)
_graph_required_initialized: bool = False


class GroupRegistryAdapter:
    """Narrow process-group registry boundary used by platform integrations."""

    def get(self, key: Union[Group, str]) -> Optional[GroupRecord]:
        return _group_records.get(key)

    def create(
        self,
        ranks: List[int],
        backend: str,
        timeout: timedelta,
        device_index: Optional[int],
        graph_required: bool,
    ) -> torch.distributed.ProcessGroup:
        return _new_group(ranks, backend, timeout, device_index, graph_required)

    def record(self, key: Union[Group, str], record: GroupRecord) -> None:
        _record_group(key, record)


_group_registry = GroupRegistryAdapter()


def _record_group(key: Union[Group, str], record: GroupRecord) -> None:
    _group_records[key] = record
    if (
        record.owned_by_rtp
        and record.purpose != "world"
        and record.process_group is not torch.distributed.group.WORLD
        and all(pg is not record.process_group for pg in _owned_group_creation_order)
    ):
        _owned_group_creation_order.append(record.process_group)


def _new_group(
    ranks: List[int],
    backend: str,
    timeout: timedelta,
    device_index: Optional[int],
    graph_required: bool,
) -> torch.distributed.ProcessGroup:
    kwargs = {"ranks": ranks, "backend": backend, "timeout": timeout}
    device_id = rocm_rccl.process_group_device_id(backend, device_index, graph_required)
    if device_id is not None:
        kwargs["device_id"] = device_id
    return torch.distributed.new_group(**kwargs)


def _init_world_process_group(
    backend: str,
    init_method: str,
    world_size: int,
    rank: int,
    timeout: timedelta,
    local_rank: int,
    graph_required: bool,
) -> None:
    kwargs = {
        "backend": backend,
        "init_method": init_method,
        "world_size": world_size,
        "rank": rank,
        "timeout": timeout,
    }
    device_id = rocm_rccl.process_group_device_id(backend, local_rank, graph_required)
    if device_id is not None:
        kwargs["device_id"] = device_id
    torch.distributed.init_process_group(**kwargs)


def _validate_graph_required_across_ranks(
    graph_required: bool,
    world_size: int,
    graph_initialized: bool = False,
) -> None:
    # All ROCm ranks participate even when graph_required is false so mixed
    # requests fail explicitly instead of hanging a subset during graph setup.
    # Reuse WORLD rather than creating a temporary gloo group: this keeps the
    # check symmetric without adding a backend dependency to non-graph jobs.
    if not rocm_rccl.is_rocm_runtime():
        return
    actual_world_size = torch.distributed.get_world_size()
    world_group = torch.distributed.group.WORLD
    world_backend = str(torch.distributed.get_backend(world_group))
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if world_backend == "nccl"
        else torch.device("cpu")
    )
    local = torch.tensor(
        [int(graph_required), int(world_size), int(graph_initialized)],
        dtype=torch.int64,
        device=device,
    )
    minimum = local.clone()
    maximum = local.clone()
    logging.info(
        "Validating ROCm graph requirement across WORLD backend=%s world_size=%d",
        world_backend,
        actual_world_size,
    )
    torch.distributed.all_reduce(
        minimum, op=torch.distributed.ReduceOp.MIN, group=world_group
    )
    torch.distributed.all_reduce(
        maximum, op=torch.distributed.ReduceOp.MAX, group=world_group
    )
    minimum_values = tuple(int(value) for value in minimum.cpu().tolist())
    maximum_values = tuple(int(value) for value in maximum.cpu().tolist())
    expected = (
        int(graph_required),
        int(actual_world_size),
        int(graph_initialized),
    )
    if minimum_values != maximum_values or minimum_values != expected:
        raise RuntimeError(
            "graph_required and world_size must be identical on every rank and "
            "match the initialized world, "
            f"min={minimum_values}, max={maximum_values}, expected={expected}"
        )


def init_distributed_environment(
    parallelism_config: ParallelismConfig,
    nccl_comm_config: NcclCommConfig,
    nccl_init_port: int,
    backend: str = "nccl",
    timeout: Optional[int] = None,
    graph_required: bool = False,
):
    """Initialize distributed environment and create process groups.

    This function creates DP, TP, and DP_AND_TP process groups using torch.distributed.
    It can only be called once unless destroy_distributed_environment() has been called.

    Args:
        parallelism_config: Configuration for parallelism setup (sizes, ranks, etc.)
        nccl_comm_config: NCCL config with nccl_ip (and other ports for C++ init).
        nccl_init_port: Port for torch.distributed init_process_group (tcp://ip:port).
        backend: Distributed backend (default: "nccl")
        timeout: Existing distributed service timeout setting. It does not
            change the effectively unbounded timeout used by long-lived NCCL
            data groups. ROCm graph control groups use a bounded five-minute
            timeout independently.
        graph_required: Whether ROCm HIPGraph communication must be configured
            and prepared before graph-runner construction.

    Raises:
        RuntimeError: If already initialized and not destroyed
    """
    global _parallelism_config, _initialized, _world_owned_by_rtp
    global _graph_required_initialized, _teardown_failed

    if _teardown_failed:
        raise RuntimeError(
            "Distributed environment teardown previously failed; retry teardown "
            "or rebuild the process before reinitializing"
        )

    graph_requested = bool(graph_required)
    graph_required = bool(
        graph_requested
        and rocm_rccl.is_rocm_runtime()
        and parallelism_config.tp_size > 1
    )
    graph_reason = (
        "enabled"
        if graph_required
        else (
            "not_requested"
            if not graph_requested
            else "not_rocm" if not rocm_rccl.is_rocm_runtime() else "tp_size_one"
        )
    )
    logging.info(
        "Resolved distributed graph communication requested=%s effective=%s "
        "reason=%s tp_size=%d world_size=%d",
        graph_requested,
        graph_required,
        graph_reason,
        parallelism_config.tp_size,
        parallelism_config.world_size,
    )
    # Preserve the historical data-plane policy: workers may legitimately wait
    # for rank 0 (for example during long model loading or prefill) rather than
    # aborting through the ProcessGroup watchdog. The service timeout is not a
    # data-plane timeout. Graph control groups have their own bounded timeout.
    group_timeout = _data_group_timeout
    # Check if already initialized (and not destroyed)
    if _initialized and torch.distributed.is_initialized():
        logging.warning(
            "Distributed environment already initialized, skipping initialization"
        )
        if not _group_records:
            raise RuntimeError(
                "Distributed environment is marked initialized but its process-group "
                "registry is empty; destroy and reinitialize the environment"
            )
        _validate_graph_required_across_ranks(
            graph_required,
            parallelism_config.world_size,
            _graph_required_initialized,
        )
        if graph_required and not _graph_required_initialized:
            raise RuntimeError(
                "Cannot enable graph_required on an already initialized distributed "
                "environment; destroy it and reinitialize with graph_required=True"
            )
        if _graph_required_initialized and not graph_required:
            raise RuntimeError(
                "Cannot disable graph_required on an already initialized distributed "
                "environment; destroy it and reinitialize with graph_required=False"
            )
        if graph_required:
            prepare_graph_communication()
        return

    assert backend in ["nccl"], "backend current only supports nccl"
    ip = nccl_comm_config.nccl_ip
    port = nccl_init_port
    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    local_rank = parallelism_config.local_rank
    rocm_rccl.prepare_distributed_environment(parallelism_config, graph_required)
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"

    # If torch.distributed is already initialized (e.g., by external code),
    # we still need to create our process groups
    if torch.distributed.is_initialized():
        logging.info("torch.distributed already initialized, creating process groups")
        _validate_graph_required_across_ranks(graph_required, world_size)
        world_record = GroupRecord(
            process_group=torch.distributed.group.WORLD,
            ranks=tuple(range(world_size)),
            backend=str(torch.distributed.get_backend()),
            device_index=local_rank,
            owned_by_rtp=_world_owned_by_rtp,
            purpose="world",
            generation=_distributed_generation,
        )
        _record_group(Group.DP_AND_TP, world_record)
        _create_process_groups(
            parallelism_config,
            backend,
            group_timeout,
            external_world=True,
            graph_required=graph_required,
        )
        _parallelism_config = parallelism_config
        if graph_required:
            try:
                _prepare_graph_communication_unchecked()
            except Exception:
                _teardown_failed = True
                raise
        _graph_required_initialized = bool(graph_required)
        _register_process_groups_to_cpp()
        _initialized = True
        return

    logging.info(
        f"[rank: {world_rank}] initialize process_group: {ip}:{port}, rank: {world_rank}, world_size: {world_size}, "
        f"local_rank: {local_rank}, backend: {backend}, timeout: {timeout}",
    )

    # DP_AND_TP (global group) - initialized via init_process_group
    _init_world_process_group(
        backend=backend,
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=world_rank,
        timeout=group_timeout,
        local_rank=local_rank,
        graph_required=graph_required,
    )
    # Creation success immediately establishes ownership, including failure
    # paths in the following barrier/consensus checks.
    _world_owned_by_rtp = True
    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    _validate_graph_required_across_ranks(graph_required, world_size)
    world_record = GroupRecord(
        process_group=torch.distributed.group.WORLD,
        ranks=tuple(range(world_size)),
        backend=backend,
        device_index=local_rank,
        owned_by_rtp=True,
        purpose="world",
        generation=_distributed_generation,
    )
    _record_group(Group.DP_AND_TP, world_record)
    logging.info(
        f"[rank: {world_rank}] Created DP_AND_TP group {torch.distributed.group.WORLD} with ranks: {list(range(world_size))}"
    )

    # Create DP and TP groups
    _create_process_groups(
        parallelism_config,
        backend,
        group_timeout,
        external_world=False,
        graph_required=graph_required,
    )
    _parallelism_config = parallelism_config
    if graph_required:
        try:
            _prepare_graph_communication_unchecked()
        except Exception:
            _teardown_failed = True
            raise
    _graph_required_initialized = bool(graph_required)
    _register_process_groups_to_cpp()
    _initialized = True
    init_user_buffers_environment(parallelism_config)


def _create_process_groups(
    parallelism_config: ParallelismConfig,
    backend: str,
    timeout: Optional[timedelta],
    external_world: bool = False,
    graph_required: bool = False,
):
    """Create DP and TP process groups.

    Args:
        parallelism_config: Configuration for parallelism setup
        backend: Distributed backend
        timeout: Timeout for process group creation
    """
    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    tp_size = parallelism_config.tp_size
    dp_size = parallelism_config.dp_size
    device_index = parallelism_config.local_rank
    group_timeout = timeout if timeout is not None else _data_group_timeout

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
                dp_group = _new_group(
                    dp_ranks,
                    backend,
                    group_timeout,
                    device_index,
                    graph_required,
                )
                # Only store the group if this rank is part of it
                if world_rank in dp_ranks:
                    group_key = Group.DP.name + str(tp_rank_val)
                    record = GroupRecord(
                        process_group=dp_group,
                        ranks=tuple(dp_ranks),
                        backend=backend,
                        device_index=device_index,
                        owned_by_rtp=True,
                        purpose="dp",
                        generation=_distributed_generation,
                    )
                    _record_group(group_key, record)
                    _record_group(Group.DP, record)
                    logging.info(
                        f"[rank: {world_rank}] Stored DP group with key: {group_key} {dp_group} with ranks: {dp_ranks}"
                    )
                # All ranks must wait for group creation to complete
                torch.distributed.barrier()

    needs_isolated_external_tp = (
        external_world and graph_required and rocm_rccl.is_rocm_runtime()
    )
    if tp_size > 1 and (world_size != tp_size or needs_isolated_external_tp):
        # Create all TP groups - all ranks must participate in creating all TP groups
        # TP group: ranks with the same dp_rank (i.e., world_rank // tp_size)
        # There are dp_size TP groups (one for each dp_rank value)
        for dp_rank_val in range(dp_size):
            tp_ranks = [r for r in range(world_size) if r // tp_size == dp_rank_val]
            if len(tp_ranks) > 0:
                logging.info(
                    f"[rank: {world_rank}] Creating TP group for dp_rank {dp_rank_val} with ranks: {tp_ranks}"
                )
                tp_group = _new_group(
                    tp_ranks,
                    backend,
                    group_timeout,
                    device_index,
                    graph_required,
                )
                # Only store the group if this rank is part of it
                if world_rank in tp_ranks:
                    group_key = Group.TP.name + str(dp_rank_val)
                    record = GroupRecord(
                        process_group=tp_group,
                        ranks=tuple(tp_ranks),
                        backend=backend,
                        device_index=device_index,
                        owned_by_rtp=True,
                        purpose="tp",
                        generation=_distributed_generation,
                    )
                    _record_group(group_key, record)
                    _record_group(Group.TP, record)
                    logging.info(
                        f"[rank: {world_rank}] Stored TP group with key: {group_key} {tp_group} with ranks: {tp_ranks}"
                    )

                init_symm_mem_communicator(tp_group)

                # All ranks must wait for group creation to complete
                torch.distributed.barrier()
    elif tp_size > 1 and world_size == tp_size:
        # Single TP group: WORLD is the TP group, init symm_mem for it
        world_record = _group_records[Group.DP_AND_TP]
        tp_record = GroupRecord(
            process_group=world_record.process_group,
            ranks=world_record.ranks,
            backend=world_record.backend,
            device_index=world_record.device_index,
            owned_by_rtp=world_record.owned_by_rtp,
            purpose="tp",
            generation=world_record.generation,
        )
        _record_group(Group.TP, tp_record)
        init_symm_mem_communicator(tp_record.process_group)

    if tp_size == 1:
        _record_group(Group.TP, _group_records[Group.DP_AND_TP])
    if dp_size == 1 or world_size == dp_size:
        _record_group(Group.DP, _group_records[Group.DP_AND_TP])


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

    # Build mode -> process_group mapping from GroupRecord.purpose, which is
    # the canonical semantic source. Registry keys only provide lookup aliases.
    mode_to_group: Dict[int, torch.distributed.ProcessGroup] = {}
    if _parallelism_config is None:
        return
    world_rank = torch.distributed.get_rank()

    def local_record(purpose: GroupPurpose) -> Optional[GroupRecord]:
        for record in _group_records.values():
            if record.purpose == purpose and world_rank in record.ranks:
                return record
        return None

    world_record = local_record("world")
    if world_record is not None:
        mode_to_group[_CPP_PARALLEL_MODE_DP_AND_TP] = world_record.process_group

    if _parallelism_config.tp_size > 1:
        tp_record = local_record("tp")
        if (
            tp_record is None
            and _parallelism_config.world_size == _parallelism_config.tp_size
        ):
            tp_record = world_record
        if tp_record is not None:
            mode_to_group[_CPP_PARALLEL_MODE_TP] = tp_record.process_group

    if _parallelism_config.dp_size > 1:
        dp_record = local_record("dp")
        if (
            dp_record is None
            and _parallelism_config.world_size == _parallelism_config.dp_size
        ):
            dp_record = world_record
        if dp_record is not None:
            mode_to_group[_CPP_PARALLEL_MODE_DP] = dp_record.process_group

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
        device_id = torch.cuda.current_device()
        rank = pg.rank()
        world_size = pg.size()
        for i, recv_buf in enumerate(recv_buffers):
            data_num = recv_buf.numel() // world_size
            recv_on_cpu = not recv_buf.is_cuda
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
                send_t = send_buffers[i]
                send_tensor, _ = _ensure_cuda(send_t, device_id)
            torch.distributed.all_gather_into_tensor(
                gpu_recv_flat, send_tensor, group=pg
            )
            if recv_on_cpu:
                recv_buf.copy_(gpu_recv)

    librtp_compute_ops.register_comm_ops(cpp_broadcast, cpp_allreduce, cpp_allgather)
    logging.info(
        f"Registered C++ comm ops callbacks (modes: {list(mode_to_group.keys())})"
    )


def distributed_environment_initialized() -> bool:
    """Check if distributed environment is initialized.

    Returns:
        True if distributed environment is initialized, False otherwise
    """
    return _initialized and torch.distributed.is_initialized() and not _teardown_failed


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

    All graph runners/model instances must be destroyed before this function is
    called. A teardown failure is terminal for initialization in this process:
    callers may retry teardown, but must not reinitialize until teardown has
    completed successfully (or the process has been rebuilt).
    """
    global _group_records, _parallelism_config, _initialized
    global _distributed_generation, _world_owned_by_rtp, _owned_group_creation_order
    global _graph_required_initialized, _teardown_failed

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    logging.info(f"[rank: {rank}] Destroying distributed environment")

    # Validate graph ownership before any irreversible teardown side effect.
    rocm_rccl.assert_graph_comm_can_shutdown()

    from rtp_llm.models_py.utils.arch import is_cuda

    if is_cuda():
        try:
            from rtp_llm.models_py.distributed.user_buffers import (
                destroy_user_buffers_communicator,
            )

            destroy_user_buffers_communicator()
        except Exception:
            logging.exception("Failed to destroy user-buffers communicator")

    try:
        import librtp_compute_ops

        if hasattr(librtp_compute_ops, "clear_comm_ops"):
            try:
                librtp_compute_ops.clear_comm_ops()
            except Exception:
                logging.exception("Failed to clear registered C++ communication ops")
    except ImportError:
        pass

    # This can fail before local IPC exports are released. Preserve all
    # registries in that case so the caller can retry teardown safely.
    try:
        rocm_rccl.shutdown_graph_comm()
        destroy_symm_mem_communicator()
    except Exception:
        _initialized = False
        _teardown_failed = True
        logging.exception(
            "Distributed communication teardown failed; initialization is disabled"
        )
        raise

    destroy_errors = []
    if torch.distributed.is_initialized():
        for process_group in list(reversed(_owned_group_creation_order)):
            try:
                torch.distributed.destroy_process_group(process_group)
            except Exception as exc:
                destroy_errors.append((process_group, exc))
                logging.exception("Failed to destroy an RTP-owned process group")
            else:
                _owned_group_creation_order[:] = [
                    pg for pg in _owned_group_creation_order if pg is not process_group
                ]
                for key, record in list(_group_records.items()):
                    if record.process_group is process_group:
                        del _group_records[key]
        if _world_owned_by_rtp and not destroy_errors:
            try:
                torch.distributed.destroy_process_group()
            except Exception as exc:
                destroy_errors.append((torch.distributed.group.WORLD, exc))
                logging.exception("Failed to destroy RTP-owned world process group")

    if destroy_errors:
        _initialized = False
        _teardown_failed = True
        failures = "; ".join(
            f"{process_group!r}: {error}" for process_group, error in destroy_errors
        )
        raise RuntimeError(
            "Distributed process-group teardown failed and the environment is "
            f"no longer usable; retry teardown or rebuild the process: {failures}"
        ) from destroy_errors[0][1]

    _group_records.clear()
    _owned_group_creation_order.clear()
    _parallelism_config = None
    _initialized = False
    _world_owned_by_rtp = False
    _graph_required_initialized = False
    _teardown_failed = False
    _distributed_generation += 1
    gc.collect()
    logging.info(f"[rank: {rank}] Distributed environment destroyed")


def _get_group(group: Group) -> torch.distributed.ProcessGroup:
    """Get process group for the specified group type.

    This function checks if the distributed environment is initialized.
    Args:
        group: Group type (DP, TP, or DP_AND_TP)

    Returns:
        Process group for the specified group type

    Raises:
        RuntimeError: If distributed environment is not initialized
        ValueError: If group type is invalid
    """
    global _parallelism_config, _initialized

    if not torch.distributed.is_initialized() or not _initialized:
        raise RuntimeError(
            "Distributed environment is not initialized. Call "
            "init_distributed_environment(...) with the required rendezvous "
            "configuration before using collectives."
        )

    if group not in _group_records:
        raise ValueError(
            f"Process group {group} not found. Make sure init_distributed_environment() was called."
        )

    return _group_records[group].process_group


def _get_group_record(group: Group) -> GroupRecord:
    _get_group(group)  # preserves the initialization/error behavior above
    return _group_records[group]


def _prepare_graph_communication_unchecked() -> None:
    if _parallelism_config is None:
        raise RuntimeError(
            "Cannot prepare graph communication before the distributed environment"
        )
    rocm_rccl.prepare_rocm_graph_communication(
        parallelism_config=_parallelism_config,
        tp=_group_records[Group.TP],
        registry=_group_registry,
        group_timeout=_DEFAULT_GRAPH_CONTROL_TIMEOUT,
    )


def prepare_graph_communication() -> None:
    """Collectively prepare graph communication for an initialized graph topology.

    Every rank in the TP group must call this function symmetrically. Production
    initialization prepares communication eagerly; this entry point is retained
    only for idempotent validation and cannot upgrade a non-graph environment.
    """
    if not _initialized or not _graph_required_initialized:
        raise RuntimeError(
            "Graph communication was not requested during distributed initialization"
        )
    _prepare_graph_communication_unchecked()


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
    captured = rocm_rccl.try_capture_all_reduce(
        tensor, group == Group.TP, lambda: _get_group(group)
    )
    if captured is not None:
        return captured

    process_group = _get_group(group)
    if group == Group.TP:
        symm_mem_comm = get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allreduce(
            tensor
        ):
            return symm_mem_comm.all_reduce(tensor)

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
    captured = rocm_rccl.try_capture_all_gather(tensor, group == Group.TP)
    if captured is not None:
        return captured

    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)
    rocm_rccl.record_eager_allgather_signature(tensor, group == Group.TP, world_size)

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


__all__ = [
    "Group",
    "GroupRecord",
    "init_distributed_environment",
    "init_user_buffers_environment",
    "distributed_environment_initialized",
    "destroy_distributed_environment",
    "prepare_graph_communication",
    "send",
    "recv",
    "broadcast",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "barrier",
]
