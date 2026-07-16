from __future__ import annotations

import gc
import hashlib
import logging
import os
import re
import secrets
import stat
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
_CPU_TP_BROADCASTER_DISABLE_ENV = "RTP_LLM_CPU_TP_BROADCASTER_DISABLE"
_CPU_TP_BROADCASTER_DIR_ENV = "RTP_LLM_CPU_TP_BROADCASTER_DIR"
_CPU_TP_BROADCASTER_ID_ENV = "RTP_LLM_CPU_TP_BROADCASTER_ID"


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
_cpu_tp_broadcaster_nccl_init_port: Optional[int] = None
_cpu_tp_broadcaster_nccl_master_addr: Optional[str] = None


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() in ("1", "true", "yes", "on")


def _should_init_cpu_tp_broadcaster(
    parallelism_config: ParallelismConfig,
) -> bool:
    if _env_flag_enabled(_CPU_TP_BROADCASTER_DISABLE_ENV):
        return False
    tp_size = parallelism_config.tp_size
    local_world_size = parallelism_config.local_world_size
    if tp_size <= 1 or local_world_size <= 0:
        return False
    expected_tp_rank = parallelism_config.world_rank % tp_size
    expected_dp_rank = parallelism_config.world_rank // tp_size
    if (parallelism_config.tp_rank, parallelism_config.dp_rank) != (
        expected_tp_rank,
        expected_dp_rank,
    ):
        return False
    if parallelism_config.local_rank != (
        parallelism_config.world_rank % local_world_size
    ):
        return False
    # UDS cannot cross nodes. Under the TP-innermost contiguous rank layout
    # used by _create_process_groups, every host block must align to tp_size.
    return tp_size <= local_world_size and local_world_size % tp_size == 0


def _should_init_cpu_tp_broadcaster_for_group(
    parallelism_config: ParallelismConfig,
) -> bool:
    local_enabled = _should_init_cpu_tp_broadcaster(parallelism_config)
    if parallelism_config.tp_size <= 1:
        return False
    if not torch.distributed.is_initialized() or not _initialized:
        return local_enabled

    try:
        tp_group = _get_group(Group.TP)
        group_size = tp_group.size()
        group_enabled: List[bool] = [False] * group_size
        torch.distributed.all_gather_object(
            group_enabled, bool(local_enabled), group=tp_group
        )
    except Exception as e:
        logging.warning(
            "Skip CpuTpBroadcaster init: failed to check TP group consistency: %s",
            e,
        )
        return False

    if all(group_enabled):
        return True
    if any(group_enabled):
        logging.warning(
            "Skip CpuTpBroadcaster init: inconsistent eligibility across TP group: %s",
            group_enabled,
        )
    return False


def _cpu_tp_broadcaster_initialized_for_group(actual_initialized: bool) -> bool:
    if _parallelism_config is None or _parallelism_config.tp_size <= 1:
        return actual_initialized
    if not torch.distributed.is_initialized() or not _initialized:
        return actual_initialized

    try:
        tp_group = _get_group(Group.TP)
        group_size = tp_group.size()
        group_initialized: List[bool] = [False] * group_size
        torch.distributed.all_gather_object(
            group_initialized, bool(actual_initialized), group=tp_group
        )
    except Exception as e:
        logging.warning(
            "Skip CpuTpBroadcaster init: failed to confirm TP group init: %s",
            e,
        )
        return False

    if all(group_initialized):
        return True
    if any(group_initialized):
        logging.warning(
            "Skip CpuTpBroadcaster init: inconsistent initialized state across TP group: %s",
            group_initialized,
        )
    return False


def _make_cpu_tp_broadcaster_session_id(
    parallelism_config: ParallelismConfig,
    nccl_init_port: int,
    nccl_master_addr: Optional[str],
) -> str:
    master_addr = nccl_master_addr or os.environ.get("MASTER_ADDR", "")
    raw_key = (
        f"master={master_addr}|port={nccl_init_port}|"
        f"world={parallelism_config.world_size}|tp={parallelism_config.tp_size}"
    )
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
    return (
        f"port{nccl_init_port}_w{parallelism_config.world_size}"
        f"_tp{parallelism_config.tp_size}_{digest}"
    )


def _make_cpu_tp_broadcaster_base_path(
    parallelism_config: ParallelismConfig,
    nccl_init_port: int,
    nccl_master_addr: Optional[str] = None,
) -> str:
    session_id = os.environ.get(_CPU_TP_BROADCASTER_ID_ENV)
    if not session_id:
        session_id = _make_cpu_tp_broadcaster_session_id(
            parallelism_config, nccl_init_port, nccl_master_addr
        )
    session_id = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)

    # The configured value is a parent directory.  Never change its
    # permissions; only the per-user child is managed by RTP-LLM.  The parent
    # chain must be controlled by root/current uid, and any shared writable
    # component must use sticky-bit rename protection.
    parent_dir = os.environ.get(_CPU_TP_BROADCASTER_DIR_ENV) or os.environ.get(
        "TMPDIR", "/tmp"
    )
    parent_dir = os.path.abspath(os.path.normpath(parent_dir))
    uid = os.geteuid()
    current_dir = parent_dir
    while True:
        current_dir_stat = os.lstat(current_dir)
        if not stat.S_ISDIR(current_dir_stat.st_mode):
            raise ValueError(
                f"CpuTpBroadcaster parent directory is not safe: {current_dir}"
            )
        if current_dir_stat.st_uid not in (0, uid):
            raise PermissionError(
                f"CpuTpBroadcaster parent directory is not trusted: {current_dir}"
            )
        writable_by_non_owner = current_dir_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        if writable_by_non_owner and not current_dir_stat.st_mode & stat.S_ISVTX:
            raise PermissionError(
                "CpuTpBroadcaster parent directory is writable without sticky bit: "
                f"{current_dir}"
            )
        next_dir = os.path.dirname(current_dir)
        if next_dir == current_dir:
            break
        current_dir = next_dir

    base_dir = os.path.join(parent_dir, f"rtp_llm_{uid}")
    try:
        os.mkdir(base_dir, mode=0o700)
    except FileExistsError:
        pass

    open_flags = os.O_RDONLY | os.O_DIRECTORY
    open_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        base_dir_fd = os.open(base_dir, open_flags)
    except OSError as e:
        raise ValueError(f"CpuTpBroadcaster directory is not safe: {base_dir}") from e
    try:
        base_dir_stat = os.fstat(base_dir_fd)
        base_dir_path_stat = os.lstat(base_dir)
        if not stat.S_ISDIR(base_dir_stat.st_mode):
            raise ValueError(f"CpuTpBroadcaster directory is not safe: {base_dir}")
        if (
            base_dir_stat.st_dev,
            base_dir_stat.st_ino,
        ) != (
            base_dir_path_stat.st_dev,
            base_dir_path_stat.st_ino,
        ):
            raise ValueError(f"CpuTpBroadcaster directory was replaced: {base_dir}")
        if base_dir_stat.st_uid != uid:
            raise PermissionError(
                f"CpuTpBroadcaster directory is not owned by current user: {base_dir}"
            )
        os.fchmod(base_dir_fd, 0o700)
    finally:
        os.close(base_dir_fd)
    base_path = os.path.join(
        base_dir, f"rtp_llm_tp_{session_id}_dp{parallelism_config.dp_rank}"
    )
    max_rank_path = f"{base_path}_{max(0, parallelism_config.tp_size - 1)}.sock"
    if len(os.fsencode(max_rank_path)) >= _UDS_SUN_PATH_LIMIT:
        raise ValueError(
            f"CpuTpBroadcaster UDS path too long ({len(os.fsencode(max_rank_path))} "
            f"bytes, limit {_UDS_SUN_PATH_LIMIT - 1}): {max_rank_path}"
        )
    return base_path


def _cpu_tp_broadcaster_preflight_for_group(
    base_path: Optional[str], local_error: Optional[str]
) -> bool:
    """Confirm every TP rank resolves one shared UDS directory before C++ init."""
    if _parallelism_config is None:
        return False
    if not torch.distributed.is_initialized() or not _initialized:
        if local_error is not None:
            logging.warning(
                "Failed to initialize CpuTpBroadcaster, fallback to NCCL broadcast: %s",
                local_error,
            )
        return base_path is not None and local_error is None

    try:
        tp_group = _get_group(Group.TP)
        group_size = tp_group.size()
    except Exception as e:
        logging.warning(
            "Skip CpuTpBroadcaster init: failed to access TP group for UDS preflight: %s",
            e,
        )
        return False
    tp_rank = _parallelism_config.tp_rank
    normalized_path = (
        os.path.realpath(os.path.abspath(base_path)) if base_path is not None else None
    )
    probe_path = None
    probe_token = None
    probe_created = False

    if tp_rank == 0 and normalized_path is not None and local_error is None:
        try:
            probe_token = secrets.token_hex(16)
            probe_path = os.path.join(
                os.path.dirname(normalized_path), f".rtp_llm_probe_{probe_token}"
            )
            probe_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            probe_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            probe_fd = os.open(probe_path, probe_flags, 0o600)
            probe_created = True
            try:
                probe_bytes = probe_token.encode("ascii")
                if os.write(probe_fd, probe_bytes) != len(probe_bytes):
                    raise OSError("short write while creating visibility probe")
            finally:
                os.close(probe_fd)
        except Exception as e:
            local_error = f"failed to create root visibility probe: {e}"

    try:
        local_candidate = {
            "rank": tp_rank,
            "base_path": normalized_path,
            "error": local_error,
            "probe_path": probe_path,
            "probe_token": probe_token,
        }
        candidates: List[object] = [None] * group_size
        torch.distributed.all_gather_object(candidates, local_candidate, group=tp_group)

        expected_ranks = set(range(group_size))
        candidate_ranks = {
            candidate.get("rank")
            for candidate in candidates
            if isinstance(candidate, dict)
        }
        candidate_paths = {
            candidate.get("base_path")
            for candidate in candidates
            if isinstance(candidate, dict)
        }
        candidate_errors = [
            candidate.get("error")
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("error")
        ]
        root_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("rank") == 0
        ]

        local_ready = (
            len(candidates) == group_size
            and candidate_ranks == expected_ranks
            and len(candidate_paths) == 1
            and None not in candidate_paths
            and not candidate_errors
            and len(root_candidates) == 1
        )
        local_reason = None
        if not local_ready:
            local_reason = (
                "inconsistent paths or local preparation failure: "
                f"paths={sorted(str(path) for path in candidate_paths)}, "
                f"errors={candidate_errors}"
            )
        else:
            root_candidate = root_candidates[0]
            root_probe_path = root_candidate.get("probe_path")
            root_probe_token = root_candidate.get("probe_token")
            if not isinstance(root_probe_path, str) or not isinstance(
                root_probe_token, str
            ):
                local_ready = False
                local_reason = "root visibility probe metadata is missing"
            else:
                try:
                    with open(root_probe_path, "rb") as probe_file:
                        observed_token = probe_file.read(128).decode("ascii")
                    if observed_token != root_probe_token:
                        raise ValueError("root visibility probe token mismatch")
                except Exception as e:
                    local_ready = False
                    local_reason = f"root visibility probe is not shared: {e}"

        local_result = {
            "rank": tp_rank,
            "ready": local_ready,
            "reason": local_reason,
        }
        results: List[object] = [None] * group_size
        torch.distributed.all_gather_object(results, local_result, group=tp_group)
        all_ready = (
            len(results) == group_size
            and {result.get("rank") for result in results if isinstance(result, dict)}
            == expected_ranks
            and all(
                isinstance(result, dict) and result.get("ready") is True
                for result in results
            )
        )
        if not all_ready:
            reasons = [
                result.get("reason")
                for result in results
                if isinstance(result, dict) and result.get("reason")
            ]
            logging.warning(
                "Skip CpuTpBroadcaster init: TP group UDS preflight failed: %s",
                reasons,
            )

        cleanup_ready = True
        cleanup_reason = None
        if tp_rank == 0 and probe_created and probe_path is not None:
            try:
                os.unlink(probe_path)
                probe_created = False
            except FileNotFoundError:
                probe_created = False
            except OSError as e:
                cleanup_ready = False
                cleanup_reason = f"failed to remove root visibility probe: {e}"
        local_cleanup = {
            "rank": tp_rank,
            "clean": cleanup_ready,
            "reason": cleanup_reason,
        }
        cleanup_results: List[object] = [None] * group_size
        torch.distributed.all_gather_object(
            cleanup_results, local_cleanup, group=tp_group
        )
        all_clean = (
            len(cleanup_results) == group_size
            and {
                result.get("rank")
                for result in cleanup_results
                if isinstance(result, dict)
            }
            == expected_ranks
            and all(
                isinstance(result, dict) and result.get("clean") is True
                for result in cleanup_results
            )
        )
        if not all_clean:
            cleanup_reasons = [
                result.get("reason")
                for result in cleanup_results
                if isinstance(result, dict) and result.get("reason")
            ]
            logging.warning(
                "Skip CpuTpBroadcaster init: TP group probe cleanup failed: %s",
                cleanup_reasons,
            )
        return all_ready and all_clean
    except Exception as e:
        logging.warning(
            "Skip CpuTpBroadcaster init: failed to run TP group UDS preflight: %s",
            e,
        )
        return False
    finally:
        if probe_created and probe_path is not None:
            try:
                os.unlink(probe_path)
            except FileNotFoundError:
                pass
            except OSError as e:
                logging.warning(
                    "Failed to remove CpuTpBroadcaster visibility probe %s: %s",
                    probe_path,
                    e,
                )


def _init_cpu_tp_broadcaster_if_needed(librtp_compute_ops) -> None:
    global _cpu_tp_broadcaster_base_path

    _cpu_tp_broadcaster_base_path = None
    if _parallelism_config is None:
        librtp_compute_ops.destroy_cpu_tp_broadcaster()
        return
    if not _should_init_cpu_tp_broadcaster_for_group(_parallelism_config):
        librtp_compute_ops.destroy_cpu_tp_broadcaster()
        return

    base_path = None
    try:
        librtp_compute_ops.destroy_cpu_tp_broadcaster()
        local_error = None
    except Exception as e:
        local_error = f"failed to reset old CpuTpBroadcaster state: {e}"
    actual_initialized = False
    if _cpu_tp_broadcaster_nccl_init_port is None:
        port_error = "nccl_init_port is unknown"
        local_error = f"{local_error}; {port_error}" if local_error else port_error
    else:
        try:
            base_path = _make_cpu_tp_broadcaster_base_path(
                _parallelism_config,
                _cpu_tp_broadcaster_nccl_init_port,
                _cpu_tp_broadcaster_nccl_master_addr,
            )
        except Exception as e:
            path_error = str(e)
            local_error = f"{local_error}; {path_error}" if local_error else path_error

    if not _cpu_tp_broadcaster_preflight_for_group(base_path, local_error):
        # This helper is also used for retries.  A failed preflight must not
        # leave an older C++ singleton active while Python falls back to NCCL.
        librtp_compute_ops.destroy_cpu_tp_broadcaster()
        return
    assert base_path is not None

    try:
        librtp_compute_ops.init_cpu_tp_broadcaster(
            _parallelism_config.tp_rank,
            _parallelism_config.tp_size,
            base_path,
        )
        actual_initialized = True
    except Exception as e:
        logging.warning(
            "Failed to initialize CpuTpBroadcaster, fallback to NCCL broadcast: %s",
            e,
        )
    # Runtime broadcasts must be all UDS or all NCCL across the TP group.
    if not _cpu_tp_broadcaster_initialized_for_group(actual_initialized):
        librtp_compute_ops.destroy_cpu_tp_broadcaster()
        return
    _cpu_tp_broadcaster_base_path = base_path
    logging.info(
        f"Initialized CpuTpBroadcaster (tp_rank={_parallelism_config.tp_rank}, "
        f"tp_size={_parallelism_config.tp_size}, base_path={base_path})"
    )


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
    global _cpu_tp_broadcaster_nccl_init_port, _cpu_tp_broadcaster_nccl_master_addr

    _cpu_tp_broadcaster_nccl_init_port = nccl_init_port
    _cpu_tp_broadcaster_nccl_master_addr = nccl_comm_config.nccl_ip

    # Check if already initialized (and not destroyed)
    if _initialized and torch.distributed.is_initialized():
        logging.warning(
            "Distributed environment already initialized, skipping initialization"
        )
        # Still need to create groups if they don't exist
        if not _group_map:
            _create_process_groups(parallelism_config, backend, timedelta(days=36500))
            _register_process_groups_to_cpp()
        if rocm_rccl.is_available_runtime() and parallelism_config.tp_size > 1:
            rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
        return

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

        The C++ execBroadcast contract relies on the regular c10d call:
        communication errors are raised here, and CUDA tensors keep PyTorch's
        stream ordering for later GPU consumers without a device-wide sync.
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

    _init_cpu_tp_broadcaster_if_needed(librtp_compute_ops)


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
    global _group_map, _parallelism_config, _initialized
    global _cpu_tp_broadcaster_base_path, _cpu_tp_broadcaster_nccl_init_port
    global _cpu_tp_broadcaster_nccl_master_addr

    rank = torch.distributed.get_rank()
    logging.info(f"[rank: {rank}] Destroying distributed environment")

    from rtp_llm.models_py.utils.arch import is_cuda

    if is_cuda():
        from rtp_llm.models_py.distributed.user_buffers import (
            destroy_user_buffers_communicator,
        )

        destroy_user_buffers_communicator()

    cleanup_error = None
    try:
        import librtp_compute_ops

        # Python and librtp_compute_ops are deployed as one build artifact.
        # Missing symbols indicate version skew and should fail loudly.
        librtp_compute_ops.clear_comm_ops()
        librtp_compute_ops.destroy_cpu_tp_broadcaster()
    except ImportError:
        pass
    except Exception as e:
        cleanup_error = e
    finally:
        # Always reset Python-side distributed state before surfacing skew.
        if rocm_rccl.is_available_runtime():
            rocm_rccl.destroy_capture_comm()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        _group_map.clear()
        logging.info(f"[rank: {rank}] Distributed environment destroyed")
        _parallelism_config = None
        _cpu_tp_broadcaster_base_path = None
        _cpu_tp_broadcaster_nccl_init_port = None
        _cpu_tp_broadcaster_nccl_master_addr = None
        _initialized = False
        gc.collect()
    if cleanup_error is not None:
        raise cleanup_error


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
        process_group = _get_group(group)
        return rocm_rccl.capture_all_gather(tensor, process_group)

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
    "init_distributed_environment",
    "init_user_buffers_environment",
    "distributed_environment_initialized",
    "destroy_distributed_environment",
    "send",
    "recv",
    "broadcast",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "barrier",
]
