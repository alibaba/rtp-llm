from __future__ import annotations

import gc
import logging
import os
import re
import socket
import stat
import struct
import threading
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed

from rtp_llm.ops import NcclCommConfig, ParallelismConfig

# ParallelMode enum values matching C++ rtp_llm::ParallelMode in OpData.h
_CPP_PARALLEL_MODE_TP = 0
_CPP_PARALLEL_MODE_DP = 1
_CPP_PARALLEL_MODE_DP_AND_TP = 2
_UDS_SUN_PATH_LIMIT = 108
# Dedicated communicator for the sleep-quiesce consensus all-reduce (see OpData.h
# ParallelMode::SLEEP_QUIESCE). Same rank set as DP_AND_TP but a separate NCCL comm.
_CPP_PARALLEL_MODE_SLEEP_QUIESCE = 6
# Bounded deadline for the one-shot SLEEP_QUIESCE group warmup at startup, so a peer that died
# during launch fails the warmup fast instead of hanging boot on the group's ~infinite timeout.
_SLEEP_QUIESCE_WARMUP_TIMEOUT_S = 30
_LEVEL3_PHASE_TIMEOUT_S = 300


class Group(Enum):
    """Process group types for collective operations"""

    DP = "DP"
    TP = "TP"
    DP_AND_TP = "DP_AND_TP"
    # Dedicated group carrying ONLY the async sleep-quiesce consensus all-reduce, so it
    # never interleaves with forward / EPLB collectives on DP_AND_TP. Created lazily and
    # only when sleep mode is enabled on a DP/EP deployment.
    SLEEP_QUIESCE = "SLEEP_QUIESCE"


@dataclass
class _CollectiveResource:
    name: str
    rebuild: Callable[[], None]
    teardown: Callable[[], None]
    active: bool = False


class _CollectiveLifecycleRegistry:
    """Order checkpoint-sensitive resources by their dependencies."""

    def __init__(self) -> None:
        self._resources: List[_CollectiveResource] = []
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        *,
        rebuild: Callable[[], None],
        teardown: Callable[[], None],
    ) -> None:
        with self._lock:
            if any(resource.name == name for resource in self._resources):
                raise ValueError(f"collective resource '{name}' is already registered")
            self._resources.append(_CollectiveResource(name, rebuild, teardown))

    def rebuild(self) -> None:
        """Rebuild dependencies first, resuming cleanly after a failed attempt."""
        with self._lock:
            if not self._resources:
                raise RuntimeError("no collective resources are registered")
            for resource in self._resources:
                if resource.active:
                    continue
                try:
                    resource.rebuild()
                except Exception as error:
                    try:
                        resource.teardown()
                    except Exception as cleanup_error:
                        raise RuntimeError(
                            f"failed to rebuild collective resource '{resource.name}' "
                            f"after {type(error).__name__}: {error}; "
                            f"cleanup also failed: {cleanup_error}"
                        ) from error
                    raise RuntimeError(
                        f"failed to rebuild collective resource '{resource.name}': "
                        f"{type(error).__name__}: {error}"
                    ) from error
                resource.active = True

    def teardown(self) -> None:
        """Destroy dependents first, resuming cleanly after a failed attempt."""
        with self._lock:
            for resource in reversed(self._resources):
                if not resource.active:
                    continue
                try:
                    resource.teardown()
                except Exception as error:
                    raise RuntimeError(
                        f"failed to teardown collective resource '{resource.name}': "
                        f"{type(error).__name__}: {error}"
                    ) from error
                resource.active = False

    def active_resources(self) -> List[str]:
        with self._lock:
            return [resource.name for resource in self._resources if resource.active]

    def reset(self) -> None:
        """Forget lifecycle state after a terminal destroy or in hermetic tests."""
        with self._lock:
            for resource in self._resources:
                resource.active = False


@dataclass(frozen=True)
class _DistributedInitSnapshot:
    parallelism_config: ParallelismConfig
    nccl_comm_config: NcclCommConfig
    nccl_init_port: int
    backend: str
    timeout: Optional[int]


# Global process group storage
# Key can be Group enum or string (for multiple DP/TP groups)
_group_map: Dict[Union[Group, str], torch.distributed.ProcessGroup] = {}
_parallelism_config: Optional[ParallelismConfig] = None
_initialized: bool = False  # Track if we've initialized (to prevent double init)
_cpu_tp_broadcaster_base_path: Optional[str] = None
_distributed_init_snapshot: Optional[_DistributedInitSnapshot] = None
_collective_lifecycle = _CollectiveLifecycleRegistry()
_collective_resources_registered = False
_lifecycle_store: Optional[Any] = None
_process_group_generation = 0
_rocm_rccl = None
_symm_mem = None


def _sleep_mode_level3_enabled() -> bool:
    return (
        os.environ.get("ENABLE_SLEEP_MODE", "0") == "1"
        and os.environ.get("SLEEP_MODE_LEVEL", "1") == "3"
    )


_SLEEP_INSTANCE_GENERATION_ENV = "RTP_LLM_SLEEP_INSTANCE_GENERATION"


def _sanitize_key_component(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value or ""))


def _level3_role_component(parallelism_config: ParallelismConfig) -> str:
    """Stable PD-role token for coordination keys (cross-rank consistent).

    Returns "" when no role is configured so the coordination keys keep their
    legacy (role-less) layout: this preserves backward compatibility with L1/L2
    deployments and only namespaces by role once a PD role is actually present.
    """
    role = getattr(parallelism_config, "role_type", None)
    if role is None:
        return ""
    # pybind RoleType enum -> its name (e.g. "PREFILL"); fall back to str().
    return _sanitize_key_component(getattr(role, "name", None) or str(role))


def _level3_key_namespace(parallelism_config: ParallelismConfig) -> str:
    """Namespace prefix for Level3 TCPStore coordination keys.

    Includes the PD role (always, cross-rank consistent from config) and, when the
    deployment publishes one, an instance-generation token shared identically
    across ranks. This prevents a shared/misconfigured TCPStore from colliding
    ready-gate keys across PD roles or across successive instance generations.
    """
    role = _level3_role_component(parallelism_config)
    generation = _sanitize_key_component(
        os.environ.get(_SLEEP_INSTANCE_GENERATION_ENV, "")
    )
    if generation:
        return f"{role}/{generation}"
    return role


_MULTICAST_KEEPER_ENABLE_ENV = "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER"
_MULTICAST_KEEPER_DIR_ENV = "NEKYIA_KEEPER_DIR"
_MULTICAST_KEEPER_SOCKET_ENV = "RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET"
_MULTICAST_KEEPER_SOCKET = "mcsk.sock"
_MULTICAST_SHIM_BASENAME = "mc_shim_unified.so"
_MULTICAST_PROTOCOL_MAGIC = 0x3250434D505452
_MULTICAST_PROTOCOL_VERSION = 3
_MULTICAST_OP_PING = 1
_MULTICAST_STATUS_OK = 0
_MULTICAST_REQUEST = struct.Struct("<QHHIQQQQIIQ")
_MULTICAST_RESPONSE = struct.Struct("<QHHIiIQQQQQIIQ")
_MULTICAST_KEEPER_PING_TIMEOUT_S = 2.0
_multicast_keeper_instance_lock = threading.RLock()
_multicast_keeper_pinned_epoch: Optional[int] = None
_multicast_keeper_pinned_instance: Optional[tuple[int, int]] = None


def _multicast_keeper_socket_path() -> tuple[Optional[str], str]:
    socket_path = os.environ.get(_MULTICAST_KEEPER_SOCKET_ENV, "").strip()
    if socket_path:
        return socket_path, ""
    keeper_dir = os.environ.get(_MULTICAST_KEEPER_DIR_ENV, "").strip()
    if not keeper_dir:
        return None, (
            f"neither {_MULTICAST_KEEPER_SOCKET_ENV} nor "
            f"{_MULTICAST_KEEPER_DIR_ENV} is set"
        )
    return os.path.join(keeper_dir, _MULTICAST_KEEPER_SOCKET), ""


def _ping_multicast_keeper(
    socket_path: str,
) -> tuple[bool, str, Optional[tuple[int, int]]]:
    request = _MULTICAST_REQUEST.pack(
        _MULTICAST_PROTOCOL_MAGIC,
        _MULTICAST_PROTOCOL_VERSION,
        _MULTICAST_OP_PING,
        _MULTICAST_REQUEST.size,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.settimeout(_MULTICAST_KEEPER_PING_TIMEOUT_S)
            client.connect(socket_path)
            client.sendall(request)
            response = client.recv(_MULTICAST_RESPONSE.size + 1)
    except OSError as error:
        return False, f"keeper ping failed for {socket_path}: {error}", None

    if len(response) != _MULTICAST_RESPONSE.size:
        return (
            False,
            "keeper ping returned an invalid response size: "
            f"{len(response)} (expected {_MULTICAST_RESPONSE.size})",
            None,
        )
    fields = _MULTICAST_RESPONSE.unpack(response)
    magic, version, opcode, struct_size, status = fields[:5]
    holder_instance_hi, holder_instance_lo = fields[6:8]
    if (
        magic != _MULTICAST_PROTOCOL_MAGIC
        or version != _MULTICAST_PROTOCOL_VERSION
        or opcode != _MULTICAST_OP_PING
        or struct_size != _MULTICAST_RESPONSE.size
        or status != _MULTICAST_STATUS_OK
        or (holder_instance_hi == 0 and holder_instance_lo == 0)
    ):
        return (
            False,
            "keeper ping returned an incompatible response: "
            f"magic=0x{magic:x} version={version} opcode={opcode} "
            f"size={struct_size} status={status}",
            None,
        )
    return True, "", (holder_instance_hi, holder_instance_lo)


def _multicast_keeper_ready() -> tuple[bool, str, Optional[tuple[int, int]]]:
    socket_path, path_error = _multicast_keeper_socket_path()
    if socket_path is None:
        return False, path_error, None
    try:
        if not stat.S_ISSOCK(os.stat(socket_path).st_mode):
            return False, f"keeper endpoint is not a UNIX socket: {socket_path}", None
    except OSError as error:
        return False, f"keeper socket is unavailable: {socket_path}: {error}", None

    ping_ok, ping_error, holder_instance = _ping_multicast_keeper(socket_path)
    if not ping_ok:
        return False, ping_error, None

    # LD_PRELOAD is inherited by multiprocessing-spawn ranks. Check the actual
    # mappings as well, because merely leaving a stale path in the environment
    # would otherwise advertise checkpoint-safe multicast without interposition.
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            if any(_MULTICAST_SHIM_BASENAME in line for line in maps):
                return True, "", holder_instance
    except OSError as error:
        return False, f"cannot inspect loaded multicast shim: {error}", None
    return False, f"{_MULTICAST_SHIM_BASENAME} is not loaded in this rank", None


def _publish_holder_instance_to_engine(
    holder_instance: Optional[tuple[int, int]]
) -> None:
    """Hand the pinned multicast keeper holder to the C++ engine.

    GetSleepStatus reports it in SleepStatusResponsePB.holder_instance so the durable
    checkpoint manifest can persist + fail-closed verify the holder. No-op unless the
    multicast keeper is enabled, so L1/L2 and non-keeper deployments are unaffected.
    Best-effort: the manifest-side re-verification is the authoritative gate, and the
    engine module may be absent under unit tests, so failures never break the barrier.
    """
    if os.environ.get(_MULTICAST_KEEPER_ENABLE_ENV, "0") != "1":
        return
    try:
        from libth_transformer import (
            clear_multicast_holder_instance,
            set_multicast_holder_instance,
        )
    except Exception as error:  # pragma: no cover - engine module optional in tests
        logging.debug("multicast holder engine setter unavailable: %s", error)
        return
    try:
        if holder_instance is None:
            clear_multicast_holder_instance()
        else:
            set_multicast_holder_instance(
                int(holder_instance[0]), int(holder_instance[1])
            )
    except Exception as error:  # pragma: no cover - best-effort
        logging.warning("failed to publish keeper holder to engine: %s", error)


def _validate_multicast_keeper_phase_instance(
    phase: str, sleep_epoch: int, holder_instance: tuple[int, int]
) -> tuple[bool, str]:
    global _multicast_keeper_pinned_epoch, _multicast_keeper_pinned_instance
    with _multicast_keeper_instance_lock:
        if phase == "collective_teardown_ready":
            if _multicast_keeper_pinned_instance is None:
                _multicast_keeper_pinned_epoch = sleep_epoch
                _multicast_keeper_pinned_instance = holder_instance
                # Publish the just-pinned holder to the engine so the coordinator's
                # GetSleepStatus poll reports it into the checkpoint manifest.
                _publish_holder_instance_to_engine(holder_instance)
                return True, ""
            if _multicast_keeper_pinned_epoch != sleep_epoch:
                return False, (
                    "multicast keeper instance is still pinned by incomplete epoch "
                    f"{_multicast_keeper_pinned_epoch}"
                )
            if _multicast_keeper_pinned_instance != holder_instance:
                return False, "multicast keeper was replaced during collective teardown"
            return True, ""

        if _multicast_keeper_pinned_instance is None:
            return False, "multicast keeper instance was not pinned for Level3 wake"
        if _multicast_keeper_pinned_epoch != sleep_epoch:
            return False, (
                "multicast keeper rebuild epoch does not match pinned epoch "
                f"{_multicast_keeper_pinned_epoch}"
            )
        if _multicast_keeper_pinned_instance != holder_instance:
            return False, "multicast keeper was replaced during Level3 wake"
        return True, ""


def _reset_multicast_keeper_phase_instance() -> None:
    global _multicast_keeper_pinned_epoch, _multicast_keeper_pinned_instance
    with _multicast_keeper_instance_lock:
        _multicast_keeper_pinned_epoch = None
        _multicast_keeper_pinned_instance = None
        # Drop the holder from the engine too so a stale value is not reported after
        # the checkpoint window closes.
        _publish_holder_instance_to_engine(None)


def get_pinned_multicast_holder_instance() -> Optional[tuple[int, int]]:
    """Return the multicast keeper holder pinned for the in-flight checkpoint.

    Exposed so the durable checkpoint manifest can persist and later re-verify the
    holder identity (fail-closed if it exits/changes). None when no keeper is
    pinned (keeper disabled, or outside a teardown->rebuild window).
    """
    with _multicast_keeper_instance_lock:
        return _multicast_keeper_pinned_instance


def _keeper_holder_store_key(namespace: str, sleep_epoch: int) -> str:
    scope = f"{namespace}/" if namespace else ""
    return f"rtp_llm_level3_keeper/{scope}{sleep_epoch}/holder"


def _publish_keeper_holder_instance(
    store: torch.distributed.Store,
    namespace: str,
    sleep_epoch: int,
    holder_instance: tuple[int, int],
) -> None:
    """Record the pinned keeper holder on the coordination store (best-effort).

    This puts the holder identity onto the existing Level3 coordination path so it
    is durable and observable per (role, generation, epoch). Failures never break
    the barrier: the manifest-side re-verification is the authoritative gate.
    """
    try:
        store.set(
            _keeper_holder_store_key(namespace, sleep_epoch),
            f"{holder_instance[0]}:{holder_instance[1]}".encode("utf-8"),
        )
    except Exception as error:  # pragma: no cover - best-effort publish
        logging.warning(
            "failed to publish multicast keeper holder for epoch %s: %s",
            sleep_epoch,
            error,
        )


def _configure_level3_multicast() -> None:
    """Select the fail-closed Level3 multicast policy before NCCL initializes."""
    if not _sleep_mode_level3_enabled():
        return

    keeper_enabled = os.environ.get(_MULTICAST_KEEPER_ENABLE_ENV, "0") == "1"
    if keeper_enabled:
        ready, reason, _ = _multicast_keeper_ready()
        if not ready:
            raise RuntimeError(
                "Sleep mode Level3 multicast keeper was requested but is not ready: "
                + reason
            )
        required = {
            "NCCL_NVLS_ENABLE": "1",
            "TORCH_SYMM_MEM_DISABLE_MULTICAST": "0",
        }
        policy = "enabled through the external CUDA-checkpoint multicast keeper"
    else:
        required = {
            "NCCL_NVLS_ENABLE": "0",
            "TORCH_SYMM_MEM_DISABLE_MULTICAST": "1",
        }
        policy = "disabled because no CUDA-checkpoint multicast keeper was requested"

    for name, value in required.items():
        configured = os.environ.get(name)
        if configured == value:
            continue
        os.environ[name] = value
        log = logging.info if configured is None else logging.warning
        log(
            "Sleep mode Level3 overrides %s=%s with %s before collective "
            "initialization; multicast is %s",
            name,
            configured,
            value,
            policy,
        )


def _enforce_level3_multicast_disabled() -> None:
    """Compatibility alias for callers/tests predating keeper support."""
    _configure_level3_multicast()


def _get_or_create_lifecycle_store(
    snapshot: _DistributedInitSnapshot,
) -> torch.distributed.Store:
    """Return a CPU-only rendezvous store that outlives every CUDA PG."""
    global _lifecycle_store

    if _lifecycle_store is not None:
        return _lifecycle_store

    parallelism_config = snapshot.parallelism_config
    timeout = timedelta(seconds=snapshot.timeout or 300)
    _lifecycle_store = torch.distributed.TCPStore(
        host_name=snapshot.nccl_comm_config.nccl_ip,
        port=snapshot.nccl_init_port,
        world_size=parallelism_config.world_size,
        is_master=parallelism_config.world_rank == 0,
        timeout=timeout,
        wait_for_workers=True,
        multi_tenant=True,
    )
    logging.info(
        "[rank: %s] created persistent process-group TCPStore at %s:%s",
        parallelism_config.world_rank,
        snapshot.nccl_comm_config.nccl_ip,
        snapshot.nccl_init_port,
    )
    return _lifecycle_store


def _wait_for_process_group_generation_ready(
    store: torch.distributed.Store,
    *,
    generation: int,
    world_rank: int,
    world_size: int,
    timeout: timedelta,
    namespace: str = "",
) -> None:
    """Wait until every rank has completed local restore for this generation."""
    scope = f"{namespace}/" if namespace else ""
    key_prefix = f"rtp_llm_pg_ready/{scope}{generation}"
    rank_key = f"{key_prefix}/rank/{world_rank}"
    ready_keys = [f"{key_prefix}/rank/{rank}" for rank in range(world_size)]
    try:
        store.set(rank_key, b"1")
        store.wait(ready_keys, timeout)
    except Exception as error:
        raise RuntimeError(
            f"process-group generation {generation} ready barrier failed for "
            f"rank {world_rank} of {world_size}: {type(error).__name__}: {error}"
        ) from error


def coordinate_level3_phase(
    phase: str,
    sleep_epoch: int,
    local_success: bool,
    *,
    timeout_s: Optional[int] = None,
) -> bool:
    """Return true only when every rank reports success for a Level3 phase.

    The TCPStore is CPU-only and intentionally survives process-group teardown,
    so this barrier is available both before checkpoint and after CUDA restore.
    Keys include the sleep epoch and phase to make retries idempotent without
    consuming results from an earlier sleep cycle.
    """
    if not phase or re.fullmatch(r"[A-Za-z0-9_.-]+", phase) is None:
        raise ValueError(f"invalid Level3 lifecycle phase: {phase!r}")
    if sleep_epoch <= 0:
        raise ValueError(f"invalid Level3 sleep epoch: {sleep_epoch}")

    keeper_enabled = (
        _sleep_mode_level3_enabled()
        and os.environ.get(_MULTICAST_KEEPER_ENABLE_ENV, "0") == "1"
    )
    holder_instance: Optional[tuple[int, int]] = None
    if (
        local_success
        and phase
        in {
            "collective_teardown_ready",
            "collective_rebuild_ready",
            "graph_recapture_ready",
            "graph_recapture_done",
        }
        and keeper_enabled
    ):
        keeper_ready, keeper_error, holder_instance = _multicast_keeper_ready()
        if keeper_ready:
            assert holder_instance is not None
            keeper_ready, keeper_error = _validate_multicast_keeper_phase_instance(
                phase, sleep_epoch, holder_instance
            )
        if not keeper_ready:
            logging.error(
                "Level3 phase %s epoch %s rejected locally: %s",
                phase,
                sleep_epoch,
                keeper_error,
            )
            local_success = False

    snapshot = _require_distributed_init_snapshot()
    parallelism_config = snapshot.parallelism_config
    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    store = _get_or_create_lifecycle_store(snapshot)
    timeout = timedelta(
        seconds=(
            timeout_s
            if timeout_s is not None
            else snapshot.timeout or _LEVEL3_PHASE_TIMEOUT_S
        )
    )
    namespace = _level3_key_namespace(parallelism_config)
    scope = f"{namespace}/" if namespace else ""
    key_prefix = f"rtp_llm_level3/{scope}{sleep_epoch}/{phase}"
    rank_key = f"{key_prefix}/rank/{world_rank}"
    rank_keys = [f"{key_prefix}/rank/{rank}" for rank in range(world_size)]

    try:
        store.set(rank_key, b"1" if local_success else b"0")
        store.wait(rank_keys, timeout)
        rank_results = [store.get(key) for key in rank_keys]
    except Exception as error:
        raise RuntimeError(
            f"Level3 phase {phase!r} epoch {sleep_epoch} coordination failed "
            f"for rank {world_rank} of {world_size}: "
            f"{type(error).__name__}: {error}"
        ) from error

    all_success = all(result == b"1" for result in rank_results)
    if not all_success:
        failed_ranks = [
            rank for rank, result in enumerate(rank_results) if result != b"1"
        ]
        logging.error(
            "Level3 phase %s epoch %s rejected by ranks %s",
            phase,
            sleep_epoch,
            failed_ranks,
        )
    elif phase == "graph_recapture_done" and keeper_enabled:
        _reset_multicast_keeper_phase_instance()
    elif (
        phase == "collective_teardown_ready"
        and keeper_enabled
        and holder_instance is not None
        and world_rank == 0
    ):
        # Publish the pinned holder onto the coordination path so the checkpoint
        # manifest can durably persist and later re-verify it (fail-closed).
        _publish_keeper_holder_instance(store, namespace, sleep_epoch, holder_instance)
    return all_success


def _get_rocm_rccl():
    """Import ROCm RCCL helpers only on ROCm runtime."""
    global _rocm_rccl
    if getattr(torch.version, "hip", None) is None:
        return None
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
    global _distributed_init_snapshot

    if backend != "nccl":
        raise ValueError(f"unsupported distributed backend: {backend}")
    _enforce_level3_multicast_disabled()
    _ensure_collective_resources_registered()
    if _initialized and _collective_lifecycle.active_resources():
        logging.warning(
            "Distributed environment already initialized, skipping initialization"
        )
        return
    if _distributed_init_snapshot is not None:
        raise RuntimeError(
            "distributed environment has a retained lifecycle snapshot; "
            "call rebuild_distributed_environment() or destroy_distributed_environment()"
        )

    _distributed_init_snapshot = _DistributedInitSnapshot(
        parallelism_config=parallelism_config,
        nccl_comm_config=nccl_comm_config,
        nccl_init_port=nccl_init_port,
        backend=backend,
        timeout=timeout,
    )
    try:
        _collective_lifecycle.rebuild()
    except Exception:
        # Keep the snapshot and successfully-created participants so a retry can
        # continue from the exact failed dependency.
        raise


def _rebuild_torch_process_groups() -> None:
    global _group_map, _parallelism_config, _initialized
    global _cpu_tp_broadcaster_base_path
    global _lifecycle_store, _process_group_generation

    _enforce_level3_multicast_disabled()
    snapshot = _require_distributed_init_snapshot()
    parallelism_config = snapshot.parallelism_config
    nccl_comm_config = snapshot.nccl_comm_config
    _normalize_parallelism_ranks(parallelism_config)
    _cpu_tp_broadcaster_base_path = _make_cpu_tp_broadcaster_base_path(
        parallelism_config, snapshot.nccl_init_port
    )

    ip = nccl_comm_config.nccl_ip
    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    local_rank = parallelism_config.local_rank
    infinite_timeout = timedelta(days=36500)
    init_timeout = timedelta(seconds=snapshot.timeout or 300)

    distributed_initialized = torch.distributed.is_initialized()
    store = None
    generation = None
    if not distributed_initialized:
        store = _get_or_create_lifecycle_store(snapshot)
        _process_group_generation += 1
        generation = _process_group_generation
        _wait_for_process_group_generation_ready(
            store,
            generation=generation,
            world_rank=world_rank,
            world_size=world_size,
            timeout=init_timeout,
            namespace=_level3_key_namespace(parallelism_config),
        )

    rocm_rccl = _get_rocm_rccl()
    if rocm_rccl is not None:
        rocm_rccl.configure_process_groups(parallelism_config)
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
    _group_map.clear()

    if not distributed_initialized:
        logging.info(
            f"[rank: {world_rank}] initialize process_group: {ip}:{snapshot.nccl_init_port}, "
            f"rank: {world_rank}, world_size: {world_size}, local_rank: {local_rank}, "
            f"backend: {snapshot.backend}, timeout: {snapshot.timeout}",
        )
        init_kwargs: Dict[str, Any] = {
            "backend": snapshot.backend,
            "world_size": world_size,
            "rank": world_rank,
            "timeout": init_timeout,
        }
        assert store is not None
        assert generation is not None
        prefix = f"rtp_llm_pg/{generation}"
        init_kwargs["store"] = torch.distributed.PrefixStore(prefix, store)
        if snapshot.backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        logging.info(
            "[rank: %s] initializing process-group generation %s with store prefix %s",
            world_rank,
            generation,
            prefix,
        )
        torch.distributed.init_process_group(**init_kwargs)
        torch.distributed.barrier(
            group=torch.distributed.group.WORLD, device_ids=[local_rank]
        )
    else:
        logging.info("torch.distributed already initialized, creating process groups")

    _group_map[Group.DP_AND_TP] = torch.distributed.group.WORLD
    _parallelism_config = parallelism_config
    _initialized = True
    _create_process_groups(parallelism_config, snapshot.backend, infinite_timeout)
    if rocm_rccl is not None and parallelism_config.tp_size > 1:
        rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))


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

                _get_symm_mem().init_symm_mem_communicator(tp_group)

                # All ranks must wait for group creation to complete
                torch.distributed.barrier()
    elif tp_size > 1 and world_size == tp_size:
        # Single TP group: WORLD is the TP group, init symm_mem for it
        _get_symm_mem().init_symm_mem_communicator(torch.distributed.group.WORLD)

    _maybe_create_sleep_quiesce_group(parallelism_config, backend)


def _sleep_quiesce_group_needed(parallelism_config: ParallelismConfig) -> bool:
    """Whether to build the dedicated sleep-quiesce communicator.

    Mirrors C++ NormalEngine::collectiveSleepQuiesceEnabled(): sleep mode enabled AND a
    multi-rank DP/EP deployment. Plain single-rank or pure-TP deployments quiesce without
    a per-step collective (releasePendingTpCollectiveForPause), so they need no extra comm.
    """
    world_size = parallelism_config.world_size
    dp_size = parallelism_config.dp_size
    ep_size = getattr(parallelism_config, "ep_size", 1) or 1
    if world_size <= 1 or not (dp_size > 1 or ep_size > 1):
        return False
    try:
        from rtp_llm.model_loader.weight_memory_saver import (
            is_enabled as _sleep_enabled,
        )

        return bool(_sleep_enabled())
    except (
        Exception
    ):  # pragma: no cover - defensive: never block group setup on this probe
        return os.environ.get("ENABLE_SLEEP_MODE", "0") == "1"


def _maybe_create_sleep_quiesce_group(
    parallelism_config: ParallelismConfig, backend: str
) -> None:
    """Create a dedicated CPU (gloo) group (all world ranks) for the sleep-quiesce consensus.

    new_group() is a collective, so every rank must call it; we gate on a launch-time
    condition (_sleep_quiesce_group_needed) that is identical on every rank, keeping the
    call rank-symmetric. The group spans the full world (same membership as DP_AND_TP)
    because the consensus needs every rank.

    The backend is GLOO (host), NOT nccl, on purpose. The consensus is a tiny 2-int
    control-plane all-reduce issued once per step during the sleep drain while the engine
    still co-steps a fake-decode forward to hold EP lockstep. If it ran on NCCL it would
    execute on a GPU stream concurrently with the fake-decode's DeepEP low-latency kernels,
    which launch full-grid persistent kernels that occupy every SM -- the tiny NCCL
    all-reduce then never gets an SM, never completes, and the next forward's process()
    blocks forever (observed: MTP tp1/dp2 decode /sleep hangs at "round in flight, poll not
    done", non-MTP escaped only because its single short forward left GPU gaps). Running the
    reduce on the host removes the SM contention entirely: it completes regardless of GPU
    occupancy, and stays async (async_op + is_completed poll) so an arm-skew of up to a step
    between ranks is tolerated without blocking the engine loop.
    """
    global _group_map
    if Group.SLEEP_QUIESCE in _group_map:
        return
    if not _sleep_quiesce_group_needed(parallelism_config):
        return
    world_size = parallelism_config.world_size
    world_rank = parallelism_config.world_rank
    quiesce_group = torch.distributed.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(days=36500),
    )
    _group_map[Group.SLEEP_QUIESCE] = quiesce_group
    logging.info(
        f"[rank: {world_rank}] Created SLEEP_QUIESCE gloo group {quiesce_group} with ranks: "
        f"{list(range(world_size))}"
    )

    # Warm the gloo group's lazy TCP rendezvous once at startup so the first arm-time
    # all-reduce during a sleep enqueues onto an already-connected group. This is a host
    # collective (CPU tensor), so unlike the old NCCL warmup it neither touches the GPU nor
    # collides with EP traffic; it is pure insurance against first-collective latency.
    # Rank-symmetric: every rank that created the group reaches this call.
    try:
        _warm = torch.zeros(1, dtype=torch.int64)  # CPU tensor for the gloo group
        # Bounded wait: the group carries a ~infinite (100-year) collective timeout so the per-step
        # async consensus never spuriously times out, but that means a BLOCKING warmup here would
        # hang launch forever if a peer died during startup. Issue it async and wait with a short
        # deadline instead -- a rank that never arrives fails the warmup fast (lazy init then runs on
        # the first sleep) rather than wedging the whole process at boot.
        _warm_work = torch.distributed.all_reduce(
            _warm, group=quiesce_group, async_op=True
        )
        _warm_work.wait(timeout=timedelta(seconds=_SLEEP_QUIESCE_WARMUP_TIMEOUT_S))
        logging.info(f"[rank: {world_rank}] warmed up SLEEP_QUIESCE gloo group")
    except Exception as e:  # pragma: no cover - never block startup on the warmup
        logging.warning(
            f"[rank: {world_rank}] SLEEP_QUIESCE group warmup failed or timed out "
            f"(lazy init will run on first sleep): {e}"
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
        elif group_key == Group.SLEEP_QUIESCE:
            if _CPP_PARALLEL_MODE_SLEEP_QUIESCE not in registered_modes:
                mode_to_group[_CPP_PARALLEL_MODE_SLEEP_QUIESCE] = pg
                registered_modes.add(_CPP_PARALLEL_MODE_SLEEP_QUIESCE)
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
        device_id = torch.cuda.current_device()
        for t in tensors:
            gpu_t, was_cpu = _ensure_cuda(t, device_id)
            torch.distributed.broadcast(gpu_t, root, group=pg)
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

    # --- Async comm ops (used by the sleep-quiesce consensus) -------------------------
    # A monotonically-increasing id keys in-flight torch Work handles. Access is
    # single-threaded in practice (only the engine loop thread issues/polls), so no lock
    # is needed; the dict just bridges the opaque uint64 handle held on the C++ side back
    # to the Python Work object.
    _async_works: Dict[int, Any] = {}
    _async_work_counter = [0]

    def cpp_allreduce_async(tensor: torch.Tensor, op: int, mode: int) -> int:
        """Enqueue an async (async_op=True) in-place all-reduce; return an opaque handle.

        Returns 0 when the group is absent/degenerate (nothing to reduce): the caller
        treats 0 as "already complete" and reads the local buffer unchanged.
        """
        pg = mode_to_group.get(mode)
        if pg is None or pg.size() < 2:
            return 0
        # The sleep-quiesce consensus runs on the gloo (CPU/host) group, so the buffer is
        # CPU-resident and the reduce executes on the host -- never competing with GPU
        # forward/DeepEP kernels for SMs. Pass the tensor through unchanged: do NOT promote
        # it to CUDA (gloo cannot reduce a CUDA tensor, and a GPU reduce is exactly the SM
        # contention we are avoiding). The reduce is in place, so the C++ caller reads the
        # summed verdict straight out of its CPU buffer once cpp_comm_poll reports complete.
        work = torch.distributed.all_reduce(
            tensor,
            op=_REDUCE_OPS.get(op, torch.distributed.ReduceOp.SUM),
            group=pg,
            async_op=True,
        )
        _async_work_counter[0] += 1
        handle = _async_work_counter[0]
        _async_works[handle] = work
        return handle

    def cpp_comm_poll(handle: int) -> bool:
        """Return True once the async collective for handle has completed.

        On completion the Work is waited on (so the reduced buffer is safe to read on the
        engine stream) and dropped. Unknown/zero handles return True (nothing to wait on).
        """
        if handle == 0:
            return True
        work = _async_works.get(handle)
        if work is None:
            return True
        if not work.is_completed():
            return False
        work.wait()
        _async_works.pop(handle, None)
        return True

    librtp_compute_ops.register_comm_ops(cpp_broadcast, cpp_allreduce, cpp_allgather)
    if hasattr(librtp_compute_ops, "register_async_comm_ops"):
        librtp_compute_ops.register_async_comm_ops(cpp_allreduce_async, cpp_comm_poll)
    logging.info(
        f"Registered C++ comm ops callbacks (modes: {list(mode_to_group.keys())})"
    )


def _init_cpu_tp_broadcaster() -> None:
    """Bootstrap the UDS-backed intra-node TP broadcaster after process groups."""
    try:
        import librtp_compute_ops
    except ImportError:
        return

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


def _destroy_user_buffers() -> None:
    snapshot = _require_distributed_init_snapshot()
    from rtp_llm.models_py.utils.arch import is_cuda

    if snapshot.parallelism_config.use_ub_comm and is_cuda():
        from rtp_llm.models_py.distributed import user_buffers

        user_buffers.destroy_user_buffers_communicator()
        # The legacy destroy helper deletes this module variable rather than
        # assigning None. Restore the sentinel so the next wake can initialize.
        user_buffers._global_communicator = None


def _clear_comm_ops() -> None:
    try:
        import librtp_compute_ops

        if hasattr(librtp_compute_ops, "clear_comm_ops"):
            librtp_compute_ops.clear_comm_ops()
    except ImportError:
        pass


def _destroy_cpu_tp_broadcaster() -> None:
    global _cpu_tp_broadcaster_base_path
    try:
        import librtp_compute_ops

        if hasattr(librtp_compute_ops, "destroy_cpu_tp_broadcaster"):
            librtp_compute_ops.destroy_cpu_tp_broadcaster()
    except ImportError:
        pass
    _cpu_tp_broadcaster_base_path = None


def _destroy_torch_process_groups() -> None:
    global _parallelism_config, _initialized

    rocm_rccl = _get_rocm_rccl()
    if rocm_rccl is not None:
        rocm_rccl.destroy_capture_comm()
    if torch.distributed.is_initialized():
        # Every rank reaches the destroy boundary before rank 0 drops the
        # rendezvous store. This also prevents a fast rank from rebuilding
        # against the previous process-group generation.
        quiesce_group = _group_map.get(Group.SLEEP_QUIESCE)
        if quiesce_group is not None:
            torch.distributed.barrier(group=quiesce_group)
        else:
            local_rank = (
                _require_distributed_init_snapshot().parallelism_config.local_rank
            )
            torch.distributed.barrier(
                group=torch.distributed.group.WORLD, device_ids=[local_rank]
            )

        # Symmetric-memory handles retain NVLS/P2P mappings and a strong PG
        # reference. They must be gone before ProcessGroupNCCL shutdown.
        _get_symm_mem().destroy_symm_mem_communicator()
        group_keys = [str(key) for key in _group_map]
        torch.distributed.destroy_process_group()
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "torch.distributed remained initialized after destroying all "
                f"process groups: {group_keys}"
            )
        logging.info("Destroyed all torch process groups: %s", group_keys)
    _group_map.clear()
    _parallelism_config = None
    _initialized = False
    cleanup_operations = [("gc.collect", gc.collect)]
    if torch.cuda.is_available():
        # Trim the CUDA IPC cache after every communication object has been
        # destroyed. Do not call empty_cache() here: in decode CUDA-graph
        # deployments backed by TMS it walks graph-private MemPools and can issue
        # a duplicate unmap/free. Level3 checkpointing releases the whole CUDA
        # context next, so allocator-cache trimming cannot improve final yield.
        cleanup_operations.extend((("cuda.ipc_collect", torch.cuda.ipc_collect),))
    for operation, reclaim in cleanup_operations:
        try:
            reclaim()
        except Exception as error:  # noqa: BLE001 - best-effort cache trim
            logging.warning(
                "Best-effort %s failed after process-group teardown " "(ignored): %s",
                operation,
                error,
            )


def _require_distributed_init_snapshot() -> _DistributedInitSnapshot:
    if _distributed_init_snapshot is None:
        raise RuntimeError(
            "distributed lifecycle has no initialization snapshot; initialize it first"
        )
    return _distributed_init_snapshot


def _ensure_collective_resources_registered() -> None:
    global _collective_resources_registered
    if _collective_resources_registered:
        return

    # Dependency order. Teardown is automatically the exact reverse order.
    _collective_lifecycle.register(
        "torch_process_groups",
        rebuild=_rebuild_torch_process_groups,
        teardown=_destroy_torch_process_groups,
    )
    _collective_lifecycle.register(
        "cpu_tp_broadcaster",
        rebuild=_init_cpu_tp_broadcaster,
        teardown=_destroy_cpu_tp_broadcaster,
    )
    _collective_lifecycle.register(
        "comm_ops",
        rebuild=_register_process_groups_to_cpp,
        teardown=_clear_comm_ops,
    )
    _collective_lifecycle.register(
        "user_buffers",
        rebuild=lambda: init_user_buffers_environment(
            _require_distributed_init_snapshot().parallelism_config
        ),
        teardown=_destroy_user_buffers,
    )
    _collective_resources_registered = True


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


def teardown_distributed_environment(*, coordinated: bool = True) -> None:
    """Suspend all collective resources while retaining rebuild configuration."""
    if _distributed_init_snapshot is None:
        if _collective_lifecycle.active_resources():
            raise RuntimeError(
                "collective resources are active without an initialization snapshot"
            )
        return

    rank = _distributed_init_snapshot.parallelism_config.world_rank
    logging.info(f"[rank: {rank}] Tearing down distributed environment")
    _collective_lifecycle.teardown()
    logging.info(f"[rank: {rank}] Distributed environment torn down")
    gc.collect()


def rebuild_distributed_environment() -> None:
    """Rebuild a previously torn-down environment in dependency order."""
    _enforce_level3_multicast_disabled()
    snapshot = _require_distributed_init_snapshot()
    if _collective_lifecycle.active_resources() and _initialized:
        # A fully active environment is an idempotent no-op. Partial state is
        # handled by registry.rebuild(), which resumes at the failed resource.
        expected = {
            "torch_process_groups",
            "cpu_tp_broadcaster",
            "comm_ops",
            "user_buffers",
        }
        if set(_collective_lifecycle.active_resources()) == expected:
            return
    logging.info(
        f"[rank: {snapshot.parallelism_config.world_rank}] Rebuilding distributed environment"
    )
    _collective_lifecycle.rebuild()


def destroy_distributed_environment() -> None:
    """Terminal, idempotent collective teardown that also drops rebuild state."""
    global _distributed_init_snapshot, _parallelism_config, _initialized
    global _cpu_tp_broadcaster_base_path
    global _lifecycle_store, _process_group_generation

    if _distributed_init_snapshot is not None:
        teardown_distributed_environment(coordinated=False)
    elif _collective_lifecycle.active_resources():
        raise RuntimeError(
            "cannot destroy active collective resources without an initialization snapshot"
        )

    _group_map.clear()
    _parallelism_config = None
    _cpu_tp_broadcaster_base_path = None
    _initialized = False
    _distributed_init_snapshot = None
    _lifecycle_store = None
    _process_group_generation = 0
    _reset_multicast_keeper_phase_instance()
    _collective_lifecycle.reset()
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


def all_reduce(
    tensor: torch.Tensor, group: Group, *, inplace: bool = False
) -> torch.Tensor:
    """All-reduce a tensor across all ranks in the group.

    Args:
        tensor: Tensor to all-reduce.
        group: Process group to use
        inplace: If true, write the symmetric-memory fast-path result back to ``tensor``.

    Returns:
        All-reduced tensor.
    """
    rocm_rccl = _get_rocm_rccl()
    if rocm_rccl is not None:
        rocm_rccl.ensure_capture_comm_ready(group == Group.TP)
        if rocm_rccl.should_use_capture_collectives(group == Group.TP):
            return rocm_rccl.capture_all_reduce(tensor, _get_group(group))

    if group == Group.TP:
        symm_mem_comm = _get_symm_mem().get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allreduce(
            tensor
        ):
            return symm_mem_comm.all_reduce(tensor, out=tensor if inplace else None)

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
    rocm_rccl = _get_rocm_rccl()
    if rocm_rccl is not None:
        rocm_rccl.ensure_capture_comm_ready(group == Group.TP)
        if rocm_rccl.should_use_capture_collectives(group == Group.TP):
            return rocm_rccl.capture_all_gather(tensor)

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
    "teardown_distributed_environment",
    "rebuild_distributed_environment",
    "send",
    "recv",
    "broadcast",
    "all_reduce",
    "all_gather",
    "barrier",
]
