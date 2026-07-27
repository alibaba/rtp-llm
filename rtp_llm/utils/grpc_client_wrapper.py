import asyncio
import importlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

import grpc
from google.protobuf.json_format import MessageToDict

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.frontend.sleep_validation import (
    dedupe_addresses,
    unsupported_lifecycle_control_field,
)
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.utils.time_util import Timer


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _error_details(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    details = []
    for result in results:
        if "error" not in result:
            continue
        detail = {
            "address": result.get("address", ""),
            "error": result.get("error", ""),
        }
        if "grpc_status" in result:
            detail["grpc_status"] = result["grpc_status"]
        details.append(detail)
    return details


def _report_metric_if_ready(metric: Any, value: float) -> None:
    if not bool(getattr(kmonitor, "_inited", False)):
        return
    kmonitor.report(metric, value)


def _report_sleep_status_metrics(status: Dict[str, Any]) -> None:
    if "error" in status:
        return
    _report_metric_if_ready(
        GaugeMetrics.SLEEP_ACTIVE_REQUEST_COUNT_METRIC,
        _as_int(status.get("active_request_count", 0)),
    )
    _report_metric_if_ready(
        GaugeMetrics.SLEEP_ACTIVE_CACHE_TRANSFER_COUNT_METRIC,
        _as_int(status.get("active_cache_transfer_count", 0)),
    )


def _normalize_json_request(req: Any) -> Dict[str, Any]:
    if isinstance(req, str):
        req = json.loads(req)
    if req is None:
        return {}
    if not isinstance(req, dict):
        raise ValueError("request body must be a JSON object")
    return req


def normalize_sleep_request(
    req: Any, configured_level: int
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Pure sleep-request validation shared by HTTP and direct callers."""

    try:
        normalized = dict(_normalize_json_request(req))
    except (TypeError, ValueError) as e:
        return None, {"error": str(e), "grpc_status": "INVALID_ARGUMENT"}
    unsupported_field = unsupported_lifecycle_control_field(normalized)
    if unsupported_field:
        return None, {
            "error": f"sleep {unsupported_field} is unsupported",
            "grpc_status": "INVALID_ARGUMENT",
        }
    try:
        level = int(normalized.get("level", configured_level))
        timeout_ms = int(normalized.get("timeout_ms", 60 * 60 * 1000))
    except (TypeError, ValueError):
        return None, {
            "error": "sleep level and timeout_ms must be integers",
            "grpc_status": "INVALID_ARGUMENT",
        }
    if level not in (0, 1, 2, 3):
        return None, {
            "error": "sleep level must be 0, 1, 2 or 3",
            "grpc_status": "INVALID_ARGUMENT",
        }
    if level != 0 and level != configured_level:
        return None, {
            "error": (
                f"sleep level={level} does not match configured "
                f"sleep_mode_level={configured_level}"
            ),
            "grpc_status": "INVALID_ARGUMENT",
        }
    mode = str(normalized.get("mode", "wait"))
    if mode not in ("wait", "abort"):
        return None, {
            "error": 'sleep mode must be "wait" or "abort"',
            "grpc_status": "INVALID_ARGUMENT",
        }
    tags = normalized.get("tags", [])
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        return None, {
            "error": "sleep tags must be a list",
            "grpc_status": "INVALID_ARGUMENT",
        }
    if any(not isinstance(tag, str) or not tag for tag in tags):
        return None, {
            "error": "sleep tags must be non-empty strings",
            "grpc_status": "INVALID_ARGUMENT",
        }
    normalized["level"] = level
    if "mode" in normalized:
        normalized["mode"] = mode
    if "timeout_ms" in normalized:
        normalized["timeout_ms"] = timeout_ms
    if "tags" in normalized:
        normalized["tags"] = list(tags)
    return normalized, None


def normalize_wake_request(
    req: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Pure wake-request validation shared by HTTP and direct callers."""

    try:
        normalized = dict(_normalize_json_request(req))
    except (TypeError, ValueError) as e:
        return None, {"error": str(e), "grpc_status": "INVALID_ARGUMENT"}
    unsupported_field = unsupported_lifecycle_control_field(normalized)
    if unsupported_field:
        return None, {
            "error": f"wake_up {unsupported_field} is unsupported",
            "grpc_status": "INVALID_ARGUMENT",
        }
    return normalized, None


def _local_process_identity(pid: Optional[int] = None) -> Dict[str, Any]:
    from rtp_llm.utils.checkpoint_controller import read_process_starttime

    process_id = os.getpid() if pid is None else int(pid)
    try:
        with open(
            "/proc/sys/kernel/random/boot_id", "r", encoding="utf-8"
        ) as boot_id_file:
            boot_id = boot_id_file.read().strip()
        pid_namespace = os.stat("/proc/self/ns/pid").st_ino
    except OSError as e:
        raise RuntimeError(f"cannot inspect local process identity: {e}") from e
    if not boot_id or pid_namespace <= 0:
        raise RuntimeError("local process identity is incomplete")
    return {
        "pid": process_id,
        "starttime": read_process_starttime(process_id),
        "pid_namespace": int(pid_namespace),
        "boot_id": boot_id,
    }


def sleep_level3_options_from_config(
    engine_config: Any, world_info: Any
) -> Dict[str, Any]:
    """Derive frontend L3 guards from the initialized runtime configuration."""

    world_size = int(engine_config.parallelism_config.world_size)
    local_world_size = int(engine_config.parallelism_config.local_world_size)
    sleep_enabled = bool(engine_config.runtime_config.enable_sleep_mode)
    configured_level = int(engine_config.runtime_config.sleep_mode_level)
    if configured_level not in (1, 2, 3):
        raise ValueError("configured sleep_mode_level must be 1, 2 or 3")
    return {
        "sleep_enabled": sleep_enabled,
        "configured_level": configured_level,
        "level3_enabled": sleep_enabled and configured_level == 3,
        "single_node": bool(
            getattr(world_info, "num_nodes", 1) == 1 and world_size == local_world_size
        ),
        "rdma_enabled": bool(engine_config.cache_store_config.cache_store_rdma_mode),
    }


class _CheckpointControllerAdapter:
    """Localized adapter for the separately implemented checkpoint controller.

    CUDA checkpoint operations run in ``asyncio.to_thread``. This adapter maps
    frontend control addresses/status dicts to the concrete controller API and
    normalizes its ``RecoveryStatus`` dataclasses into local dictionaries.
    """

    MODULE_NAME = "rtp_llm.utils.checkpoint_controller"

    def _module(self) -> Any:
        return importlib.import_module(self.MODULE_NAME)

    def preflight(
        self, control_addresses: Sequence[str], namespace: Optional[str] = None
    ) -> None:
        module = self._module()
        required = (
            "CheckpointTarget",
            "checkpoint_manifest_path",
            "checkpoint_all",
            "read_process_starttime",
            "restore_all",
            "recovery_status",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(
                "checkpoint controller is missing APIs: " + ", ".join(missing)
            )
        manifest_path = module.checkpoint_manifest_path(
            control_addresses, namespace=namespace
        )
        # recovery_status constructs the real driver even when no manifest is
        # present, so missing CUDA checkpoint symbols fail before backend drain.
        module.recovery_status(manifest_path)

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    @classmethod
    def _normalize_status(cls, status: Any) -> Dict[str, Any]:
        processes = []
        for process in tuple(getattr(status, "processes", ()) or ()):
            processes.append(
                {
                    "pid": int(process.pid),
                    "starttime": int(process.starttime),
                    "rank": int(process.rank),
                    "address": str(process.address),
                    "state": str(cls._enum_value(process.state)),
                    "identity_valid": bool(process.identity_valid),
                    "driver_state": process.driver_state,
                    "error": process.error,
                }
            )
        phase = str(getattr(status, "phase", "")).upper()
        recovery_required = bool(getattr(status, "recovery_required", False))
        checkpoint_complete = bool(getattr(status, "checkpoint_complete", False))
        restore_complete = bool(getattr(status, "restore_complete", False))
        if recovery_required:
            state = "RECOVERY_REQUIRED"
        elif checkpoint_complete or phase == "CHECKPOINTED":
            state = "CHECKPOINTED"
        elif restore_complete and processes:
            state = "RUNNING"
        else:
            state = phase or "NONE"
        all_running = (
            bool(processes)
            and restore_complete
            and all(
                process["identity_valid"] and process["driver_state"] == "RUNNING"
                for process in processes
            )
        )
        last_error = getattr(status, "last_error", None)
        return {
            "state": state,
            "phase": phase,
            "epoch": getattr(status, "epoch", None),
            "manifest_exists": bool(getattr(status, "manifest_exists", False)),
            "recovery_required": recovery_required,
            "checkpoint_complete": checkpoint_complete,
            "restore_complete": restore_complete,
            "all_running": all_running,
            "processes": processes,
            "pids": [process["pid"] for process in processes],
            "last_error": last_error or "",
            "error": last_error or "",
        }

    @staticmethod
    def _target_rank(status: Dict[str, Any], default_rank: int) -> int:
        # The backend reports its authoritative world_rank in SleepStatusResponsePB.
        # Prefer it over control-address order: a shared/misconfigured store or a
        # non-address-ordered control list must not silently reassign ranks. Fall
        # back to address order only for a legacy backend that omits world_rank
        # (rolling upgrade), keeping L1/L2 backward compatible.
        if "world_rank" in status and status.get("world_rank") is not None:
            rank = status.get("world_rank")
        elif "rank" in status and status.get("rank") is not None:
            rank = status.get("rank")
        else:
            rank = default_rank
        if isinstance(rank, bool):
            raise RuntimeError("checkpoint target rank must be an integer")
        try:
            normalized = int(rank)
        except (TypeError, ValueError) as e:
            raise RuntimeError("checkpoint target rank must be an integer") from e
        if normalized < 0:
            raise RuntimeError("checkpoint target rank must be non-negative")
        return normalized

    def _targets_and_epoch(
        self,
        module: Any,
        control_addresses: Sequence[str],
        terminal_statuses: Sequence[Dict[str, Any]],
    ) -> tuple[List[Any], int]:
        if len(control_addresses) != len(terminal_statuses) or not terminal_statuses:
            raise RuntimeError("checkpoint target/status coverage is incomplete")
        targets = []
        epochs = set()
        pids = set()
        ranks = set()
        local_identity = _local_process_identity()
        for default_rank, (address, status) in enumerate(
            zip(control_addresses, terminal_statuses)
        ):
            if str(status.get("address", "")) != str(address):
                raise RuntimeError("checkpoint status address order is inconsistent")
            pid = _as_int(status.get("process_id", 0))
            if pid <= 0 or pid in pids:
                raise RuntimeError("checkpoint target PIDs must be unique and positive")
            if isinstance(status.get("sleep_epoch"), bool):
                raise RuntimeError("checkpoint sleep_epoch must be an integer")
            try:
                epoch = int(status.get("sleep_epoch"))
            except (TypeError, ValueError) as e:
                raise RuntimeError("checkpoint sleep_epoch must be an integer") from e
            if epoch < 0:
                raise RuntimeError("checkpoint sleep_epoch must be non-negative")
            rank = self._target_rank(status, default_rank)
            if rank in ranks:
                raise RuntimeError("checkpoint target ranks must be unique")
            starttime = _as_int(status.get("process_starttime", 0))
            pid_namespace = _as_int(status.get("process_pid_namespace", 0))
            boot_id = str(status.get("process_boot_id", ""))
            if starttime <= 0:
                raise RuntimeError("checkpoint target starttime must be positive")
            if pid_namespace != local_identity["pid_namespace"]:
                raise RuntimeError(
                    "checkpoint coordinator and backend must share a PID namespace"
                )
            if boot_id != local_identity["boot_id"]:
                raise RuntimeError(
                    "checkpoint coordinator and backend must run on the same host boot"
                )
            if module.read_process_starttime(pid) != starttime:
                raise RuntimeError(
                    "checkpoint target PID identity differs from backend status"
                )
            pids.add(pid)
            ranks.add(rank)
            epochs.add(epoch)
            targets.append(module.CheckpointTarget(pid, rank, str(address), starttime))
        if len(epochs) != 1:
            raise RuntimeError("checkpoint terminal sleep_epoch is inconsistent")
        return targets, epochs.pop()

    def checkpoint_all(
        self,
        control_addresses: Sequence[str],
        terminal_statuses: Sequence[Dict[str, Any]],
        namespace: Optional[str] = None,
        holder_instance: Optional[str] = None,
        team: Optional[str] = None,
    ) -> Any:
        module = self._module()
        manifest_path = module.checkpoint_manifest_path(
            control_addresses, namespace=namespace
        )
        targets, epoch = self._targets_and_epoch(
            module, control_addresses, terminal_statuses
        )
        try:
            status = module.checkpoint_all(
                manifest_path,
                targets,
                epoch,
                holder_instance=holder_instance,
                team=team,
            )
        except Exception as e:
            recovery_status = getattr(e, "status", None)
            if recovery_status is not None:
                normalized = self._normalize_status(recovery_status)
                e.checkpoint_status = normalized
                e.all_running = normalized["all_running"]
            raise
        normalized = self._normalize_status(status)
        if not normalized["checkpoint_complete"]:
            raise RuntimeError("checkpoint_all returned before checkpoint completed")
        return normalized

    def restore_all(
        self,
        control_addresses: Sequence[str],
        namespace: Optional[str] = None,
        holder_instance: Optional[str] = None,
        team: Optional[str] = None,
    ) -> Any:
        module = self._module()
        manifest_path = module.checkpoint_manifest_path(
            control_addresses, namespace=namespace
        )
        status = module.restore_all(
            manifest_path,
            expected_holder_instance=holder_instance,
            expected_team=team,
        )
        normalized = self._normalize_status(status)
        if not normalized["restore_complete"]:
            raise RuntimeError("restore_all returned before restore completed")
        return normalized

    def read_manifest(
        self, control_addresses: Sequence[str], namespace: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            module = self._module()
        except ModuleNotFoundError as e:
            if e.name == self.MODULE_NAME:
                return None
            raise
        manifest_path = module.checkpoint_manifest_path(
            control_addresses, namespace=namespace
        )
        if not os.path.exists(manifest_path):
            return None
        normalized = self._normalize_status(module.recovery_status(manifest_path))
        if not normalized["manifest_exists"]:
            return None
        return normalized

    def clear_manifest(
        self, control_addresses: Sequence[str], namespace: Optional[str] = None
    ) -> None:
        """Best-effort removal of a durable checkpoint manifest file.

        Used to discard a manifest that has been proven stale (all recorded
        backend processes are gone). Never raises for a missing file.
        """
        try:
            module = self._module()
        except ModuleNotFoundError:
            return
        manifest_path = module.checkpoint_manifest_path(
            control_addresses, namespace=namespace
        )
        try:
            os.remove(manifest_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            logging.warning(
                "failed to remove stale checkpoint manifest %s: %s", manifest_path, e
            )


class GrpcClientWrapper:
    """Wrapper for direct gRPC calls to replace async_request_server"""

    LIFECYCLE_LEASE_KEY = "rtp_llm_instance_lifecycle_lease"
    LIFECYCLE_RECOVERY_KEY = "rtp_llm_instance_lifecycle_recovery"
    COMMIT_MAX_ATTEMPTS = 3
    LIFECYCLE_STATUS_MAX_ATTEMPTS = 3
    LIFECYCLE_STATUS_POLL_INTERVAL_S = 0.05
    CHECKPOINT_STATES = {
        "CHECKPOINTING",
        "CHECKPOINTED",
        "RESTORING",
        "RECOVERY_REQUIRED",
    }

    def __init__(
        self,
        server_port: int,
        dp_addresses: Optional[List[str]] = None,
        control_addresses: Optional[List[str]] = None,
        expected_control_address_count: Optional[int] = None,
        control_address_resolver: Optional[Callable[[], List[str]]] = None,
        lifecycle_store: Optional[Any] = None,
        lifecycle_store_factory: Optional[Callable[[], Optional[Any]]] = None,
        require_instance_lease: bool = False,
        checkpoint_controller: Optional[Any] = None,
        sleep_enabled: bool = True,
        configured_level: int = 1,
        level3_enabled: Optional[bool] = None,
        single_node: Optional[bool] = None,
        rdma_enabled: bool = False,
    ):
        self.server_port = server_port
        self.address = f"localhost:{server_port}"
        self.channel = None
        self.stub = None
        # Serving-route broadcast targets, normally one representative per DP
        # group. Do not use these for sleep/wake_up: lifecycle control must
        # reach every backend rank process that owns GPU resources.
        self.dp_addresses = dedupe_addresses(
            dp_addresses if dp_addresses else [self.address]
        )
        self.control_addresses = dedupe_addresses(
            control_addresses if control_addresses else [self.address]
        )
        self.expected_control_address_count = expected_control_address_count
        self._control_address_resolver = control_address_resolver
        self._lifecycle_store = lifecycle_store
        self._lifecycle_store_factory = lifecycle_store_factory
        self._require_instance_lease = (
            require_instance_lease or lifecycle_store is not None
        )
        configured_level = int(configured_level)
        if configured_level not in (1, 2, 3):
            raise ValueError("configured_level must be 1, 2 or 3")
        self._sleep_enabled = bool(sleep_enabled)
        self._configured_level = configured_level
        expected_level3_enabled = self._sleep_enabled and configured_level == 3
        if level3_enabled is None:
            level3_enabled = expected_level3_enabled
        self._level3_enabled = bool(level3_enabled)
        if self._level3_enabled != expected_level3_enabled:
            raise ValueError(
                "level3_enabled must match sleep_enabled and configured_level=3"
            )
        self._checkpoint_controller = None
        if self._level3_enabled:
            self._checkpoint_controller = (
                checkpoint_controller
                if checkpoint_controller is not None
                else _CheckpointControllerAdapter()
            )
        self._single_node = single_node
        self._rdma_enabled = bool(rdma_enabled)
        self._lifecycle_holder = uuid.uuid4().hex
        # This lock handles re-entrancy within one frontend. The TCPStore lease
        # below is the authoritative instance-wide serialization mechanism.
        self._lifecycle_lock = asyncio.Lock()
        self._dp_channels: Dict[str, Any] = {}
        self._dp_stubs: Dict[str, Any] = {}
        self._frontend_lifecycle_state = ""
        self._frontend_lifecycle_error = ""
        self._terminal_sleep_statuses: List[Dict[str, Any]] = []
        self._terminal_sleep_status: Dict[str, Any] = {}
        self._level3_wake_completed = False
        # Backend-reported instance identity, resolved lazily from sleep status.
        # Lease/recovery/manifest keys are namespaced by (generation, role) so a
        # shared TCPStore cannot collide across PD roles or instance generations.
        self._instance_role: Optional[str] = None
        self._instance_generation: Optional[str] = None

    @property
    def configured_sleep_level(self) -> int:
        return self._configured_level

    def _control_address_coverage_error(self) -> str:
        if not self.expected_control_address_count:
            return ""
        actual = len(self.control_addresses)
        expected = int(self.expected_control_address_count)
        if actual >= expected:
            return ""
        return (
            "sleep mode disabled: lifecycle control address coverage incomplete, "
            f"expected {expected} backend ranks but discovered {actual}"
        )

    def _refresh_control_addresses_if_needed(self) -> None:
        if self._control_address_resolver is None:
            return
        expected = int(self.expected_control_address_count or 0)
        if expected > 0 and len(self.control_addresses) >= expected:
            return
        try:
            resolved_addresses = dedupe_addresses(
                self._control_address_resolver() or []
            )
        except Exception as e:
            logging.warning("sleep control address resolver failed: %s", e)
            return
        if not resolved_addresses:
            return
        if expected > 0 and len(resolved_addresses) < len(self.control_addresses):
            return
        if resolved_addresses == self.control_addresses:
            return
        logging.info(
            "refresh sleep control addresses: old=%s, new=%s",
            self.control_addresses,
            resolved_addresses,
        )
        self.control_addresses = resolved_addresses

    async def _ensure_connection(self):
        """Ensure gRPC channel and stub are created"""
        if self.channel is None or self.stub is None:
            self.channel = grpc.aio.insecure_channel(
                self.address,
                options=[
                    ("grpc.max_metadata_size", 1024 * 1024 * 1024),
                ],
            )
            self.stub = RpcServiceStub(self.channel)

    async def _ensure_dp_connection(self, address: str):
        """Ensure gRPC channel and stub are created for a specific DP address"""
        if address not in self._dp_channels or self._dp_stubs.get(address) is None:
            self._dp_channels[address] = grpc.aio.insecure_channel(
                address,
                options=[
                    ("grpc.max_metadata_size", 1024 * 1024 * 1024),
                ],
            )
            self._dp_stubs[address] = RpcServiceStub(self._dp_channels[address])

    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
        for address, channel in self._dp_channels.items():
            try:
                await channel.close()
            except Exception as e:
                logging.warning(f"Failed to close DP channel for {address}: {e}")
        self._dp_channels.clear()
        self._dp_stubs.clear()

    async def _reset_main_channel(self) -> None:
        """Tear down ONLY the health/status channel so the next probe reconnects.

        Deliberately does not touch self._dp_channels: those carry in-flight
        sleep/wake lifecycle RPCs. Closing a channel while one of its calls is
        genuinely in-flight (server-accepted, still processing) raises
        asyncio.CancelledError into the awaiting coroutine -- a BaseException
        that bypasses every ``except Exception`` on the lifecycle path. A
        routine health-probe timeout during a sleep/wake drain would otherwise
        cancel the unrelated lifecycle operation and surface HTTP 500 while the
        backend keeps transitioning to SLEEPING -- a control-plane split brain.
        """
        if self.channel:
            try:
                await self.channel.close()
            except Exception as e:
                logging.warning(f"Failed to close health channel: {e}")
        self.channel = None
        self.stub = None

    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            await self._ensure_connection()
            # Using a simple request to check if server is responsive
            request = pb2.EmptyPB()
            await self.stub.CheckHealth(request, timeout=1)
            return {"status": "ok"}
        except Exception as e:
            # Reset only the health channel. Never close the shared lifecycle
            # (_dp_channels) here: see _reset_main_channel -- doing so would
            # cancel an in-flight sleep/wake RPC that shares this wrapper.
            await self._reset_main_channel()
            return {
                "status": "error",
                "message": e,
            }

    async def get_cache_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cache status from gRPC server"""
        try:
            start_time = time.time() * 1000
            await self._ensure_connection()
            request = pb2.CacheVersionPB(
                latest_cache_version=query_params.get("latest_cache_version", -1),
                need_cache_keys=query_params.get("need_cache_keys", True),
            )
            response = await self.stub.GetCacheStatus(request, timeout=1)
            # Convert response to dict format expected by frontend
            result = MessageToDict(
                response,
                preserving_proto_field_name=True,
                including_default_value_fields=True,
            )
            kmonitor.report(AccMetrics.CACHE_STATUS_QPS_METRIC, 1)
            kmonitor.report(
                GaugeMetrics.CACHE_STATUS_QPS_LATENCY_METRIC,
                time.time() * 1000 - start_time,
            )
            return result

        except Exception as e:
            logging.error(f"Get cache status failed: {e}")
            return {"error": f"Failed to get cache status: {str(e)}"}

    async def get_worker_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get worker status from gRPC server"""
        try:
            start_time = time.time() * 1000
            await self._ensure_connection()
            request = pb2.StatusVersionPB(
                latest_cache_version=query_params.get("latest_cache_version", -1),
                latest_finished_version=query_params.get("latest_finished_version", -1),
            )
            response = await self.stub.GetWorkerStatus(request, timeout=1)
            # Convert response to dict format expected by frontend
            result = MessageToDict(
                response,
                preserving_proto_field_name=True,
                including_default_value_fields=True,
            )
            kmonitor.report(AccMetrics.WORKER_STATUS_QPS_METRIC, 1)
            kmonitor.report(
                GaugeMetrics.WORKER_STATUS_QPS_LANTENCY_METRIC,
                time.time() * 1000 - start_time,
            )
            return result
        except Exception as e:
            logging.error(f"Get worker status failed: {e}")
            return {"error": f"Failed to get worker status: {str(e)}"}

    async def set_log_level(self, req: Any) -> Dict[str, Any]:
        """Set log level - this would need to be implemented based on your requirements"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            request = pb2.SetLogLevelRequestPB(
                log_level=req.get("log_level", "INFO"),
            )
            await self.stub.SetLogLevel(request, timeout=3)
            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Set log level failed: {e}")
            return {"error": f"Failed to set log level: {str(e)}"}

    async def _call_control_rpc(
        self, address: str, rpc_name: str, request: Any, timeout_s: float
    ) -> Dict[str, Any]:
        try:
            await self._ensure_dp_connection(address)
            rpc = getattr(self._dp_stubs[address], rpc_name)
            response = await rpc(request, timeout=timeout_s)
            result: Dict[str, Any] = {"address": address, "status": "ok"}
            if response is not None and not isinstance(response, pb2.EmptyPB):
                result.update(
                    MessageToDict(
                        response,
                        preserving_proto_field_name=True,
                        including_default_value_fields=True,
                    )
                )
            return result
        except grpc.aio.AioRpcError as e:
            logging.error("%s failed on %s: %s", rpc_name, address, e.details())
            return {
                "address": address,
                "error": str(e.details()),
                "grpc_status": e.code().name,
            }
        except Exception as e:
            logging.error("%s failed on %s: %s", rpc_name, address, e)
            return {"address": address, "error": str(e)}

    async def _broadcast_control_rpc(
        self, rpc_name: str, request: Any, timeout_s: float
    ) -> List[Dict[str, Any]]:
        return await self._broadcast_control_rpc_to(
            self.control_addresses, rpc_name, request, timeout_s
        )

    async def _broadcast_control_rpc_to(
        self,
        addresses: List[str],
        rpc_name: str,
        request: Any,
        timeout_s: float,
    ) -> List[Dict[str, Any]]:
        tasks = [
            self._call_control_rpc(address, rpc_name, request, timeout_s)
            for address in addresses
        ]
        return await asyncio.gather(*tasks)

    def _lease_record(self, operation: str) -> str:
        identity = _local_process_identity()
        return json.dumps(
            {
                "holder": self._lifecycle_holder,
                "operation": operation,
                **identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _lease_owner_is_stale(record: str) -> bool:
        try:
            owner = json.loads(record)
            local_identity = _local_process_identity()
            if (
                not isinstance(owner, dict)
                or owner.get("boot_id") != local_identity["boot_id"]
                or _as_int(owner.get("pid_namespace", 0))
                != local_identity["pid_namespace"]
            ):
                return False
            owner_pid = _as_int(owner.get("pid", 0))
            owner_starttime = _as_int(owner.get("starttime", 0))
            if owner_pid <= 0 or owner_starttime <= 0:
                return False
            from rtp_llm.utils.checkpoint_controller import read_process_starttime

            try:
                return read_process_starttime(owner_pid) != owner_starttime
            except Exception:
                return True
        except Exception:
            return False

    @staticmethod
    def _sanitize_key_component(value: Any) -> str:
        text = str(value or "")
        return re.sub(r"[^A-Za-z0-9._-]", "_", text)

    def _key_namespace_suffix(self) -> str:
        # Empty until identity is resolved (legacy/unresolved -> base keys, keeping
        # backward compatibility). Once resolved, every lease/recovery/manifest key
        # is scoped to this instance generation and PD role so a shared TCPStore
        # cannot collide across roles or instance generations.
        if not self._instance_generation and not self._instance_role:
            return ""
        role = self._sanitize_key_component(self._instance_role or "unknown")
        generation = self._sanitize_key_component(self._instance_generation or "0")
        return f"/{role}/{generation}"

    def _lifecycle_lease_key(self) -> str:
        return f"{self.LIFECYCLE_LEASE_KEY}{self._key_namespace_suffix()}"

    def _lifecycle_recovery_key(self) -> str:
        return f"{self.LIFECYCLE_RECOVERY_KEY}{self._key_namespace_suffix()}"

    def _manifest_namespace(self) -> Optional[str]:
        # Scope the durable checkpoint manifest to this instance generation + role
        # so a node that reuses control addresses across generations/roles cannot
        # pick up a stale manifest. None until identity is resolved (legacy path).
        suffix = self._key_namespace_suffix()
        return suffix.lstrip("/") if suffix else None

    def _keeper_holder_instance(
        self, statuses: Optional[Sequence[Dict[str, Any]]] = None
    ) -> Optional[str]:
        # The multicast keeper holder identity, if the backend reports it on the
        # rank-status path. Persisted into the manifest and re-verified on restore
        # so a holder that exits/changes fails closed instead of a silent bad wake.
        statuses = statuses if statuses is not None else self._terminal_sleep_statuses
        for status in statuses or ():
            if not isinstance(status, dict):
                continue
            if _as_int(status.get("world_rank", -1), -1) == 0 and status.get(
                "holder_instance"
            ):
                return str(status.get("holder_instance"))
        holders = {
            str(status.get("holder_instance"))
            for status in (statuses or ())
            if isinstance(status, dict) and status.get("holder_instance")
        }
        if len(holders) == 1:
            return holders.pop()
        return None

    def _keeper_team(self) -> Optional[str]:
        if self._instance_role is None and self._instance_generation is None:
            return None
        return self._manifest_namespace()

    def _apply_instance_identity(self, statuses: Sequence[Dict[str, Any]]) -> str:
        """Adopt the backend-reported (role, instance_generation) for key namespacing.

        The generation is taken from the world_rank==0 rank (one per instance), so
        per-process generation uuids need no cross-rank agreement. All reported roles
        must agree; a mismatch means a shared store is being addressed by inconsistent
        backends and is rejected (the caller fails closed).
        """
        roles = {str(status.get("role")) for status in statuses if status.get("role")}
        if len(roles) > 1:
            return f"backend ranks report inconsistent roles: {sorted(roles)}"
        generation = None
        for status in statuses:
            if _as_int(status.get("world_rank", -1), -1) == 0:
                generation = status.get("instance_generation_uuid") or None
                break
        if generation is None:
            generations = {
                str(status.get("instance_generation_uuid"))
                for status in statuses
                if status.get("instance_generation_uuid")
            }
            if len(generations) == 1:
                generation = generations.pop()
        role = roles.pop() if roles else None
        if role is not None:
            self._instance_role = role
        if generation is not None:
            self._instance_generation = str(generation)
        return ""

    async def _resolve_instance_identity(self) -> None:
        """Best-effort: populate instance identity from backend sleep status.

        Failures are non-fatal (the backend may be checkpointed on wake); the
        identity resolved during sleep persists on this instance for later wake.
        """
        if self._instance_generation or self._instance_role:
            return
        try:
            statuses = await self._raw_sleep_statuses()
        except Exception as e:
            logging.warning("failed to resolve backend instance identity: %s", e)
            return
        statuses = [status for status in statuses if isinstance(status, dict)]
        error = self._rank_identity_error(statuses)
        if not error:
            error = self._apply_instance_identity(statuses)
        if error:
            logging.warning("backend instance identity is inconsistent: %s", error)

    def _get_lifecycle_store(self) -> Optional[Any]:
        if self._lifecycle_store is None and self._lifecycle_store_factory is not None:
            try:
                self._lifecycle_store = self._lifecycle_store_factory()
            except Exception as e:
                logging.error("failed to establish lifecycle TCPStore: %s", e)
        return self._lifecycle_store

    def _acquire_lifecycle_lease(
        self, operation: str
    ) -> tuple[Optional[str], Dict[str, Any]]:
        store = self._get_lifecycle_store()
        if store is None:
            if not self._require_instance_lease:
                return None, {}
            return None, {
                "error": "instance-wide lifecycle coordination is unavailable",
                "grpc_status": "FAILED_PRECONDITION",
            }
        recovery_record, recovery_error = self._read_lifecycle_recovery_record(store)
        if recovery_error:
            return None, {
                "error": f"RECOVERY_REQUIRED: {recovery_error}",
                "grpc_status": "FAILED_PRECONDITION",
                "recovery_required": True,
            }
        if recovery_record:
            return None, {
                "error": "RECOVERY_REQUIRED: "
                f"{self._recovery_record_error(recovery_record)}",
                "grpc_status": "FAILED_PRECONDITION",
                "recovery_required": True,
            }
        record = self._lease_record(operation)
        try:
            current = store.compare_set(self._lifecycle_lease_key(), "", record)
            if isinstance(current, bytes):
                current = current.decode("utf-8")
        except Exception as e:
            return None, {
                "error": f"instance-wide lifecycle coordination failed: {e}",
                "grpc_status": "FAILED_PRECONDITION",
            }
        if current != record:
            if current and self._lease_owner_is_stale(current):
                try:
                    current = store.compare_set(
                        self._lifecycle_lease_key(), current, record
                    )
                    if isinstance(current, bytes):
                        current = current.decode("utf-8")
                    if current == record:
                        logging.warning(
                            "reclaimed lifecycle lease from a dead frontend process"
                        )
                except Exception as e:
                    return None, {
                        "error": f"instance-wide lifecycle lease recovery failed: {e}",
                        "grpc_status": "FAILED_PRECONDITION",
                    }
        if current != record:
            return None, {
                "error": "another lifecycle operation holds the instance lease",
                "grpc_status": "FAILED_PRECONDITION",
            }
        return record, {}

    def _release_lifecycle_lease(self, record: Optional[str]) -> None:
        if record is None or self._lifecycle_store is None:
            return
        try:
            self._lifecycle_store.compare_set(self._lifecycle_lease_key(), record, "")
        except Exception as e:
            # Fail closed: an uncertain/non-released record must continue blocking
            # lifecycle operations until the instance (and its TCPStore) restarts.
            logging.error("failed to release instance lifecycle lease: %s", e)

    @staticmethod
    def _decode_store_value(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value or "")

    @staticmethod
    def _recovery_record_error(record: str) -> str:
        try:
            payload = json.loads(record)
            if isinstance(payload, dict):
                return str(payload.get("error", "") or "instance state is uncertain")
        except (TypeError, ValueError):
            pass
        return "instance state is uncertain"

    def _read_lifecycle_recovery_record(
        self, store: Optional[Any] = None
    ) -> tuple[str, str]:
        store = store if store is not None else self._get_lifecycle_store()
        if store is None:
            return "", ""
        try:
            current = store.compare_set(self._lifecycle_recovery_key(), "", "")
            return self._decode_store_value(current), ""
        except Exception as e:
            return "", f"lifecycle recovery state is unavailable: {e}"

    def _persist_lifecycle_recovery(self, error: str) -> bool:
        store = self._get_lifecycle_store()
        if store is None:
            return not self._require_instance_lease
        try:
            record = json.dumps(
                {
                    "state": "RECOVERY_REQUIRED",
                    "error": error,
                    "holder": self._lifecycle_holder,
                    **_local_process_identity(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            current = store.compare_set(self._lifecycle_recovery_key(), "", record)
            current = self._decode_store_value(current)
            if not current:
                raise RuntimeError("recovery marker remained empty")
            return True
        except Exception as e:
            logging.error("failed to persist lifecycle recovery state: %s", e)
            return False

    async def _checkpoint_controller_call(
        self, method: str, *args: Any, **kwargs: Any
    ) -> Any:
        if self._checkpoint_controller is None:
            raise RuntimeError("level-3 checkpoint controller is disabled")
        callback = getattr(self._checkpoint_controller, method)
        return await asyncio.to_thread(lambda: callback(*args, **kwargs))

    def _set_frontend_lifecycle_state(self, state: str, error: str = "") -> None:
        self._frontend_lifecycle_state = state
        self._frontend_lifecycle_error = error
        if state == "RECOVERY_REQUIRED":
            self._persist_lifecycle_recovery(error or "lifecycle state is uncertain")

    @staticmethod
    def _manifest_state(manifest: Optional[Dict[str, Any]]) -> str:
        if not manifest:
            return ""
        state = str(manifest.get("state", "")).upper()
        if state:
            return state
        if manifest.get("pids") or manifest.get("targets"):
            return "CHECKPOINTED"
        return ""

    async def _read_checkpoint_manifest(self) -> Optional[Dict[str, Any]]:
        manifest = await self._checkpoint_controller_call(
            "read_manifest",
            tuple(self.control_addresses),
            namespace=self._manifest_namespace(),
        )
        if manifest is None:
            return None
        if not isinstance(manifest, dict):
            raise RuntimeError("checkpoint controller returned an invalid manifest")
        return manifest

    def _synthesize_checkpoint_status(
        self, state: str, manifest: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        manifest = manifest or {}
        status = dict(self._terminal_sleep_status)
        status.update(
            {
                "state": state,
                "sleep_mode_enabled": True,
                "effective": True,
                "supported_levels": [3],
                "supported_modes": ["wait", "abort"],
                "disabled_reason": "",
                "active_request_count": 0,
                "active_cache_transfer_count": 0,
                "device_kv_cache_valid": False,
            }
        )
        gpu_states = {
            "CHECKPOINTING": "CHECKPOINTING",
            "CHECKPOINTED": "RELEASED",
            "RESTORING": "RESTORING",
            "RECOVERY_REQUIRED": "ERROR",
        }
        status["gpu_resource_state"] = gpu_states[state]
        status.setdefault("kv_memory_state", "PAUSED")
        status.setdefault(
            "sleep_epoch",
            _as_int(manifest.get("sleep_epoch", manifest.get("epoch", 0))),
        )
        pids = manifest.get("pids")
        if pids is None:
            pids = [
                _as_int(result.get("process_id", 0))
                for result in self._terminal_sleep_statuses
                if _as_int(result.get("process_id", 0)) > 0
            ]
        status["process_ids"] = list(pids or [])
        error = str(manifest.get("error", "") or self._frontend_lifecycle_error or "")
        status["last_error"] = error
        if state == "RECOVERY_REQUIRED":
            status.update(
                {
                    "error": f"RECOVERY_REQUIRED: {error or 'checkpoint state is uncertain'}",
                    "grpc_status": "FAILED_PRECONDITION",
                    "recovery_required": True,
                }
            )
        return status

    @staticmethod
    def _manifest_is_stale(manifest: Optional[Dict[str, Any]]) -> bool:
        """A durable L3 checkpoint manifest whose recorded backend processes are
        ALL gone is a leftover from a prior instance generation that reused the
        same control addresses (the manifest path is address-keyed until instance
        identity is resolved, so it is stable across relaunches). A genuinely
        checkpointed L3 process stays alive (frozen) with its pid/starttime
        preserved, so if not a single recorded process is live, the manifest is
        stale and must not wedge a fresh frontend's startup health gate.
        """
        if not manifest:
            return False
        processes = manifest.get("processes") or []
        if not processes:
            return False
        from rtp_llm.utils.checkpoint_controller import read_process_starttime

        for proc in processes:
            pid = _as_int(proc.get("pid", 0))
            if pid <= 0:
                continue
            starttime = _as_int(proc.get("starttime", 0))
            try:
                live_starttime = read_process_starttime(pid)
            except Exception:
                # Cannot read /proc for this pid -> treat as gone, keep checking.
                continue
            if live_starttime and (starttime <= 0 or live_starttime == starttime):
                return False  # at least one recorded process is still alive
        return True  # no recorded process is alive -> stale

    async def _checkpoint_status_if_any(self) -> Optional[Dict[str, Any]]:
        if not self._level3_enabled:
            return None
        recovery_record, recovery_read_error = self._read_lifecycle_recovery_record()
        if recovery_read_error and self._require_instance_lease:
            self._frontend_lifecycle_state = "RECOVERY_REQUIRED"
            self._frontend_lifecycle_error = recovery_read_error
            return self._synthesize_checkpoint_status("RECOVERY_REQUIRED")
        if recovery_record:
            self._frontend_lifecycle_state = "RECOVERY_REQUIRED"
            self._frontend_lifecycle_error = self._recovery_record_error(
                recovery_record
            )
            return self._synthesize_checkpoint_status("RECOVERY_REQUIRED")
        if self._frontend_lifecycle_state in {"CHECKPOINTING", "RESTORING"}:
            return self._synthesize_checkpoint_status(self._frontend_lifecycle_state)
        try:
            manifest = await self._read_checkpoint_manifest()
        except Exception as e:
            logging.error("failed to read level-3 checkpoint manifest: %s", e)
            self._set_frontend_lifecycle_state("RECOVERY_REQUIRED", str(e))
            return self._synthesize_checkpoint_status("RECOVERY_REQUIRED")

        if self._manifest_is_stale(manifest):
            logging.warning(
                "ignoring stale level-3 checkpoint manifest (all recorded backend "
                "processes are gone; a prior instance generation likely reused the "
                "same control addresses) and discarding it"
            )
            try:
                await self._checkpoint_controller_call(
                    "clear_manifest",
                    tuple(self.control_addresses),
                    namespace=self._manifest_namespace(),
                )
            except Exception as e:
                logging.warning("failed to discard stale checkpoint manifest: %s", e)
            manifest = None

        manifest_state = self._manifest_state(manifest)
        if manifest_state in self.CHECKPOINT_STATES:
            return self._synthesize_checkpoint_status(manifest_state, manifest)
        if manifest_state == "RUNNING" and bool(manifest.get("manifest_exists", True)):
            return self._synthesize_checkpoint_status("RESTORING", manifest)
        if manifest_state and manifest_state != "RUNNING":
            error = f"unexpected checkpoint manifest state {manifest_state}"
            self._set_frontend_lifecycle_state("RECOVERY_REQUIRED", error)
            return self._synthesize_checkpoint_status(
                "RECOVERY_REQUIRED", {**(manifest or {}), "error": error}
            )
        if self._frontend_lifecycle_state == "RECOVERY_REQUIRED":
            return self._synthesize_checkpoint_status(
                self._frontend_lifecycle_state, manifest
            )
        if self._frontend_lifecycle_state == "CHECKPOINTED" and manifest is not None:
            return self._synthesize_checkpoint_status("CHECKPOINTED", manifest)
        if self._frontend_lifecycle_state == "CHECKPOINTED" and manifest is None:
            self._set_frontend_lifecycle_state("")
        return None

    def _level3_precondition_error(self) -> str:
        if self._single_node is False:
            return "level-3 deep sleep currently supports a single node only"
        if self._single_node is None:
            hosts = {
                address.rsplit(":", 1)[0].strip("[]")
                for address in self.control_addresses
            }
            if len(hosts) != 1:
                return "level-3 deep sleep currently supports a single node only"
        return ""

    def _validate_checkpoint_targets(
        self, statuses: Sequence[Dict[str, Any]]
    ) -> List[int]:
        if len(statuses) != len(self.control_addresses):
            raise RuntimeError("cannot collect backend pids from all control ranks")
        if any("error" in status for status in statuses):
            raise RuntimeError("cannot checkpoint ranks with failed terminal status")
        if any(str(status.get("state", "")) != "SLEEPING" for status in statuses):
            raise RuntimeError("all backend ranks must be SLEEPING before checkpoint")
        pids = [_as_int(status.get("process_id", 0)) for status in statuses]
        if any(pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
            raise RuntimeError("invalid or duplicate backend process ids")
        return pids

    @staticmethod
    def _rollback_confirmed_all_running(
        error: BaseException, manifest: Optional[Dict[str, Any]]
    ) -> bool:
        if bool(getattr(error, "all_running", False)) or bool(
            getattr(error, "rollback_confirmed", False)
        ):
            return True
        if not manifest:
            return False
        if bool(manifest.get("all_running", False)):
            return True
        if str(manifest.get("state", "")).upper() == "RUNNING":
            return True
        targets = manifest.get("targets", [])
        return bool(targets) and all(
            isinstance(target, dict)
            and str(target.get("state", "")).upper() == "RUNNING"
            for target in targets
        )

    async def _raw_sleep_statuses(self) -> List[Dict[str, Any]]:
        return await self._broadcast_control_rpc(
            "GetSleepStatus", pb2.EmptyPB(), timeout_s=3
        )

    def _rank_identity_error(self, statuses: Sequence[Dict[str, Any]]) -> str:
        if len(statuses) != len(self.control_addresses):
            return "status coverage is incomplete"
        addresses = [str(status.get("address", "")) for status in statuses]
        if len(set(addresses)) != len(addresses) or set(addresses) != set(
            self.control_addresses
        ):
            return "status addresses do not exactly match control ranks"

        # Older backends do not report these additive identity fields. Once any
        # rank reports them, require a complete identity from every rank.
        has_rank_identity = any(
            status.get("role") or status.get("instance_generation_uuid")
            for status in statuses
        )
        if not has_rank_identity:
            return ""
        roles = [str(status.get("role", "")) for status in statuses]
        generations = [
            str(status.get("instance_generation_uuid", "")) for status in statuses
        ]
        world_ranks = [_as_int(status.get("world_rank", -1), -1) for status in statuses]
        process_ids = [_as_int(status.get("process_id", 0)) for status in statuses]
        if any(not role for role in roles) or len(set(roles)) != 1:
            return "backend ranks report missing or inconsistent roles"
        if any(not generation for generation in generations) or len(
            set(generations)
        ) != len(generations):
            return "backend ranks report missing or duplicate process generations"
        expected_ranks = set(range(len(self.control_addresses)))
        if set(world_ranks) != expected_ranks or len(set(world_ranks)) != len(
            world_ranks
        ):
            return "backend world ranks do not exactly cover the configured rank set"
        if any(process_id <= 0 for process_id in process_ids) or len(
            set(process_ids)
        ) != len(process_ids):
            return "backend ranks report invalid or duplicate process ids"
        return ""

    @staticmethod
    def _same_sleep_epoch(statuses: Sequence[Dict[str, Any]]) -> bool:
        epochs = {_as_int(status.get("sleep_epoch", -1), -1) for status in statuses}
        return len(epochs) == 1 and next(iter(epochs)) >= 0

    def _rank_statuses_match(
        self,
        statuses: Sequence[Dict[str, Any]],
        expected_state: str,
        *,
        require_same_sleep_epoch: bool = False,
    ) -> bool:
        if self._rank_identity_error(statuses):
            return False
        if any("error" in status for status in statuses):
            return False
        if any(str(status.get("state", "")) != expected_state for status in statuses):
            return False
        if require_same_sleep_epoch:
            if not self._same_sleep_epoch(statuses):
                return False
        return True

    async def _poll_rank_state(
        self,
        expected_state: str,
        *,
        require_same_sleep_epoch: bool = False,
    ) -> tuple[bool, List[Dict[str, Any]]]:
        statuses: List[Dict[str, Any]] = []
        for attempt in range(self.LIFECYCLE_STATUS_MAX_ATTEMPTS):
            statuses = await self._raw_sleep_statuses()
            if self._rank_statuses_match(
                statuses,
                expected_state,
                require_same_sleep_epoch=require_same_sleep_epoch,
            ):
                return True, statuses
            if attempt + 1 < self.LIFECYCLE_STATUS_MAX_ATTEMPTS:
                await asyncio.sleep(self.LIFECYCLE_STATUS_POLL_INTERVAL_S)
        return False, statuses

    async def _rollback_sleep_prepare_to_running(self) -> Dict[str, Any]:
        try:
            rollback_results = await self._broadcast_control_rpc(
                "WakeUpServing", pb2.WakeUpRequestPB(), timeout_s=60
            )
        except asyncio.CancelledError:
            rollback_results = [
                {
                    "address": "",
                    "error": "rollback RPC task was cancelled",
                    "grpc_status": "CANCELLED",
                }
            ]
        rollback_failures = [result for result in rollback_results if "error" in result]
        try:
            converged, statuses = await self._poll_rank_state(
                "RUNNING", require_same_sleep_epoch=True
            )
        except asyncio.CancelledError:
            converged = False
            statuses = [
                {
                    "address": "",
                    "error": "rollback status probe was cancelled",
                    "grpc_status": "CANCELLED",
                }
            ]
        if not rollback_failures and converged:
            return {"status": "ok"}
        reason_parts = []
        if rollback_failures:
            reason_parts.append("rollback RPC failed on some control ranks")
        if not converged:
            reason_parts.append("control ranks did not converge to RUNNING")
        return {
            "error": "; ".join(reason_parts),
            "grpc_status": "FAILED_PRECONDITION",
            "recovery_required": True,
            "details": _error_details(rollback_results)
            + [
                {
                    "address": status.get("address", ""),
                    "state": status.get("state", "UNREACHABLE"),
                    **({"error": status["error"]} if "error" in status else {}),
                }
                for status in statuses
            ],
        }

    def _mark_level3_recovery_required(
        self, error: str, details: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        return self._mark_recovery_required(error, details)

    def _mark_recovery_required(
        self,
        error: str,
        details: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self._set_frontend_lifecycle_state("RECOVERY_REQUIRED", error)
        if self._level3_enabled:
            result = self._synthesize_checkpoint_status(
                "RECOVERY_REQUIRED", {"error": error}
            )
        else:
            result = {
                "error": f"RECOVERY_REQUIRED: {error}",
                "grpc_status": "FAILED_PRECONDITION",
                "recovery_required": True,
            }
        if details:
            result["details"] = details
        return result

    async def _initial_lifecycle_status(self, operation: str) -> Dict[str, Any]:
        """Probe the pre-condition state shared by every control rank.

        Contract (intended, not a limitation): sleep and wake_up are atomic,
        uninterruptible instance-wide transitions with no addressable
        intermediate state. Either every control rank is in the same state, or
        the instance is faulted. A mixed rank state -- e.g. some ranks SLEEPING
        while others are still DRAINING/WAKING_UP -- is therefore reported as
        ``RECOVERY_REQUIRED`` (FAILED_PRECONDITION) and the caller must restart
        the instance; we deliberately do NOT try to reconcile the ranks forward
        or backward here. Rationale: level-2 sleep discards GPU memory with no
        backup, so a diverged set of ranks cannot be reconstructed into a known-
        good state -- silently waking into a wrong/partial state is more
        dangerous than an honest restart. In-progress divergence during a single
        commit is a separate, recoverable case handled by ``_converge_commit``
        (which retries laggards up to a bound); this gate only fires when the
        ranks are already inconsistent *before* the operation begins.
        """
        self._refresh_control_addresses_if_needed()
        statuses = await self._raw_sleep_statuses()
        status = self._aggregate_sleep_status(statuses)
        _report_sleep_status_metrics(status)
        if "error" in status:
            return self._recovery_required(
                operation,
                "could not establish the initial rank state",
                statuses,
            )
        state = str(status.get("state", ""))
        known_states = {
            "RUNNING",
            "DRAINING",
            "SUSPENDING",
            "SLEEPING",
            "WAKING_UP",
        }
        if state not in known_states:
            return self._recovery_required(
                operation,
                "observed an invalid initial rank state",
                statuses,
            )
        return status

    def _recovery_required(
        self,
        operation: str,
        reason: str,
        statuses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        details = []
        for status in statuses:
            detail = {"address": status.get("address", "")}
            if "error" in status:
                detail["error"] = status["error"]
                if "grpc_status" in status:
                    detail["grpc_status"] = status["grpc_status"]
            else:
                detail["state"] = status.get("state", "")
            details.append(detail)
        error = f"{operation} {reason}"
        return self._mark_recovery_required(error, details)

    async def _drive_to_terminal(self, coro: Any) -> Any:
        """Run an irreversible lifecycle transition to completion even if the
        driving request coroutine is cancelled.

        Once the commit phase has started running the GPU-release / GPU-restore
        hooks there is no consistent rollback: freed device memory (or a level-2
        disk dump/reload) cannot be un-done, so the only valid terminal states
        are the two endpoints of the transition -- every rank RUNNING or every
        rank SLEEPING. If the frontend request task is cancelled midway (a
        concurrent /health probe tearing down a shared channel, a worker
        recycle, a client disconnect) we must NOT abandon the transition
        half-committed and release the lifecycle lease -- that is the
        control-plane split brain that leaves the instance with half its device
        memory freed and no owner driving it to a consistent state.

        So we absorb the cancellation and keep awaiting until the backend
        converges, then report the true terminal state to whoever is left. The
        underlying `_converge_commit` is itself bounded (COMMIT_MAX_ATTEMPTS x
        per-attempt timeout), so this cannot hang indefinitely.
        """
        task = asyncio.ensure_future(coro)
        absorbed_cancel = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.done():
                    # The transition already finished; honor its result and let
                    # the cancellation die here (the irreversible work is done).
                    result = task.result()
                    break
                # Still committing an irreversible transition -- swallow the
                # cancellation and keep waiting for the backend to converge.
                absorbed_cancel = True
                continue
        if absorbed_cancel:
            logging.warning(
                "lifecycle commit was driven to its terminal state despite "
                "request cancellation; absorbed the cancel to avoid a "
                "half-committed instance"
            )
        return result

    async def _converge_commit(
        self,
        operation: str,
        rpc_name: str,
        commit_request: Any,
        timeout_s: float,
        transitional_state: str,
        final_state: str,
    ) -> Dict[str, Any]:
        result, _ = await self._converge_commit_with_statuses(
            operation=operation,
            rpc_name=rpc_name,
            commit_request=commit_request,
            timeout_s=timeout_s,
            transitional_state=transitional_state,
            final_state=final_state,
        )
        return result

    async def _converge_commit_with_statuses(
        self,
        operation: str,
        rpc_name: str,
        commit_request: Any,
        timeout_s: float,
        transitional_state: str,
        final_state: str,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        pending = list(self.control_addresses)
        last_statuses: List[Dict[str, Any]] = []
        for _ in range(self.COMMIT_MAX_ATTEMPTS):
            await self._broadcast_control_rpc_to(
                pending, rpc_name, commit_request, timeout_s
            )
            last_statuses = await self._raw_sleep_statuses()
            if len(last_statuses) != len(self.control_addresses):
                return (
                    self._recovery_required(
                        operation,
                        "status coverage is incomplete",
                        last_statuses,
                    ),
                    last_statuses,
                )
            if any("error" in status for status in last_statuses):
                return (
                    self._recovery_required(
                        operation,
                        "status probe failed",
                        last_statuses,
                    ),
                    last_statuses,
                )
            identity_error = self._rank_identity_error(last_statuses)
            if identity_error or not self._same_sleep_epoch(last_statuses):
                reason = identity_error or "sleep epochs differ across control ranks"
                return (
                    self._recovery_required(
                        operation,
                        reason,
                        last_statuses,
                    ),
                    last_statuses,
                )
            states = [str(status.get("state", "")) for status in last_statuses]
            allowed_states = {transitional_state, final_state}
            if any(state not in allowed_states for state in states):
                return (
                    self._recovery_required(
                        operation,
                        "observed an unrecoverable rank state",
                        last_statuses,
                    ),
                    last_statuses,
                )
            pending = [
                status["address"]
                for status, state in zip(last_statuses, states)
                if state == transitional_state
            ]
            if not pending:
                return {"status": "ok"}, last_statuses
        return (
            self._recovery_required(
                operation,
                f"did not converge after {self.COMMIT_MAX_ATTEMPTS} commit attempts",
                last_statuses,
            ),
            last_statuses,
        )

    def _aggregate_sleep_status(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successes = [result for result in results if "error" not in result]
        failures = [result for result in results if "error" in result]
        if not successes:
            return {
                "error": "Failed to get sleep status from all control ranks",
                "grpc_status": (
                    failures[0].get("grpc_status", "UNAVAILABLE")
                    if failures
                    else "UNAVAILABLE"
                ),
                "details": _error_details(results),
            }
        if failures:
            return {
                "error": "Failed to get sleep status from some control ranks",
                "grpc_status": failures[0].get("grpc_status", "UNAVAILABLE"),
                "details": _error_details(results),
            }

        identity_error = self._rank_identity_error(successes)
        if identity_error:
            return {
                "error": f"Invalid sleep rank identity: {identity_error}",
                "grpc_status": "FAILED_PRECONDITION",
            }

        states = {str(result.get("state", "")) for result in successes}
        sleep_epochs = {
            _as_int(result.get("sleep_epoch", -1), -1) for result in successes
        }
        enabled = {
            bool(result.get("sleep_mode_enabled", False)) for result in successes
        }
        effective = {bool(result.get("effective", False)) for result in successes}
        gpu_states = {str(result.get("gpu_resource_state", "")) for result in successes}
        kv_states = {str(result.get("kv_memory_state", "")) for result in successes}
        supported_levels = {
            tuple(result.get("supported_levels", [])) for result in successes
        }
        supported_modes = {
            tuple(result.get("supported_modes", [])) for result in successes
        }
        if (
            len(states) != 1
            or len(sleep_epochs) != 1
            or next(iter(sleep_epochs)) < 0
            or len(enabled) != 1
            or len(effective) != 1
            or len(gpu_states) != 1
            or len(kv_states) != 1
            or len(supported_levels) != 1
            or len(supported_modes) != 1
        ):
            return {
                "error": "Sleep status did not converge across control ranks",
                "grpc_status": "FAILED_PRECONDITION",
            }

        aggregate = dict(successes[0])
        aggregate["sleep_epoch"] = next(iter(sleep_epochs))
        aggregate["active_request_count"] = sum(
            _as_int(result.get("active_request_count", 0)) for result in successes
        )
        aggregate["active_cache_transfer_count"] = sum(
            _as_int(result.get("active_cache_transfer_count", 0))
            for result in successes
        )
        aggregate["device_kv_cache_valid"] = all(
            bool(result.get("device_kv_cache_valid", False)) for result in successes
        )
        aggregate.pop("address", None)
        aggregate.pop("status", None)
        coverage_error = self._control_address_coverage_error()
        if coverage_error:
            aggregate["effective"] = False
            aggregate["supported_levels"] = []
            aggregate["supported_modes"] = []
            aggregate["disabled_reason"] = coverage_error
        return aggregate

    async def _complete_level3_sleep(
        self, commit_request: Any, timeout_s: float
    ) -> Dict[str, Any]:
        try:
            result, terminal_statuses = await self._converge_commit_with_statuses(
                operation="commit sleep",
                rpc_name="SleepServing",
                commit_request=commit_request,
                timeout_s=timeout_s,
                transitional_state="DRAINING",
                final_state="SLEEPING",
            )
        except asyncio.CancelledError:
            return self._mark_level3_recovery_required(
                "commit sleep was cancelled after the irreversible Level3 phase began"
            )
        except Exception as e:
            return self._mark_level3_recovery_required(
                f"commit sleep failed after the irreversible Level3 phase began: {e}"
            )
        if "error" in result:
            return result

        self._terminal_sleep_statuses = [dict(status) for status in terminal_statuses]
        self._terminal_sleep_status = self._aggregate_sleep_status(
            self._terminal_sleep_statuses
        )
        checkpoint_started = False
        try:
            self._validate_checkpoint_targets(self._terminal_sleep_statuses)
            self._set_frontend_lifecycle_state("CHECKPOINTING")
            checkpoint_started = True
            await self._checkpoint_controller_call(
                "checkpoint_all",
                tuple(self.control_addresses),
                tuple(self._terminal_sleep_statuses),
                namespace=self._manifest_namespace(),
                holder_instance=self._keeper_holder_instance(),
                team=self._keeper_team(),
            )
        except asyncio.CancelledError:
            return self._mark_level3_recovery_required(
                "backend checkpoint was cancelled after the irreversible Level3 phase began"
            )
        except Exception as e:
            logging.error("level-3 backend checkpoint failed: %s", e)
            manifest = getattr(e, "checkpoint_status", None)
            manifest_error = None
            try:
                durable_manifest = await self._read_checkpoint_manifest()
                if durable_manifest is not None:
                    manifest = durable_manifest
            except Exception as read_error:
                manifest_error = read_error
                logging.error(
                    "failed to inspect manifest after checkpoint failure: %s",
                    read_error,
                )
            all_running = (
                not checkpoint_started
                or (manifest is None and manifest_error is None)
                or self._rollback_confirmed_all_running(e, manifest)
            )
            if all_running and manifest_error is None:
                self._set_frontend_lifecycle_state("RESTORING", str(e))
                if checkpoint_started and manifest is not None:
                    try:
                        finalized = await self._checkpoint_controller_call(
                            "restore_all",
                            tuple(self.control_addresses),
                            namespace=self._manifest_namespace(),
                            holder_instance=self._keeper_holder_instance(),
                            team=self._keeper_team(),
                        )
                        if finalized is False:
                            raise RuntimeError(
                                "rollback manifest disappeared before finalization"
                            )
                    except asyncio.CancelledError:
                        recovery_error = (
                            f"checkpoint failed ({e}); rollback manifest "
                            "finalization was cancelled"
                        )
                        return self._mark_level3_recovery_required(recovery_error)
                    except Exception as finalize_error:
                        recovery_error = (
                            f"checkpoint failed ({e}); rollback manifest "
                            f"finalization failed ({finalize_error})"
                        )
                        return self._mark_level3_recovery_required(recovery_error)
                try:
                    compensation = await self._wake_backend_locked(drive_commit=False)
                except asyncio.CancelledError:
                    return self._mark_level3_recovery_required(
                        f"checkpoint failed ({e}); backend wake compensation was cancelled"
                    )
                except Exception as compensation_error:
                    return self._mark_level3_recovery_required(
                        f"checkpoint failed ({e}); backend wake compensation failed "
                        f"({compensation_error})"
                    )
                if "error" not in compensation:
                    self._set_frontend_lifecycle_state("")
                    return {
                        "error": "Failed to checkpoint level-3 backend processes; "
                        "backend wake compensation completed",
                        "grpc_status": "FAILED_PRECONDITION",
                        "recovered": True,
                        "details": [{"error": str(e)}],
                    }
                recovery_error = (
                    f"checkpoint failed ({e}); backend wake compensation failed "
                    f"({compensation.get('error', 'unknown error')})"
                )
            else:
                recovery_error = f"checkpoint failed with uncertain driver state: {e}"
                if manifest_error is not None:
                    recovery_error += f"; manifest inspection failed: {manifest_error}"
            return self._mark_level3_recovery_required(recovery_error)

        self._set_frontend_lifecycle_state("CHECKPOINTED")
        return {"status": "ok"}

    async def sleep_serving(self, req: Any) -> Dict[str, Any]:
        """Trigger engine sleep on every lifecycle control rank."""
        start_time = time.time() * 1000
        req, validation_error = normalize_sleep_request(req, self._configured_level)
        if validation_error is not None:
            return validation_error
        assert req is not None
        level = int(req["level"])
        if level == 0:
            return {
                "error": "sleep level=0 state-preserving sleep is defined but not implemented",
                "grpc_status": "UNIMPLEMENTED",
                "supported_levels": [self._configured_level],
                "supported_modes": ["wait", "abort"],
            }
        if not self._sleep_enabled:
            return {
                "error": "sleep mode is disabled",
                "grpc_status": "UNIMPLEMENTED",
                "sleep_mode_enabled": False,
                "effective": False,
                "supported_levels": [],
                "supported_modes": [],
            }
        if self._level3_enabled:
            precondition_error = self._level3_precondition_error()
            if precondition_error:
                return {
                    "error": precondition_error,
                    "grpc_status": "FAILED_PRECONDITION",
                }
            checkpoint_status = await self._checkpoint_status_if_any()
            if checkpoint_status is not None:
                if checkpoint_status.get("state") == "CHECKPOINTED":
                    return {"status": "ok"}
                return {
                    **checkpoint_status,
                    "error": checkpoint_status.get(
                        "error", "level-3 lifecycle operation is already in progress"
                    ),
                    "grpc_status": checkpoint_status.get(
                        "grpc_status", "FAILED_PRECONDITION"
                    ),
                }
        async with self._lifecycle_lock:
            if self._level3_enabled:
                await self._resolve_instance_identity()
            lease_record, lease_error = self._acquire_lifecycle_lease("sleep")
            if lease_error:
                result = lease_error
            else:
                try:
                    result = await self._sleep_serving_locked(req)
                finally:
                    if self._frontend_lifecycle_state == "RECOVERY_REQUIRED":
                        logging.error(
                            "retaining lifecycle lease because recovery is required"
                        )
                    else:
                        self._release_lifecycle_lease(lease_record)
        # sleep is a rare control-plane action: the synchronous response is the
        # authoritative outcome and a failure is handled as an incident, so a
        # single grep-able log line per call is the useful signal (no QPS metric).
        duration_ms = time.time() * 1000 - start_time
        if "error" in result:
            logging.warning(
                "sleep action failed in %.0fms: %s (grpc_status=%s)",
                duration_ms,
                result.get("error"),
                result.get("grpc_status", "UNKNOWN"),
            )
        else:
            logging.info("sleep action completed ok in %.0fms", duration_ms)
        return result

    async def _sleep_serving_locked(self, req: Any) -> Dict[str, Any]:
        try:
            level = int(req["level"])
            timeout_ms = int(req.get("timeout_ms", 60 * 60 * 1000))
            mode = str(req.get("mode", "wait"))
            tags = list(req.get("tags", []))
            self._refresh_control_addresses_if_needed()
            checkpoint_status = (
                await self._checkpoint_status_if_any() if self._level3_enabled else None
            )
            if checkpoint_status is not None:
                if level == 3 and checkpoint_status.get("state") == "CHECKPOINTED":
                    return {"status": "ok"}
                return {
                    **checkpoint_status,
                    "error": checkpoint_status.get(
                        "error",
                        "level-3 lifecycle operation is already in progress",
                    ),
                    "grpc_status": checkpoint_status.get(
                        "grpc_status", "FAILED_PRECONDITION"
                    ),
                }
            if level == 3:
                precondition_error = self._level3_precondition_error()
                if precondition_error:
                    return {
                        "error": precondition_error,
                        "grpc_status": "FAILED_PRECONDITION",
                    }
                try:
                    await self._checkpoint_controller_call(
                        "preflight",
                        tuple(self.control_addresses),
                        namespace=self._manifest_namespace(),
                    )
                except Exception as e:
                    return {
                        "error": f"level-3 checkpoint preflight failed: {e}",
                        "grpc_status": "FAILED_PRECONDITION",
                    }
            status = await self._initial_lifecycle_status("sleep")
            if "error" in status:
                return status
            self._level3_wake_completed = False
            if not bool(status.get("effective", False)):
                return {
                    "error": status.get("disabled_reason", "sleep mode is disabled"),
                    "grpc_status": "UNIMPLEMENTED",
                    "sleep_mode_enabled": bool(status.get("sleep_mode_enabled", False)),
                    "effective": False,
                    "supported_levels": status.get("supported_levels", []),
                    "supported_modes": status.get("supported_modes", []),
                }
            request = pb2.SleepRequestPB(
                level=level,
                mode=mode,
                timeout_ms=timeout_ms,
                reason=str(req.get("reason", "")),
                tags=list(tags),
            )
            prepare_request = pb2.SleepRequestPB()
            prepare_request.CopyFrom(request)
            prepare_request.prepare_only = True
            commit_request = pb2.SleepRequestPB()
            commit_request.CopyFrom(request)
            commit_request.commit_only = True
            commit_request.timeout_ms = 0

            # prepare blocks on drain; leave headroom on top of drain timeout.
            # Only after every rank is drained do we send commit, avoiding a
            # partially sleeping instance when one rank times out.
            timeout_s = max(60.0, timeout_ms / 1000.0 + 30.0)
            try:
                prepare_results = await self._broadcast_control_rpc(
                    "SleepServing", prepare_request, timeout_s
                )
            except asyncio.CancelledError:
                # Prepare only closes admission and drains in-flight work -- no
                # device memory has been released yet, so this phase is fully
                # reversible. Roll every rank that may have entered DRAINING
                # back to RUNNING (uninterruptibly, so the rollback itself
                # completes) before honoring the cancellation. Without this the
                # instance would be stuck admission-closed with no owner.
                logging.warning(
                    "sleep prepare cancelled; rolling back drain to RUNNING"
                )
                rollback_result = await self._drive_to_terminal(
                    self._rollback_sleep_prepare_to_running()
                )
                if "error" in rollback_result:
                    recovery_error = (
                        "sleep prepare cancellation rollback failed: "
                        f"{rollback_result['error']}"
                    )
                    self._mark_recovery_required(
                        recovery_error, rollback_result.get("details")
                    )
                    logging.error(recovery_error)
                raise
            failures = [result for result in prepare_results if "error" in result]
            if failures:
                rollback_result = await self._drive_to_terminal(
                    self._rollback_sleep_prepare_to_running()
                )
                if "error" in rollback_result:
                    details = _error_details(prepare_results) + rollback_result.get(
                        "details", []
                    )
                    recovery_error = (
                        "sleep prepare failed and rollback did not establish RUNNING: "
                        f"{rollback_result['error']}"
                    )
                    return self._mark_recovery_required(recovery_error, details)
                return {
                    "error": "Failed to prepare sleep on some control ranks (rolled back)",
                    "grpc_status": failures[0].get("grpc_status", "UNKNOWN"),
                    "details": _error_details(prepare_results),
                }

            prepared = True
            prepared_statuses: List[Dict[str, Any]] = []
            if level == 3:
                prepared, prepared_statuses = await self._poll_rank_state(
                    "DRAINING", require_same_sleep_epoch=True
                )
            if level == 3 and not prepared:
                rollback_result = await self._drive_to_terminal(
                    self._rollback_sleep_prepare_to_running()
                )
                details = [
                    {
                        "address": status.get("address", ""),
                        "state": status.get("state", "UNREACHABLE"),
                        **({"error": status["error"]} if "error" in status else {}),
                    }
                    for status in prepared_statuses
                ]
                if "error" in rollback_result:
                    details += rollback_result.get("details", [])
                    recovery_error = (
                        "sleep prepare status barrier failed and rollback did not "
                        f"establish RUNNING: {rollback_result['error']}"
                    )
                    return self._mark_level3_recovery_required(recovery_error, details)
                return {
                    "error": "Sleep prepare did not converge across all control ranks "
                    "(rolled back)",
                    "grpc_status": "FAILED_PRECONDITION",
                    "details": details,
                }

            # commit runs the GPU-release hooks; for level-2 that includes dumping
            # the ~weights-sized raw backup to disk, which can take far longer than
            # a level-1 tms pause. Reuse the drain-derived headroom so a slow dump
            # does not spuriously trip the commit deadline. Once commit starts the
            # device memory release is irreversible, so drive it to the terminal
            # SLEEPING state even if this request is cancelled.
            completion = (
                self._complete_level3_sleep(commit_request, timeout_s)
                if level == 3
                else self._converge_commit(
                    operation="commit sleep",
                    rpc_name="SleepServing",
                    commit_request=commit_request,
                    timeout_s=timeout_s,
                    transitional_state="DRAINING",
                    final_state="SLEEPING",
                )
            )
            return await self._drive_to_terminal(completion)
        except grpc.aio.AioRpcError as e:
            logging.error(f"Sleep serving failed: {e.details()}")
            return {
                "error": f"Failed to sleep serving: {e.details()}",
                "grpc_status": e.code().name,
            }
        except Exception as e:
            logging.error(f"Sleep serving failed: {e}")
            return {"error": f"Failed to sleep serving: {str(e)}"}

    async def wake_up_serving(self, req: Any = None) -> Dict[str, Any]:
        """Trigger engine wake_up on every lifecycle control rank."""
        start_time = time.time() * 1000
        req, validation_error = normalize_wake_request(req)
        if validation_error is not None:
            return validation_error
        assert req is not None
        if not self._sleep_enabled:
            return {
                "error": "sleep mode is disabled",
                "grpc_status": "UNIMPLEMENTED",
                "sleep_mode_enabled": False,
                "effective": False,
                "supported_levels": [],
                "supported_modes": [],
            }
        async with self._lifecycle_lock:
            lease_record, lease_error = self._acquire_lifecycle_lease("wake_up")
            if lease_error:
                result = lease_error
            else:
                try:
                    result = await self._wake_up_serving_locked(req)
                finally:
                    if self._frontend_lifecycle_state == "RECOVERY_REQUIRED":
                        logging.error(
                            "retaining lifecycle lease because recovery is required"
                        )
                    else:
                        self._release_lifecycle_lease(lease_record)
        duration_ms = time.time() * 1000 - start_time
        if "error" in result:
            logging.warning(
                "wake_up action failed in %.0fms: %s (grpc_status=%s)",
                duration_ms,
                result.get("error"),
                result.get("grpc_status", "UNKNOWN"),
            )
        else:
            logging.info("wake_up action completed ok in %.0fms", duration_ms)
        return result

    async def _wake_up_serving_locked(self, req: Any = None) -> Dict[str, Any]:
        try:
            self._refresh_control_addresses_if_needed()
            checkpoint_status = (
                await self._checkpoint_status_if_any() if self._level3_enabled else None
            )
            if checkpoint_status is not None:
                return await self._drive_to_terminal(
                    self._restore_level3_and_wake_backend(checkpoint_status)
                )
            return await self._wake_backend_locked(
                running_is_success=self._level3_wake_completed
            )
        except grpc.aio.AioRpcError as e:
            logging.error(f"Wake_up serving failed: {e.details()}")
            return {
                "error": f"Failed to wake_up serving: {e.details()}",
                "grpc_status": e.code().name,
            }
        except Exception as e:
            logging.error(f"Wake_up serving failed: {e}")
            return {"error": f"Failed to wake_up serving: {str(e)}"}

    async def _restore_level3_and_wake_backend(
        self, checkpoint_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._set_frontend_lifecycle_state("RESTORING")
        try:
            restored = await self._checkpoint_controller_call(
                "restore_all",
                tuple(self.control_addresses),
                namespace=self._manifest_namespace(),
                holder_instance=self._keeper_holder_instance(),
                team=self._keeper_team(),
            )
            if restored is False:
                raise RuntimeError(
                    "checkpoint manifest disappeared before restore completed"
                )
            if isinstance(restored, dict) and not restored.get("processes"):
                raise RuntimeError(
                    "checkpoint manifest disappeared before restore completed"
                )
        except asyncio.CancelledError:
            error = "level-3 backend restore task was cancelled"
            logging.error(error)
            return self._mark_level3_recovery_required(error)
        except Exception as e:
            error = f"level-3 backend restore failed: {e}"
            logging.error(error)
            return self._mark_level3_recovery_required(error)

        # This is deliberately the first backend lifecycle interaction after
        # restore_all. A checkpointed process cannot answer even status RPCs.
        try:
            result = await self._wake_backend_locked(
                drive_commit=False,
                running_is_success=True,
            )
        except asyncio.CancelledError:
            return self._mark_level3_recovery_required(
                "backend wake was cancelled after level-3 restore"
            )
        except Exception as e:
            return self._mark_level3_recovery_required(
                f"backend wake failed after level-3 restore: {e}"
            )
        if "error" in result:
            if result.get("recovery_required"):
                return result
            error = f"backend wake failed after level-3 restore: {result['error']}"
            return self._mark_level3_recovery_required(error, result.get("details"))
        self._set_frontend_lifecycle_state("")
        self._terminal_sleep_statuses = []
        self._terminal_sleep_status = {}
        self._level3_wake_completed = True
        return result

    async def _wake_backend_locked(
        self,
        *,
        drive_commit: bool = True,
        running_is_success: bool = False,
    ) -> Dict[str, Any]:
        status = await self._initial_lifecycle_status("wake_up")
        if "error" in status:
            return status
        if status.get("state") == "RUNNING" and running_is_success:
            return {"status": "ok"}
        if not bool(status.get("effective", False)):
            return {
                "error": status.get("disabled_reason", "sleep mode is disabled"),
                "grpc_status": "UNIMPLEMENTED",
                "sleep_mode_enabled": bool(status.get("sleep_mode_enabled", False)),
                "effective": False,
                "supported_levels": status.get("supported_levels", []),
                "supported_modes": status.get("supported_modes", []),
            }
        prepare_request = pb2.WakeUpRequestPB(prepare_only=True)
        commit_request = pb2.WakeUpRequestPB(commit_only=True)

        try:
            prepare_results = await self._broadcast_control_rpc(
                "WakeUpServing", prepare_request, timeout_s=600
            )
        except asyncio.CancelledError:
            self._mark_recovery_required(
                "prepare wake_up was cancelled after backend mutation may have begun"
            )
            raise
        failures = [result for result in prepare_results if "error" in result]
        if failures:
            error = "Failed to prepare wake_up on some control ranks"
            details = _error_details(prepare_results)
            return self._mark_recovery_required(error, details)

        completion = self._converge_commit(
            operation="commit wake_up",
            rpc_name="WakeUpServing",
            commit_request=commit_request,
            timeout_s=600,
            transitional_state="WAKING_UP",
            final_state="RUNNING",
        )
        if drive_commit:
            return await self._drive_to_terminal(completion)
        return await completion

    async def get_sleep_status(self, req: Any = None) -> Dict[str, Any]:
        """Get aggregate sleep lifecycle status from every control rank."""
        try:
            self._refresh_control_addresses_if_needed()
            checkpoint_status = (
                await self._checkpoint_status_if_any() if self._level3_enabled else None
            )
            if checkpoint_status is not None:
                return checkpoint_status
            request = pb2.EmptyPB()
            results = await self._broadcast_control_rpc(
                "GetSleepStatus", request, timeout_s=3
            )
            status = self._aggregate_sleep_status(results)
            _report_sleep_status_metrics(status)
            return status
        except Exception as e:
            logging.error(f"Get sleep status failed: {e}")
            return {"error": f"Failed to get sleep status: {str(e)}"}

    async def is_sleeping(self, req: Any = None) -> Dict[str, Any]:
        status = await self.get_sleep_status(req)
        if "error" in status:
            return status
        return {
            "is_sleeping": status.get("state")
            in {"SLEEPING", "CHECKPOINTING", "CHECKPOINTED"},
            "sleep_mode_enabled": bool(status.get("sleep_mode_enabled", False)),
            "effective": bool(status.get("effective", False)),
            "supported_levels": status.get("supported_levels", []),
            "supported_modes": status.get("supported_modes", []),
            "state": status.get("state", ""),
            "disabled_reason": status.get("disabled_reason", ""),
        }

    async def start_profile(self, req: Any) -> Dict[str, Any]:
        """Start profiling switch in backend process"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            if req is None:
                req = {}
            request = pb2.StartProfileRequestPB(
                trace_name=str(req.get("trace_name", "")),
                start_step=int(req.get("start_step", 0)),
                num_steps=int(req.get("num_steps", 0)),
                enable_all_rank=bool(
                    req.get("enable_all_rank", req.get("all_tp", False))
                ),
            )
            await self.stub.StartProfile(request, timeout=3)
            return {"status": "ok"}

        except Exception as e:
            logging.error(f"Start profile failed: {e}")
            return {"error": f"Failed to start profile: {str(e)}"}

    async def update_eplb_config(self, req: Any) -> Dict[str, Any]:
        """Update EPLB config - this would need to be implemented based on your requirements"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            epld_req = pb2.UpdateEplbConfigRequestPB(
                mode=req.get("mode", "NONE"),
                update_time=int(time.time()),
            )
            await self.stub.UpdateEplbConfig(epld_req)
            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Update EPLB config failed: {e}")
            return {"error": f"Failed to update EPLB config: {str(e)}"}

    async def update_scheduler_info(self, req: Any) -> Dict[str, Any]:
        """Update scheduler info on all DP addresses"""
        try:
            if isinstance(req, str):
                req = json.loads(req)
            update_schedule_info_req = pb2.UpdateSchedulerInfoRequestPB(
                scheduler_info=json.dumps(req)
            )

            async def send_to_address(address: str):
                await self._ensure_dp_connection(address)
                await self._dp_stubs[address].UpdateSchedulerInfo(
                    update_schedule_info_req
                )

            tasks = [send_to_address(addr) for addr in self.dp_addresses]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            errors = [
                f"{self.dp_addresses[i]}: {str(r)}"
                for i, r in enumerate(results)
                if isinstance(r, Exception)
            ]
            if errors:
                logging.error(
                    f"Update scheduler info failed on some addresses: {errors}"
                )
                return {"error": f"Failed on some addresses: {errors}"}

            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Update scheduler info failed: {e}")
            return {"error": f"Failed to update scheduler info: {str(e)}"}

    async def post_request(self, uri: str, req: Dict[str, Any]) -> Dict[str, Any]:
        """Generic POST request handler - routes to appropriate method based on URI"""
        try:
            if (
                self._level3_enabled
                and uri
                not in {
                    "sleep",
                    "wake_up",
                    "is_sleeping",
                    "sleep_status",
                }
                and self._frontend_lifecycle_state in self.CHECKPOINT_STATES
            ):
                checkpoint_status = self._synthesize_checkpoint_status(
                    self._frontend_lifecycle_state
                )
                if checkpoint_status:
                    return {
                        **checkpoint_status,
                        "error": "backend RPC is unavailable during level-3 deep sleep",
                        "grpc_status": "FAILED_PRECONDITION",
                    }
            if uri == "health_check":
                return await self.health_check()
            elif uri == "cache_status":
                return await self.get_cache_status(req)
            elif uri == "worker_status":
                return await self.get_worker_status(req)
            elif uri == "set_log_level":
                return await self.set_log_level(req)
            elif uri == "sleep":
                return await self.sleep_serving(req)
            elif uri == "wake_up":
                return await self.wake_up_serving(req)
            elif uri == "is_sleeping":
                return await self.is_sleeping(req)
            elif uri == "sleep_status":
                return await self.get_sleep_status(req)
            elif uri == "start_profile":
                return await self.start_profile(req)
            elif uri == "update_eplb_config":
                return await self.update_eplb_config(req)
            elif uri == "update_scheduler_info":
                return await self.update_scheduler_info(req)
            else:
                # Default case - return empty success
                return {"status": "ok"}
        except Exception as e:
            logging.error(f"POST request to {uri} failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
