import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

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


# grpc_status values that map to a deterministic client error (HTTP 4xx / 501):
# retrying will not change the outcome. When control ranks fail with different
# statuses we prefer to report one of these over a transient server-side status
# (UNAVAILABLE, DEADLINE_EXCEEDED, ...) that a client would otherwise retry forever.
_DETERMINISTIC_GRPC_STATUSES = (
    "INVALID_ARGUMENT",
    "FAILED_PRECONDITION",
    "UNIMPLEMENTED",
    "NOT_FOUND",
    "PERMISSION_DENIED",
    "OUT_OF_RANGE",
    "ALREADY_EXISTS",
)


def _representative_failure(failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the failure whose grpc_status best represents the aggregate outcome.

    When ranks fail with different statuses, a deterministic client error (e.g.
    INVALID_ARGUMENT from one rank) is more actionable than a transient server
    error (e.g. UNAVAILABLE from another): the former tells the client to fix the
    request (HTTP 4xx), the latter invites an infinite retry (HTTP 5xx). Prefer
    the first deterministic failure; fall back to the first failure otherwise.
    """
    for failure in failures:
        if failure.get("grpc_status") in _DETERMINISTIC_GRPC_STATUSES:
            return failure
    return failures[0]


def _configured_sleep_level() -> int:
    """The single non-zero sleep level this process was started with.

    Reads SLEEP_MODE_LEVEL, which --sleep-mode-level is mirrored into for every
    process in the instance at startup (see server_args). Kept as a local env
    read so the frontend coordinator does not pull in the model_loader package
    just to learn a startup scalar; the backend controller stays the runtime
    authority and rejects a level mismatch on the actual sleep request.
    """
    try:
        return int(os.environ.get("SLEEP_MODE_LEVEL", "1"))
    except (TypeError, ValueError):
        return 1


def _report_metric_if_ready(metric: Any, value: float) -> None:
    if not kmonitor.is_inited:
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


class GrpcClientWrapper:
    """Wrapper for direct gRPC calls to replace async_request_server"""

    LIFECYCLE_LEASE_KEY = "rtp_llm_instance_lifecycle_lease"
    COMMIT_MAX_ATTEMPTS = 3

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
        self._lifecycle_holder = uuid.uuid4().hex
        # This lock handles re-entrancy within one frontend. The TCPStore lease
        # below is the authoritative instance-wide serialization mechanism.
        self._lifecycle_lock = asyncio.Lock()
        self._dp_channels: Dict[str, Any] = {}
        self._dp_stubs: Dict[str, Any] = {}

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

    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            await self._ensure_connection()
            # Using a simple request to check if server is responsive
            request = pb2.EmptyPB()
            await self.stub.CheckHealth(request, timeout=1)
            return {"status": "ok"}
        except Exception as e:
            return {
                "status": "error",
                "message": e,
            }

    async def get_cache_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cache status from all DP backends.

        Returns a list of per-DP results under the ``"results"`` key.
        If no backend is reachable, returns ``{"error": ...}``.
        """
        start_time = time.time() * 1000
        request = pb2.CacheVersionPB(
            latest_cache_version=query_params.get("latest_cache_version", -1),
            need_cache_keys=query_params.get("need_cache_keys", True),
        )
        results = []
        for addr in self.dp_addresses:
            try:
                await self._ensure_dp_connection(addr)
                response = await self._dp_stubs[addr].GetCacheStatus(request, timeout=1)
                result = MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )
                result["address"] = addr
                results.append(result)
            except Exception as e:
                logging.warning(f"Get cache status from {addr} failed: {e}")
        kmonitor.report(AccMetrics.CACHE_STATUS_QPS_METRIC, 1)
        kmonitor.report(
            GaugeMetrics.CACHE_STATUS_QPS_LATENCY_METRIC,
            time.time() * 1000 - start_time,
        )
        if not results:
            return {"error": "No backend available for cache_status"}
        # For backward compatibility, merge first result's fields at top level
        merged = dict(results[0])
        merged["results"] = results
        merged["dp_size"] = len(results)
        return merged

    async def get_worker_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get worker status from all DP backends.

        Returns a list of per-DP results under the ``"results"`` key.
        If no backend is reachable, returns ``{"error": ...}``.
        """
        start_time = time.time() * 1000
        request = pb2.StatusVersionPB(
            latest_cache_version=query_params.get("latest_cache_version", -1),
            latest_finished_version=query_params.get("latest_finished_version", -1),
        )
        results = []
        for addr in self.dp_addresses:
            try:
                await self._ensure_dp_connection(addr)
                response = await self._dp_stubs[addr].GetWorkerStatus(
                    request, timeout=1
                )
                result = MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )
                result["address"] = addr
                results.append(result)
            except Exception as e:
                logging.warning(f"Get worker status from {addr} failed: {e}")
        kmonitor.report(AccMetrics.WORKER_STATUS_QPS_METRIC, 1)
        kmonitor.report(
            GaugeMetrics.WORKER_STATUS_QPS_LANTENCY_METRIC,
            time.time() * 1000 - start_time,
        )
        if not results:
            return {"error": "No backend available for worker_status"}
        merged = dict(results[0])
        merged["results"] = results
        merged["dp_size"] = len(results)
        return merged

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
        return json.dumps(
            {"holder": self._lifecycle_holder, "operation": operation},
            sort_keys=True,
            separators=(",", ":"),
        )

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
        record = self._lease_record(operation)
        try:
            current = store.compare_set(self.LIFECYCLE_LEASE_KEY, "", record)
            if isinstance(current, bytes):
                current = current.decode("utf-8")
        except Exception as e:
            return None, {
                "error": f"instance-wide lifecycle coordination failed: {e}",
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
            self._lifecycle_store.compare_set(self.LIFECYCLE_LEASE_KEY, record, "")
        except Exception as e:
            # Fail closed: an uncertain/non-released record must continue blocking
            # lifecycle operations until the instance (and its TCPStore) restarts.
            logging.error("failed to release instance lifecycle lease: %s", e)

    @staticmethod
    def _status_probe_timeout(operation_timeout_s: float) -> float:
        """Derive the GetSleepStatus probe timeout from the operation it verifies.

        A fixed 3s probe is tighter than the commit it confirms (whose timeout can
        reach hundreds of seconds for a level-2 checkpoint reload), so a merely slow
        rank was being misjudged as failed. Scale the probe with the operation but
        keep it bounded: a status probe is cheap, so [10s, 30s] is ample headroom
        while never being shorter than a quick operation.
        """
        return max(10.0, min(operation_timeout_s, 30.0))

    async def _raw_sleep_statuses(
        self, timeout_s: float = 10.0
    ) -> List[Dict[str, Any]]:
        return await self._broadcast_control_rpc(
            "GetSleepStatus", pb2.EmptyPB(), timeout_s=timeout_s
        )

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
                operation, "could not establish the initial rank state", statuses
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
                operation, "observed an invalid initial rank state", statuses
            )
        return status

    def _recovery_required(
        self,
        operation: str,
        reason: str,
        statuses: List[Dict[str, Any]],
        extra_details: Optional[List[Dict[str, Any]]] = None,
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
        # Commit-RPC-level failures (tagged source="commit_rpc") are folded in so
        # the response carries the real commit error, not just the derived states.
        for extra in extra_details or []:
            tagged = dict(extra)
            tagged["source"] = "commit_rpc"
            details.append(tagged)
        return {
            "error": f"RECOVERY_REQUIRED: {operation} {reason}",
            "grpc_status": "FAILED_PRECONDITION",
            "recovery_required": True,
            "details": details,
        }

    async def _converge_commit(
        self,
        operation: str,
        rpc_name: str,
        commit_request: Any,
        timeout_s: float,
        transitional_state: str,
        final_state: str,
    ) -> Dict[str, Any]:
        probe_timeout_s = self._status_probe_timeout(timeout_s)
        # Single wall-clock bound over all commit attempts. Each attempt costs at
        # most one commit broadcast (timeout_s) plus one status probe
        # (probe_timeout_s); budgeting that across the allowed attempts gives one
        # explicit upper bound so a run of slow-but-not-timed-out attempts cannot
        # stack unbounded. time.monotonic() is immune to wall-clock adjustments.
        deadline = (
            time.monotonic() + (timeout_s + probe_timeout_s) * self.COMMIT_MAX_ATTEMPTS
        )
        pending = list(self.control_addresses)
        last_statuses: List[Dict[str, Any]] = []
        last_commit_failures: List[Dict[str, Any]] = []
        for _ in range(self.COMMIT_MAX_ATTEMPTS):
            if time.monotonic() >= deadline:
                return self._recovery_required(
                    operation,
                    "commit did not converge before its wall-clock deadline",
                    last_statuses,
                    extra_details=_error_details(last_commit_failures),
                )
            # Capture the commit broadcast's own per-rank errors and fold them into
            # any recovery response below, so the real commit grpc_status is carried
            # out rather than inferred solely from the later status probe.
            commit_results = await self._broadcast_control_rpc_to(
                pending, rpc_name, commit_request, timeout_s
            )
            last_commit_failures = [r for r in commit_results if "error" in r]
            last_statuses = await self._raw_sleep_statuses(probe_timeout_s)
            if len(last_statuses) != len(self.control_addresses):
                return self._recovery_required(
                    operation,
                    "status coverage is incomplete",
                    last_statuses,
                    extra_details=_error_details(last_commit_failures),
                )
            if any("error" in status for status in last_statuses):
                return self._recovery_required(
                    operation,
                    "status probe failed",
                    last_statuses,
                    extra_details=_error_details(last_commit_failures),
                )
            states = [str(status.get("state", "")) for status in last_statuses]
            allowed_states = {transitional_state, final_state}
            if any(state not in allowed_states for state in states):
                return self._recovery_required(
                    operation,
                    "observed an unrecoverable rank state",
                    last_statuses,
                    extra_details=_error_details(last_commit_failures),
                )
            pending = [
                status["address"]
                for status, state in zip(last_statuses, states)
                if state == transitional_state
            ]
            if not pending:
                return {"status": "ok"}
        return self._recovery_required(
            operation,
            f"did not converge after {self.COMMIT_MAX_ATTEMPTS} commit attempts",
            last_statuses,
            extra_details=_error_details(last_commit_failures),
        )

    def _aggregate_sleep_status(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successes = [result for result in results if "error" not in result]
        failures = [result for result in results if "error" in result]
        if not successes:
            return {
                "error": "Failed to get sleep status from all control ranks",
                "grpc_status": (
                    _representative_failure(failures).get("grpc_status", "UNAVAILABLE")
                    if failures
                    else "UNAVAILABLE"
                ),
                "details": _error_details(results),
            }
        if failures:
            return {
                "error": "Failed to get sleep status from some control ranks",
                "grpc_status": _representative_failure(failures).get(
                    "grpc_status", "UNAVAILABLE"
                ),
                "details": _error_details(results),
            }

        states = {str(result.get("state", "")) for result in successes}
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
        aggregate["sleep_epoch"] = max(
            _as_int(result.get("sleep_epoch", 0)) for result in successes
        )
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

    async def sleep_serving(self, req: Any) -> Dict[str, Any]:
        """Trigger engine sleep on every lifecycle control rank."""
        start_time = time.time() * 1000
        async with self._lifecycle_lock:
            lease_record, lease_error = self._acquire_lifecycle_lease("sleep")
            if lease_error:
                result = lease_error
            else:
                try:
                    result = await self._sleep_serving_locked(req)
                finally:
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
            try:
                req = _normalize_json_request(req)
            except ValueError as e:
                return {
                    "error": str(e),
                    "grpc_status": "INVALID_ARGUMENT",
                }
            unsupported_field = unsupported_lifecycle_control_field(req)
            if unsupported_field:
                return {
                    "error": f"sleep {unsupported_field} is unsupported",
                    "grpc_status": "INVALID_ARGUMENT",
                }
            try:
                level = int(req.get("level", 1))
                timeout_ms = int(req.get("timeout_ms", 0))
            except (TypeError, ValueError):
                return {
                    "error": "sleep level and timeout_ms must be integers",
                    "grpc_status": "INVALID_ARGUMENT",
                }
            if level == 0:
                # level 0 (state-preserving sleep) is defined in the API but not
                # implemented. Advertise the single non-zero level this process was
                # started with (SLEEP_MODE_LEVEL, mirrored into the environment for
                # both frontend and backend at startup) so a level-2 deployment
                # reports [2] rather than a hardcoded [1] that would misdirect the
                # client toward an unsupported level. Derived locally so this stays
                # a cheap frontend rejection with no backend status probe. Drain
                # modes (wait/abort) are a fixed, level-independent capability.
                return {
                    "error": "sleep level=0 state-preserving sleep is defined but not implemented",
                    "grpc_status": "UNIMPLEMENTED",
                    "supported_levels": [_configured_sleep_level()],
                    "supported_modes": ["wait", "abort"],
                }
            # level 1 (host backup) and level 2 (discard weights) are both valid
            # requests; the backend is the authority on which one this process
            # supports (fixed at startup by sleep_mode_level) and returns
            # INVALID_ARGUMENT on a mismatch.
            if level not in (1, 2):
                return {
                    "error": "sleep level must be 0, 1 or 2",
                    "grpc_status": "INVALID_ARGUMENT",
                }
            mode = str(req.get("mode", "wait"))
            if mode not in ("wait", "abort"):
                return {
                    "error": 'sleep mode must be "wait" or "abort"',
                    "grpc_status": "INVALID_ARGUMENT",
                }
            tags = req.get("tags", [])
            if tags is None:
                tags = []
            if not isinstance(tags, list):
                return {
                    "error": "sleep tags must be a list",
                    "grpc_status": "INVALID_ARGUMENT",
                }
            if any(not isinstance(tag, str) or not tag for tag in tags):
                return {
                    "error": "sleep tags must be non-empty strings",
                    "grpc_status": "INVALID_ARGUMENT",
                }
            status = await self._initial_lifecycle_status("sleep")
            if "error" in status:
                return status
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
            prepare_results = await self._broadcast_control_rpc(
                "SleepServing", prepare_request, timeout_s
            )
            failures = [result for result in prepare_results if "error" in result]
            if failures:
                abort_results = await self._broadcast_control_rpc(
                    "WakeUpServing", pb2.WakeUpRequestPB(), timeout_s=60
                )
                abort_failures = [r for r in abort_results if "error" in r]
                if abort_failures:
                    # Prepare failed AND the rollback (wake_up abort) also failed on
                    # some ranks. The instance is now in an unreconciled partial state
                    # (some ranks may still be DRAINING with admission closed). Surface
                    # BOTH sets of details and escalate — do NOT let the non-empty
                    # prepare details short-circuit the abort failure out of the
                    # response, or the control plane will believe only prepare failed
                    # and never learn the rollback did not take.
                    return {
                        "error": "Failed to prepare sleep and failed to roll back on "
                        "some control ranks; instance may be in an inconsistent state, "
                        "issue wake_up or restart the process to recover",
                        "grpc_status": "FAILED_PRECONDITION",
                        "details": _error_details(prepare_results)
                        + _error_details(abort_results),
                    }
                return {
                    "error": "Failed to prepare sleep on some control ranks (rolled back)",
                    "grpc_status": _representative_failure(failures).get(
                        "grpc_status", "UNKNOWN"
                    ),
                    "details": _error_details(prepare_results),
                }

            # commit runs the GPU-release hooks. Reuse the drain-derived headroom
            # so a slow device synchronization or allocator reclaim does not
            # spuriously trip the commit deadline.
            return await self._converge_commit(
                operation="commit sleep",
                rpc_name="SleepServing",
                commit_request=commit_request,
                timeout_s=timeout_s,
                transitional_state="DRAINING",
                final_state="SLEEPING",
            )
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
        async with self._lifecycle_lock:
            lease_record, lease_error = self._acquire_lifecycle_lease("wake_up")
            if lease_error:
                result = lease_error
            else:
                try:
                    result = await self._wake_up_serving_locked(req)
                finally:
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
            try:
                req = _normalize_json_request(req)
            except ValueError as e:
                return {
                    "error": str(e),
                    "grpc_status": "INVALID_ARGUMENT",
                }
            unsupported_field = unsupported_lifecycle_control_field(req)
            if unsupported_field:
                return {
                    "error": f"wake_up {unsupported_field} is unsupported",
                    "grpc_status": "INVALID_ARGUMENT",
                }
            status = await self._initial_lifecycle_status("wake_up")
            if "error" in status:
                return status
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

            prepare_results = await self._broadcast_control_rpc(
                "WakeUpServing", prepare_request, timeout_s=600
            )
            failures = [result for result in prepare_results if "error" in result]
            if failures:
                return {
                    "error": "Failed to prepare wake_up on some control ranks",
                    "grpc_status": _representative_failure(failures).get(
                        "grpc_status", "UNKNOWN"
                    ),
                    "details": _error_details(prepare_results),
                }

            # commit runs the GPU-restore hooks; for level-2 that includes reloading
            # the original checkpoint through the model loader (read + H2D copy),
            # which can take far longer than a level-1 tms resume. Give it the same
            # headroom as wake prepare so a slow restore does not trip the deadline.
            return await self._converge_commit(
                operation="commit wake_up",
                rpc_name="WakeUpServing",
                commit_request=commit_request,
                timeout_s=600,
                transitional_state="WAKING_UP",
                final_state="RUNNING",
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

    async def get_sleep_status(self, req: Any = None) -> Dict[str, Any]:
        """Get aggregate sleep lifecycle status from every control rank."""
        try:
            self._refresh_control_addresses_if_needed()
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
            "is_sleeping": status.get("state") == "SLEEPING",
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
