"""Frontend lifecycle orchestration; execution coordination belongs to the engine.

Owns the instance lease, phase barriers, cancellation/rollback and convergence.
The injected RPC transport owns channels; this controller never owns a wrapper
or a model execution round.
"""

import asyncio
import logging
import uuid
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict, List, Optional

import grpc

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
from rtp_llm.aios.kmonitor.python_client.kmonitor.reporting import (
    set_instance_reporting,
)
from rtp_llm.frontend.sleep_validation import (
    dedupe_addresses,
    normalize_lifecycle_request,
    unsupported_lifecycle_control_field,
    validate_sleep_request,
)
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.utils.lifecycle_lease import LifecycleLease
from rtp_llm.utils.lifecycle_quiesce import prepare_sleep_quiesce
from rtp_llm.utils.lifecycle_rpc import ControlRpcTransport
from rtp_llm.utils.lifecycle_status import (
    _as_int,
    aggregate,
    error_details,
    recovery_required,
)
from rtp_llm.utils.sleep_timing import log_sleep_timing


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


class LifecycleController:
    """Serialize and converge sleep/wake across all resource-owning ranks."""

    COMMIT_MAX_ATTEMPTS = 3
    COMMIT_POLL_INTERVAL_S = 0.1

    def __init__(
        self,
        rpc: ControlRpcTransport,
        control_addresses: List[str],
        *,
        expected_control_address_count: Optional[int] = None,
        control_address_resolver: Optional[Callable[[], List[str]]] = None,
        lifecycle_store: Optional[Any] = None,
        lifecycle_store_factory: Optional[Callable[[], Optional[Any]]] = None,
        require_instance_lease: bool = False,
    ):
        self._rpc = rpc
        self.control_addresses = dedupe_addresses(control_addresses)
        self.expected_control_address_count = expected_control_address_count
        self._control_address_resolver = control_address_resolver
        self._lifecycle_lease = LifecycleLease(
            lifecycle_store, lifecycle_store_factory, require_instance_lease
        )
        # Local re-entrancy is separate from authoritative instance-wide CAS.
        self._lifecycle_lock = asyncio.Lock()

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

    async def _refresh_control_addresses_if_needed(self) -> None:
        if self._control_address_resolver is None:
            return
        expected = int(self.expected_control_address_count or 0)
        if expected > 0 and len(self.control_addresses) >= expected:
            return
        try:
            resolved_addresses = dedupe_addresses(
                await asyncio.to_thread(self._control_address_resolver) or []
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

    async def _call_control_rpc(
        self, address: str, rpc_name: str, request: Any, timeout_s: float
    ) -> Dict[str, Any]:
        return await self._rpc.call(address, rpc_name, request, timeout_s)

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
        return await self._rpc.broadcast(addresses, rpc_name, request, timeout_s)

    async def _acquire_lifecycle_lease(
        self, operation: str
    ) -> tuple[Optional[str], Dict[str, Any]]:
        # TCPStore connection/CAS can block. Cancellation cannot stop its worker:
        # wait for a late acquisition and release it before propagating cancel.
        task = asyncio.create_task(
            asyncio.to_thread(self._lifecycle_lease.acquire, operation)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            record, _ = await self._drive_to_terminal(task)
            await self._release_lifecycle_lease(record)
            raise

    async def _release_lifecycle_lease(self, record: Optional[str]) -> None:
        await self._drive_to_terminal(
            asyncio.to_thread(self._lifecycle_lease.release, record)
        )

    async def _raw_sleep_statuses(self) -> List[Dict[str, Any]]:
        return await self._broadcast_control_rpc(
            "GetSleepStatus", pb2.EmptyPB(), timeout_s=3
        )

    async def _initial_lifecycle_status(
        self, operation: str, *, rank_snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Probe the pre-condition state shared by every control rank.

        Each operation owns the instance-wide lease through its terminal state.
        Drain may roll back under that lease; after resource release starts,
        sleep/wake must converge despite request cancellation. A mixed rank
        state left by a previous operation -- e.g. some ranks SLEEPING
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
        await self._refresh_control_addresses_if_needed()
        statuses = await self._raw_sleep_statuses()
        rank_snapshots.extend(statuses)
        status = self._aggregate_sleep_status(statuses)
        _report_sleep_status_metrics(status)
        if "error" in status:
            return recovery_required(
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
            return recovery_required(
                operation, "observed an invalid initial rank state", statuses
            )
        return status

    async def _drive_to_terminal(self, coro: Any) -> Any:
        """Finish lifecycle commit, drain rollback or lease I/O despite cancellation.

        Once an irreversible phase has started running the GPU-release /
        GPU-restore hooks there is no consistent rollback: freed device memory
        (or a level-2 reload) cannot be un-done, so the only valid terminal states
        are the two endpoints of the transition -- every rank RUNNING or every
        rank SLEEPING. If the frontend request task is cancelled midway (a
        concurrent /health probe tearing down a shared channel, a worker
        recycle, a client disconnect) we must NOT abandon the transition
        half-committed and release the lifecycle lease -- that is the
        control-plane split brain that leaves the instance with half its device
        memory freed and no owner driving it to a consistent state.

        A reversible prepare may be cancelled, but its rollback must likewise
        finish before releasing the instance lease. Cancelling that compensation
        would strand drained ranks with admission closed and no coordinator.

        So we absorb the cancellation and keep awaiting until the backend
        converges, then report the true terminal state to whoever is left. Both
        prepare RPCs and `_converge_commit` have bounded deadlines, so this cannot
        hang indefinitely.
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
                # Still driving an irreversible transition -- swallow the
                # cancellation and keep waiting for the backend to converge.
                absorbed_cancel = True
                continue
        if absorbed_cancel:
            logging.warning(
                "lifecycle transition was driven to its terminal state despite "
                "request cancellation; absorbed the cancel to avoid a "
                "half-committed instance"
            )
        return result

    async def _rollback_sleep_prepare(
        self, quiesce_token: str = ""
    ) -> List[Dict[str, Any]]:
        """Abort drain and verify every rank is usable before returning ownership.

        Called only before commit, inside ``_drive_to_terminal``. RPC success
        alone is not proof of recovery; keep the lease through the status probe.
        In-flight requests may still be running, so their counts need not be zero.
        """
        results = await self._broadcast_control_rpc(
            "WakeUpServing",
            pb2.WakeUpRequestPB(cancel_quiesce_token=quiesce_token),
            timeout_s=75,
        )
        statuses = await self._raw_sleep_statuses()
        if {status.get("address") for status in statuses} != set(
            self.control_addresses
        ) or len(statuses) != len(self.control_addresses):
            results.append({"error": "drain rollback status coverage is incomplete"})
        for status in statuses:
            if "error" in status:
                results.append(status)
            elif (
                status.get("state") != "RUNNING"
                or status.get("gpu_resource_state") != "ACTIVE"
                or status.get("kv_memory_state") != "ACTIVE"
                or not status.get("device_kv_cache_valid", False)
            ):
                results.append(
                    {
                        "address": status.get("address", ""),
                        "error": "drain rollback did not restore RUNNING with valid "
                        f"GPU/KV resources: {status}",
                        "grpc_status": "FAILED_PRECONDITION",
                    }
                )
        return results

    async def _converge_commit(
        self,
        operation: str,
        rpc_name: str,
        commit_request: Any,
        timeout_s: float,
        transitional_state: str,
        final_state: str,
        rank_snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pending = list(self.control_addresses)
        last_statuses: List[Dict[str, Any]] = []
        # Preserve the former maximum transport budget, but do not confuse
        # three fast/lost replies with an execution hook that has failed.
        deadline = perf_counter() + timeout_s * self.COMMIT_MAX_ATTEMPTS
        attempt = 0
        expected = {s["address"]: s for s in rank_snapshots}
        while perf_counter() < deadline:
            attempt += 1
            attempt_started = perf_counter()
            rpc_timeout = min(timeout_s, max(0.001, deadline - perf_counter()))
            if rpc_name == "WakeUpServing":
                await asyncio.gather(
                    *(
                        self._call_control_rpc(
                            address,
                            rpc_name,
                            self._fenced_wake_request(
                                commit_request, expected[address]
                            ),
                            rpc_timeout,
                        )
                        for address in pending
                    )
                )
            else:
                await self._broadcast_control_rpc_to(
                    pending, rpc_name, commit_request, rpc_timeout
                )
            last_statuses = await self._raw_sleep_statuses()
            log_sleep_timing(
                "sleep" if operation.endswith("sleep") else "wake",
                "commit_attempt",
                (perf_counter() - attempt_started) * 1000.0,
                scope="controller",
                fields={
                    "attempt": attempt,
                    "pending_count": len(pending),
                    "status_count": len(last_statuses),
                },
            )
            if len(last_statuses) != len(self.control_addresses) or {
                s.get("address") for s in last_statuses
            } != set(self.control_addresses):
                return recovery_required(
                    operation, "status coverage is incomplete", last_statuses
                )
            if any("error" in status for status in last_statuses):
                return recovery_required(
                    operation, "status probe failed", last_statuses
                )
            if not self._matching_rank_identities(last_statuses, expected):
                return recovery_required(
                    operation,
                    "worker incarnation or sleep epoch changed",
                    last_statuses,
                )
            states = [str(status.get("state", "")) for status in last_statuses]
            allowed_states = {transitional_state, final_state}
            if rpc_name == "SleepServing":
                allowed_states.add("SUSPENDING")
            if any(state not in allowed_states for state in states):
                return recovery_required(
                    operation, "observed an unrecoverable rank state", last_statuses
                )
            pending = [
                status["address"]
                for status, state in zip(last_statuses, states)
                if state != final_state
            ]
            if not pending:
                if final_state == "RUNNING":
                    return await self._resume_metrics_after_wake(last_statuses)
                await asyncio.to_thread(set_instance_reporting, False)
                return {"status": "ok"}
            remaining = deadline - perf_counter()
            if remaining > 0:
                await asyncio.sleep(min(self.COMMIT_POLL_INTERVAL_S, remaining))
        return recovery_required(
            operation,
            "total commit deadline exceeded while ranks had not completed; "
            "this does not prove an in-progress resource hook has stopped",
            last_statuses,
        )

    @staticmethod
    def _valid_rank_identity(status):
        incarnation = status.get("worker_incarnation")
        epoch = status.get("sleep_epoch")
        return (
            isinstance(incarnation, str)
            and bool(incarnation)
            and type(epoch) in (int, str)
            and 0 <= _as_int(epoch, -1) < (1 << 63)
        )

    @staticmethod
    def _wake_prepare_capability_error(
        operation: str, statuses: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        # A false wake_prepared is a normal pre-wake state, not a capability.
        # Check static support before either sleep drain or irreversible restore;
        # otherwise a lost prepare reply from an old rank cannot be reconciled.
        unsupported = [
            {
                "address": status.get("address", ""),
                "wake_prepare_protocol": status.get("wake_prepare_protocol", 0),
            }
            for status in statuses
            if type(status.get("wake_prepare_protocol")) not in (int, str)
            or _as_int(status.get("wake_prepare_protocol"), -1) != 1
        ]
        if not statuses or unsupported:
            return {
                "error": f"{operation} requires wake prepare protocol 1 on every "
                "backend rank; upgrade frontend and backend together before "
                "using coordinated sleep/wake",
                "grpc_status": "UNIMPLEMENTED",
                "details": unsupported,
            }
        return None

    @staticmethod
    def _matching_rank_identities(statuses, expected):
        return all(
            LifecycleController._valid_rank_identity(status)
            and status.get("worker_incarnation")
            == expected[status["address"]].get("worker_incarnation")
            and _as_int(status.get("sleep_epoch"))
            == _as_int(expected[status["address"]].get("sleep_epoch"))
            for status in statuses
        )

    @staticmethod
    def _fenced_wake_request(request, status):
        return pb2.WakeUpRequestPB(
            prepare_only=request.prepare_only,
            commit_only=request.commit_only,
            expected_incarnation=status.get("worker_incarnation", ""),
            expected_sleep_epoch=_as_int(status.get("sleep_epoch")),
        )

    async def _resume_metrics_after_wake(
        self, statuses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Called under the existing lifecycle lease after all ranks reached
        # RUNNING. Per-rank epoch/incarnation fences late notification retries.
        pending = {status["address"]: status for status in statuses}
        failures = []
        for _ in range(self.COMMIT_MAX_ATTEMPTS):
            results = await asyncio.gather(
                *(
                    self._call_control_rpc(
                        address,
                        "WakeUpServing",
                        pb2.WakeUpRequestPB(
                            resume_metrics_only=True,
                            expected_incarnation=status.get("worker_incarnation", ""),
                            expected_sleep_epoch=_as_int(status.get("sleep_epoch", 0)),
                        ),
                        timeout_s=10,
                    )
                    for address, status in pending.items()
                )
            )
            failures = [result for result in results if "error" in result]
            if not failures:
                await asyncio.to_thread(set_instance_reporting, True)
                return {"status": "ok"}
            pending = {
                result["address"]: pending[result["address"]] for result in failures
            }
        return {
            "error": "all ranks are RUNNING; metrics resume failed, retry wake_up",
            "grpc_status": "UNAVAILABLE",
            "details": error_details(failures),
        }

    def _aggregate_sleep_status(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate(results, self._control_address_coverage_error())

    async def _run_operation(
        self,
        operation: str,
        req: Any,
        action: Callable[[Any], Awaitable[Dict[str, Any]]],
        metric: Any,
    ) -> Dict[str, Any]:
        """Keep ownership through compensation/commit and report one outcome."""
        started = perf_counter()
        async with self._lifecycle_lock:
            lease_record, lease_error = await self._acquire_lifecycle_lease(operation)
            if lease_error:
                result = lease_error
            else:
                try:
                    result = await action(req)
                finally:
                    await self._release_lifecycle_lease(lease_record)
        duration_ms = (perf_counter() - started) * 1000.0
        _report_metric_if_ready(metric, duration_ms)
        log_operation = "wake" if operation == "wake_up" else operation
        if "error" in result:
            log_sleep_timing(
                log_operation,
                "end",
                duration_ms,
                status="error",
                scope="controller",
                fields={"grpc_status": result.get("grpc_status", "UNKNOWN")},
            )
        else:
            log_sleep_timing(log_operation, "end", duration_ms, scope="controller")
        return result

    async def sleep_serving(self, req: Any) -> Dict[str, Any]:
        return await self._run_operation(
            "sleep",
            req,
            self._sleep_serving_locked,
            GaugeMetrics.SLEEP_ACTION_RT_METRIC,
        )

    async def _sleep_serving_locked(self, req: Any) -> Dict[str, Any]:
        try:
            try:
                req = normalize_lifecycle_request(req)
            except ValueError as e:
                return {
                    "error": str(e),
                    "grpc_status": "INVALID_ARGUMENT",
                }
            try:
                options = validate_sleep_request(req)
            except ValueError as error:
                return {"error": str(error), "grpc_status": "INVALID_ARGUMENT"}
            level, mode, timeout_ms = options.level, options.mode, options.timeout_ms
            rank_snapshots: List[Dict[str, Any]] = []
            status = await self._initial_lifecycle_status(
                "sleep", rank_snapshots=rank_snapshots
            )
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
            if level == 0:
                return {
                    "error": "sleep level=0 state-preserving sleep is defined but not implemented",
                    "grpc_status": "UNIMPLEMENTED",
                    "supported_levels": status.get("supported_levels", []),
                    "supported_modes": status.get("supported_modes", []),
                }
            if level not in status.get("supported_levels", []):
                return {
                    "error": "sleep level does not match the startup sleep_mode_level",
                    "grpc_status": "INVALID_ARGUMENT",
                    "supported_levels": status.get("supported_levels", []),
                }
            capability_error = self._wake_prepare_capability_error(
                "sleep", rank_snapshots
            )
            if capability_error:
                return capability_error
            if status.get("state") == "SLEEPING":
                # A standalone frontend may have restarted after the backends
                # slept. Keep the lease until its local sender fence finishes,
                # even if the caller cancels this idempotent request.
                await self._drive_to_terminal(
                    asyncio.to_thread(set_instance_reporting, False)
                )
                return {"status": "ok"}
            if status.get("state") != "RUNNING":
                return {
                    "error": "sleep requires all ranks RUNNING; cancel an unfinished drain with wake_up first",
                    "grpc_status": "FAILED_PRECONDITION",
                }
            request = pb2.SleepRequestPB(
                level=level,
                mode=mode,
                timeout_ms=timeout_ms,
                reason=str(req.get("reason", "")),
                tags=[],
                quiesce_token=uuid.uuid4().hex,
            )
            commit_request = pb2.SleepRequestPB()
            commit_request.CopyFrom(request)
            commit_request.commit_only = True
            commit_request.timeout_ms = 0

            # Every rank drains while empty peers still execute fake forwards.
            # Freeze only after all drain ACKs; backends then coordinate their
            # execution boundary without exposing round IDs to this frontend.
            prepare_rpc_timeout_s = max(60.0, timeout_ms / 1000.0 + 30.0)
            try:
                prepare_results = await prepare_sleep_quiesce(
                    request,
                    self.control_addresses,
                    rank_snapshots,
                    self._call_control_rpc,
                    self._broadcast_control_rpc,
                    prepare_rpc_timeout_s,
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
                abort_results = await self._drive_to_terminal(
                    self._rollback_sleep_prepare(request.quiesce_token)
                )
                if any("error" in result for result in abort_results):
                    return {
                        "error": "RECOVERY_REQUIRED: cancelled sleep prepare failed "
                        "to roll back; restart the instance",
                        "grpc_status": "FAILED_PRECONDITION",
                        "recovery_required": True,
                        "details": error_details(abort_results),
                    }
                raise
            except Exception as error:
                # A malformed freeze response or local transport exception is
                # still a reversible prepare failure; do not strand frozen ranks.
                logging.exception("sleep quiesce prepare failed")
                prepare_results = [{"error": str(error), "grpc_status": "UNKNOWN"}]
            failures = [result for result in prepare_results if "error" in result]
            if failures:
                abort_results = await self._drive_to_terminal(
                    self._rollback_sleep_prepare(request.quiesce_token)
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
                        "error": "RECOVERY_REQUIRED: failed to prepare sleep and "
                        "failed to roll back on "
                        "some control ranks; instance may be in an inconsistent state, "
                        "restart the instance to recover",
                        "grpc_status": "FAILED_PRECONDITION",
                        "recovery_required": True,
                        "details": error_details(prepare_results)
                        + error_details(abort_results),
                    }
                return {
                    "error": "Failed to prepare sleep on some control ranks (rolled back)",
                    "grpc_status": failures[0].get("grpc_status", "UNKNOWN"),
                    "details": error_details(prepare_results),
                }

            # Commit releases GPU resources. Level 2 discards weight pages
            # without writing a backup; wake reloads the original checkpoint.
            # The request timeout bounds drain, not cancellation of a commit:
            # once release starts, drive all ranks to SLEEPING (or report a
            # recovery-required failure), even if this request is cancelled.
            # Host backup/resource release needs at least wake's transport
            # budget. Preserve any longer deadline previously allowed by drain.
            return await self._drive_to_terminal(
                self._converge_commit(
                    operation="commit sleep",
                    rpc_name="SleepServing",
                    commit_request=commit_request,
                    timeout_s=max(600.0, prepare_rpc_timeout_s),
                    transitional_state="DRAINING",
                    final_state="SLEEPING",
                    rank_snapshots=[
                        {
                            **snapshot,
                            "sleep_epoch": _as_int(snapshot.get("sleep_epoch")) + 1,
                        }
                        for snapshot in rank_snapshots
                    ],
                )
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
        return await self._run_operation(
            "wake_up",
            req,
            self._wake_up_serving_locked,
            GaugeMetrics.WAKE_UP_ACTION_RT_METRIC,
        )

    async def _wake_up_to_terminal(
        self,
        prepare_request: Any,
        commit_request: Any,
        rank_snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run irreversible wake preparation and commit as one protected unit."""
        deadline = perf_counter() + 600
        expected = {s["address"]: s for s in rank_snapshots}
        prepare_results = await asyncio.gather(
            *(
                self._call_control_rpc(
                    address,
                    "WakeUpServing",
                    self._fenced_wake_request(prepare_request, expected[address]),
                    600,
                )
                for address in self.control_addresses
            )
        )
        failures = [result for result in prepare_results if "error" in result]
        while failures:
            statuses = await self._raw_sleep_statuses()
            covered = (
                len(statuses) == len(self.control_addresses)
                and {s.get("address") for s in statuses} == set(self.control_addresses)
                and all("error" not in status for status in statuses)
            )
            matched = covered and self._matching_rank_identities(statuses, expected)
            if matched and all(
                s.get("state") == "RUNNING"
                or (s.get("state") == "WAKING_UP" and s.get("wake_prepared") is True)
                for s in statuses
            ):
                logging.warning(
                    "wake_up prepare RPC reported failure, but every control rank "
                    "confirmed preparation for the same incarnation/epoch; continuing commit"
                )
                break
            if matched and all(
                s.get("state") in ("WAKING_UP", "RUNNING") for s in statuses
            ):
                remaining = deadline - perf_counter()
                if remaining > 0:
                    await asyncio.sleep(min(self.COMMIT_POLL_INTERVAL_S, remaining))
                    continue
            recovery = recovery_required(
                "prepare wake_up",
                "could not confirm completed preparation for every original rank",
                statuses,
            )
            recovery["prepare_details"] = error_details(prepare_results)
            return recovery

        return await self._converge_commit(
            operation="commit wake_up",
            rpc_name="WakeUpServing",
            commit_request=commit_request,
            timeout_s=600,
            transitional_state="WAKING_UP",
            final_state="RUNNING",
            rank_snapshots=rank_snapshots,
        )

    async def _wake_up_serving_locked(self, req: Any = None) -> Dict[str, Any]:
        try:
            try:
                req = normalize_lifecycle_request(req)
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
            rank_snapshots: List[Dict[str, Any]] = []
            status = await self._initial_lifecycle_status(
                "wake_up", rank_snapshots=rank_snapshots
            )
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
            # An empty incarnation selects the backend's legacy unfenced path.
            # Never silently downgrade coordinated wake when a peer omits it.
            if (
                len(rank_snapshots) != len(self.control_addresses)
                or {s.get("address") for s in rank_snapshots}
                != set(self.control_addresses)
                or not all(self._valid_rank_identity(s) for s in rank_snapshots)
            ):
                return recovery_required(
                    "wake_up",
                    "missing or invalid initial rank identity",
                    rank_snapshots,
                )
            capability_error = self._wake_prepare_capability_error(
                "wake_up", rank_snapshots
            )
            if capability_error:
                return capability_error
            prepare_request = pb2.WakeUpRequestPB(prepare_only=True)
            commit_request = pb2.WakeUpRequestPB(commit_only=True)

            # Wake prepare already restores VMM backing and, for level 2, reloads
            # weights from the checkpoint. Protect prepare and commit together so
            # cancellation cannot release the lifecycle lease in WAKING_UP.
            return await self._drive_to_terminal(
                self._wake_up_to_terminal(
                    prepare_request, commit_request, rank_snapshots
                )
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
            await self._refresh_control_addresses_if_needed()
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
