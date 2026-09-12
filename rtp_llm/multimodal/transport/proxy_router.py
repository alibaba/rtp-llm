import logging
import numbers
import threading
import time
from collections import Counter, defaultdict
from itertools import islice
from typing import Optional

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalOutputPB,
    ReleaseLeasePB,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics

DEFAULT_HANDLE_ROUTE_TTL_SECONDS = 120.0
HANDLE_ROUTE_CLEANUP_INTERVAL_SECONDS = 10.0
DEFAULT_RELEASE_TIMEOUT_SECONDS = 1.0
HANDLE_ROUTE_GC_SAFETY_SECONDS = 5.0
# Keep routing bounded by both the KVCM receipt contract and the shared gRPC
# control client's pending-release capacity.
_MAX_KVCM_OBJECTS_PER_RECEIPT = 1024
_MAX_KVCM_KEY_BYTES = 512
_MAX_RELEASE_HANDLES = 1024


def _safe_warning(message: str, *args) -> None:
    """Keep route ownership transitions independent of logging handlers."""
    try:
        logging.warning(message, *args)
    except Exception:  # noqa: BLE001 - observability must not break cleanup
        pass


def _context_deadline_seconds(context, max_timeout_seconds: float) -> Optional[float]:
    remaining = None
    if context is not None and hasattr(context, "time_remaining"):
        try:
            remaining = context.time_remaining()
        except Exception as error:  # noqa: BLE001 - a missing deadline is valid
            _safe_warning(
                "Failed to read gRPC context time remaining (exception_type=%s)",
                type(error).__name__,
            )
    if not isinstance(remaining, numbers.Real):
        remaining = None
    if remaining is not None and remaining <= 0:
        return None
    timeout = (
        max_timeout_seconds
        if remaining is None
        else min(max_timeout_seconds, remaining)
    )
    return time.monotonic() + timeout


class MMOutputProxyRouter:
    """Routes transport-owned leases through a multi-worker ViT proxy."""

    def __init__(self, connection_pool, transport_config=None):
        self._connection_pool = connection_pool
        self._handle_routes: dict[str, tuple[str, float, int]] = {}
        self._handle_collisions: dict[str, float] = {}
        self._handle_generations: dict[str, int] = {}
        self._next_generation = 0
        self._lock = threading.Lock()
        self._last_cleanup = 0.0

        if transport_config is None:
            self._route_ttl_seconds = DEFAULT_HANDLE_ROUTE_TTL_SECONDS
            self._release_timeout_seconds = DEFAULT_RELEASE_TIMEOUT_SECONDS
        else:
            if transport_config.mode == "kvcm":
                gc_timeout_ms = transport_config.kvcm.object_gc_timeout_ms
            else:
                gc_timeout_ms = transport_config.rdma.slot_gc_timeout_ms
            slot_gc_seconds = max(0.0, gc_timeout_ms / 1000.0)
            self._route_ttl_seconds = max(
                1.0, slot_gc_seconds + HANDLE_ROUTE_GC_SAFETY_SECONDS
            )
            self._release_timeout_seconds = max(
                0.001, transport_config.control.release_timeout_ms / 1000.0
            )

    def record_receipt(self, worker_address: str, receipt: MultimodalOutputPB) -> None:
        kvcm_handles = []
        invalid_kvcm_keys = 0
        for obj in islice(receipt.output_kvcm_objects, _MAX_KVCM_OBJECTS_PER_RECEIPT):
            key = obj.key
            try:
                valid_key = (
                    bool(key) and len(key.encode("utf-8")) <= _MAX_KVCM_KEY_BYTES
                )
            except UnicodeEncodeError:
                valid_key = False
            if valid_key:
                kvcm_handles.append(key)
            else:
                invalid_kvcm_keys += 1

        overflow_count = max(
            0,
            len(receipt.output_kvcm_objects) - _MAX_KVCM_OBJECTS_PER_RECEIPT,
        )
        if invalid_kvcm_keys:
            _safe_warning(
                "Ignoring %d invalid KVCM receipt key(s); worker GC will reclaim them",
                invalid_kvcm_keys,
            )
            self._report_error("kvcm_route_invalid_key", invalid_kvcm_keys)
        if overflow_count:
            _safe_warning(
                "Ignoring %d KVCM receipt object(s) beyond the route limit; "
                "worker GC will reclaim them",
                overflow_count,
            )
            self._report_error("kvcm_route_object_limit", overflow_count)

        handles = list(
            dict.fromkeys(
                [
                    slot.rdma_descriptor.lease_id
                    for slot in receipt.output_rdma_slots
                    if slot.rdma_descriptor.lease_id
                ]
                + kvcm_handles
            )
        )
        if not handles:
            return

        now = time.monotonic()
        collision_count = 0
        with self._lock:
            if now - self._last_cleanup >= HANDLE_ROUTE_CLEANUP_INTERVAL_SECONDS:
                self._sweep_locked(now)
            for handle in handles:
                self._next_generation += 1
                generation = self._next_generation
                self._handle_generations[handle] = generation
                if handle in self._handle_collisions:
                    continue
                existing = self._handle_routes.get(handle)
                # A receipt can be observed more than once while the request is
                # retried or forwarded. The same worker is still the only safe
                # release destination, so refresh that route below. Different
                # workers claiming one opaque handle is genuinely ambiguous and
                # must fail closed to avoid freeing the wrong producer's object.
                if existing is not None and existing[0] != worker_address:
                    self._handle_routes.pop(handle, None)
                    self._handle_collisions[handle] = now
                    collision_count += 1
                    continue
                self._handle_routes[handle] = (worker_address, now, generation)

        if collision_count:
            _safe_warning(
                "%d multimodal transport handle collision(s) across workers; "
                "ambiguous routes were poisoned",
                collision_count,
            )
            self._report_error("rdma_handle_collision", collision_count)

    def release(self, request: ReleaseLeasePB, context) -> None:
        handles_by_worker: dict[str, list[str]] = defaultdict(list)
        selected_routes: dict[str, tuple[str, float, int]] = {}
        skipped_routes = Counter()
        release_handles = list(islice(request.lease_id, _MAX_RELEASE_HANDLES))
        overflow_count = max(0, len(request.lease_id) - _MAX_RELEASE_HANDLES)
        if overflow_count:
            _safe_warning(
                "Ignoring %d multimodal release handle(s) beyond the control limit; "
                "worker GC will reclaim them",
                overflow_count,
            )
            self._report_error("release_handle_limit", overflow_count)
        now = time.monotonic()
        with self._lock:
            for handle in dict.fromkeys(release_handles):
                if self._handle_collisions.pop(handle, None) is not None:
                    self._handle_generations.pop(handle, None)
                    skipped_routes["release_handle_collision"] += 1
                    continue
                route = self._handle_routes.pop(handle, None)
                if route is None:
                    skipped_routes["release_handle_unknown"] += 1
                    continue
                if now - route[1] > self._route_ttl_seconds:
                    if self._handle_generations.get(handle) == route[2]:
                        self._handle_generations.pop(handle, None)
                    skipped_routes["release_handle_expired"] += 1
                    continue
                handles_by_worker[route[0]].append(handle)
                selected_routes[handle] = route
            if now - self._last_cleanup >= HANDLE_ROUTE_CLEANUP_INTERVAL_SECONDS:
                self._sweep_locked(now)

        for reason, count in skipped_routes.items():
            _safe_warning(
                "Cannot route %d multimodal transport handle(s) (%s); "
                "worker GC will reclaim them",
                count,
                reason,
            )
            self._report_error(reason, count)

        deadline = _context_deadline_seconds(context, self._release_timeout_seconds)
        if deadline is None and handles_by_worker:
            skipped_count = sum(len(handles) for handles in handles_by_worker.values())
            _safe_warning(
                "Multimodal release deadline exhausted; skipping %d handles",
                skipped_count,
            )
            self._report_error("release_deadline_exhausted", skipped_count)
            self._restore_claims(selected_routes)
            return

        release_groups = list(handles_by_worker.items())
        for group_index, (worker_address, handles) in enumerate(release_groups):
            try:
                timeout = (
                    deadline - time.monotonic()
                    if deadline is not None
                    else self._release_timeout_seconds
                )
                if timeout <= 0:
                    skipped_count = sum(
                        len(group_handles)
                        for _, group_handles in release_groups[group_index:]
                    )
                    _safe_warning(
                        "Multimodal release deadline exhausted; "
                        "skipping remaining %d handles",
                        skipped_count,
                    )
                    self._report_error("release_deadline_exhausted", skipped_count)
                    remaining_handles = [
                        handle
                        for _, group_handles in release_groups[group_index:]
                        for handle in group_handles
                    ]
                    self._restore_claims(
                        {
                            handle: selected_routes[handle]
                            for handle in remaining_handles
                        }
                    )
                    break
                stub = self._connection_pool.get_stub(worker_address)
                stub.ReleaseRdmaLease(ReleaseLeasePB(lease_id=handles), timeout=timeout)
                self._complete_claims(handles, selected_routes)
            except Exception as error:  # noqa: BLE001 - worker GC is the backstop
                _safe_warning(
                    "Failed to release %d multimodal handle(s) on a VIT worker; "
                    "worker GC will reclaim them (exception_type=%s)",
                    len(handles),
                    type(error).__name__,
                )
                self._restore_claims(
                    {handle: selected_routes[handle] for handle in handles}
                )

    def _complete_claims(self, handles, selected_routes) -> None:
        with self._lock:
            for handle in handles:
                route = selected_routes[handle]
                if self._handle_generations.get(handle) == route[2]:
                    self._handle_generations.pop(handle, None)

    def _restore_claims(self, selected_routes) -> None:
        with self._lock:
            for handle, route in selected_routes.items():
                if (
                    self._handle_generations.get(handle) == route[2]
                    and handle not in self._handle_routes
                    and handle not in self._handle_collisions
                ):
                    self._handle_routes[handle] = route

    def _sweep_locked(self, now: float) -> None:
        cutoff = now - self._route_ttl_seconds
        for handle in [
            key for key, route in self._handle_routes.items() if route[1] < cutoff
        ]:
            route = self._handle_routes.pop(handle, None)
            if route is not None and self._handle_generations.get(handle) == route[2]:
                self._handle_generations.pop(handle, None)
        for handle in [
            key
            for key, timestamp in self._handle_collisions.items()
            if timestamp < cutoff
        ]:
            self._handle_collisions.pop(handle, None)
            self._handle_generations.pop(handle, None)
        self._last_cleanup = now

    @staticmethod
    def _report_error(reason: str, value: int = 1) -> None:
        try:
            kmonitor.report(
                AccMetrics.VIT_RPC_PROXY_ERROR_QPS_METRIC,
                value,
                {"source": "vit_proxy", "reason": reason},
            )
        except Exception as error:  # noqa: BLE001 - metrics are best effort
            _safe_warning(
                "Failed to report multimodal proxy metric (exception_type=%s)",
                type(error).__name__,
            )
