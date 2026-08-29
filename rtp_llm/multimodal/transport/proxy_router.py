import logging
import numbers
import threading
import time
from collections import defaultdict
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


def _context_deadline_seconds(context, max_timeout_seconds: float) -> Optional[float]:
    remaining = None
    if context is not None and hasattr(context, "time_remaining"):
        try:
            remaining = context.time_remaining()
        except Exception as error:  # noqa: BLE001 - a missing deadline is valid
            logging.warning("Failed to read gRPC context time remaining: %s", error)
    if not isinstance(remaining, numbers.Real):
        remaining = None
    if remaining is not None and remaining <= 0:
        return None
    timeout = max_timeout_seconds if remaining is None else min(max_timeout_seconds, remaining)
    return time.monotonic() + timeout


class MMOutputProxyRouter:
    """Routes transport-owned leases through a multi-worker ViT proxy."""

    def __init__(self, connection_pool, transport_config=None):
        self._connection_pool = connection_pool
        self._handle_routes: dict[str, tuple[str, float]] = {}
        self._handle_collisions: dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_cleanup = 0.0

        if transport_config is None:
            self._route_ttl_seconds = DEFAULT_HANDLE_ROUTE_TTL_SECONDS
            self._release_timeout_seconds = DEFAULT_RELEASE_TIMEOUT_SECONDS
        else:
            slot_gc_seconds = max(
                0.0, transport_config.rdma.slot_gc_timeout_ms / 1000.0
            )
            self._route_ttl_seconds = max(
                1.0, slot_gc_seconds + HANDLE_ROUTE_GC_SAFETY_SECONDS
            )
            self._release_timeout_seconds = max(
                0.001, transport_config.control.release_timeout_ms / 1000.0
            )

    def record_receipt(
        self, worker_address: str, receipt: MultimodalOutputPB
    ) -> None:
        handles = [
            slot.rdma_descriptor.lease_id
            for slot in receipt.output_rdma_slots
            if slot.rdma_descriptor.lease_id
        ]
        if not handles:
            return

        now = time.monotonic()
        collisions = []
        with self._lock:
            if now - self._last_cleanup >= HANDLE_ROUTE_CLEANUP_INTERVAL_SECONDS:
                self._sweep_locked(now)
            for handle in handles:
                if handle in self._handle_collisions:
                    continue
                existing = self._handle_routes.get(handle)
                if existing is not None and existing[0] != worker_address:
                    self._handle_routes.pop(handle, None)
                    self._handle_collisions[handle] = now
                    collisions.append((handle, existing[0], worker_address))
                    continue
                self._handle_routes[handle] = (worker_address, now)

        for handle, old_worker, new_worker in collisions:
            logging.warning(
                "RDMA handle %s was issued by both %s and %s; route poisoned",
                handle,
                old_worker,
                new_worker,
            )
            self._report_error("rdma_handle_collision")

    def release(self, request: ReleaseLeasePB, context) -> None:
        handles_by_worker: dict[str, list[str]] = defaultdict(list)
        skipped_routes: list[tuple[str, str]] = []
        now = time.monotonic()
        with self._lock:
            for handle in request.lease_id:
                if self._handle_collisions.pop(handle, None) is not None:
                    skipped_routes.append((handle, "release_handle_collision"))
                    continue
                route = self._handle_routes.pop(handle, None)
                if route is None:
                    skipped_routes.append((handle, "release_handle_unknown"))
                    continue
                if now - route[1] > self._route_ttl_seconds:
                    skipped_routes.append((handle, "release_handle_expired"))
                    continue
                handles_by_worker[route[0]].append(handle)
            if now - self._last_cleanup >= HANDLE_ROUTE_CLEANUP_INTERVAL_SECONDS:
                self._sweep_locked(now)

        for handle, reason in skipped_routes:
            logging.warning(
                "Cannot route RDMA handle %s (%s); worker GC will reclaim it",
                handle,
                reason,
            )
            self._report_error(reason)

        deadline = _context_deadline_seconds(context, self._release_timeout_seconds)
        if deadline is None and handles_by_worker:
            skipped_count = sum(len(handles) for handles in handles_by_worker.values())
            logging.warning(
                "RDMA release deadline exhausted; skipping %d handles", skipped_count
            )
            self._report_error("release_deadline_exhausted", skipped_count)
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
                    logging.warning(
                        "RDMA release deadline exhausted; skipping remaining %d handles",
                        skipped_count,
                    )
                    self._report_error("release_deadline_exhausted", skipped_count)
                    break
                stub = self._connection_pool.get_stub(worker_address)
                stub.ReleaseRdmaLease(
                    ReleaseLeasePB(lease_id=handles), timeout=timeout
                )
            except Exception:  # noqa: BLE001 - worker GC is the backstop
                logging.exception(
                    "Failed to release RDMA handles on VIT worker %s; "
                    "worker GC will reclaim them",
                    worker_address,
                )

    def _sweep_locked(self, now: float) -> None:
        cutoff = now - self._route_ttl_seconds
        for handle in [
            key for key, route in self._handle_routes.items() if route[1] < cutoff
        ]:
            self._handle_routes.pop(handle, None)
        for handle in [
            key for key, timestamp in self._handle_collisions.items() if timestamp < cutoff
        ]:
            self._handle_collisions.pop(handle, None)
        self._last_cleanup = now

    @staticmethod
    def _report_error(reason: str, value: int = 1) -> None:
        kmonitor.report(
            AccMetrics.VIT_RPC_PROXY_ERROR_QPS_METRIC,
            value,
            {"source": "vit_proxy", "reason": reason},
        )
