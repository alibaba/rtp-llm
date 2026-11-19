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
    timeout = (
        max_timeout_seconds
        if remaining is None
        else min(max_timeout_seconds, remaining)
    )
    return time.monotonic() + timeout


class MMOutputProxyRouter:
    """Routes transport-owned leases through a multi-worker ViT proxy."""

    def __init__(self, connection_pool, transport_config=None, worker_addresses=()):
        self._connection_pool = connection_pool
        self._worker_addresses = tuple(worker_addresses)
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
            slot_gc_seconds = max(
                0.0, transport_config.rdma.slot_gc_timeout_ms / 1000.0
            )
            self._route_ttl_seconds = max(
                1.0, slot_gc_seconds + HANDLE_ROUTE_GC_SAFETY_SECONDS
            )
            self._release_timeout_seconds = max(
                0.001, transport_config.control.release_timeout_ms / 1000.0
            )

    def record_receipt(self, worker_address: str, receipt: MultimodalOutputPB) -> None:
        handles = list(
            dict.fromkeys(
                slot.rdma_descriptor.lease_id
                for slot in receipt.output_rdma_slots
                if slot.rdma_descriptor.lease_id
            )
        )
        if not handles:
            return

        now = time.monotonic()
        collisions = []
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
                if existing is not None and existing[0] != worker_address:
                    self._handle_routes.pop(handle, None)
                    self._handle_collisions[handle] = now
                    collisions.append((handle, existing[0], worker_address))
                    continue
                self._handle_routes[handle] = (worker_address, now, generation)

        for handle, old_worker, new_worker in collisions:
            logging.warning(
                "RDMA handle %s was issued by both %s and %s; route poisoned",
                handle,
                old_worker,
                new_worker,
            )
            self._report_error("rdma_handle_collision")

    def claim(self, request: ReleaseLeasePB, context) -> None:
        # One embedding receipt always belongs to one worker. Claim it atomically there.
        with self._lock:
            owners = {
                self._handle_routes[h][0] if h in self._handle_routes else None
                for h in request.lease_id
            }
        if len(owners) != 1 or None in owners:
            raise ValueError("RDMA offer route is unknown or ambiguous")
        deadline = _context_deadline_seconds(context, self._release_timeout_seconds)
        if deadline is None:
            raise TimeoutError("RDMA claim deadline exhausted")
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("RDMA claim deadline exhausted")
        self._connection_pool.get_stub(owners.pop()).ClaimRdmaLease(
            request, timeout=timeout
        )

    def release(self, request: ReleaseLeasePB, context) -> None:
        groups: dict[str, list[str]] = defaultdict(list)
        routes = {}
        with self._lock:
            now = time.monotonic()
            if now - self._last_cleanup >= HANDLE_ROUTE_CLEANUP_INTERVAL_SECONDS:
                self._sweep_locked(now)
            for handle in dict.fromkeys(request.lease_id):
                if handle in self._handle_collisions:
                    self._report_error("release_handle_collision")
                    raise ValueError("RDMA release handle has conflicting owners")
                route = self._handle_routes.get(handle)
                if route is not None:
                    groups[route[0]].append(handle)
                    routes[handle] = route
                else:
                    # A claim can outlive its proxy route or a proxy restart. Native
                    # lease IDs are globally unique and releasing an unknown ID is a no-op.
                    for worker in self._worker_addresses:
                        groups[worker].append(handle)

        deadline = _context_deadline_seconds(context, self._release_timeout_seconds)
        failure = None
        for worker, handles in groups.items():
            timeout = deadline - time.monotonic() if deadline is not None else 0
            if timeout <= 0:
                self._report_error("release_deadline_exhausted")
                raise TimeoutError("RDMA release deadline exhausted")
            try:
                self._connection_pool.get_stub(worker).ReleaseRdmaLease(
                    ReleaseLeasePB(lease_id=handles), timeout=timeout
                )
            except Exception as exc:
                failure = exc
                continue
            with self._lock:
                for handle in handles:
                    if (
                        handle in routes
                        and self._handle_routes.get(handle) == routes[handle]
                    ):
                        del self._handle_routes[handle]
                        self._handle_generations.pop(handle, None)
        if failure is not None:
            raise RuntimeError(
                "RDMA release was not acknowledged by every worker"
            ) from failure

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
        kmonitor.report(
            AccMetrics.VIT_RPC_PROXY_ERROR_QPS_METRIC,
            value,
            {"source": "vit_proxy", "reason": reason},
        )
