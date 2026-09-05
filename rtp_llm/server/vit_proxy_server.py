"""
VIT Proxy Server - 主进程代理服务器
负责接收外部请求并分发到工作进程，解决 SO_REUSEPORT 流量打偏问题
"""

import logging
import queue
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import NamedTuple, Optional

import grpc

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    EmptyPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    ReleaseLeasePB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.mm_profiler import MMProfiler
from rtp_llm.multimodal.transport.proxy_router import MMOutputProxyRouter

# Proxy forwarding includes transport and serialization after the worker's
# preprocessing budget. The margin lets the worker return its own timeout error;
# the caller's gRPC deadline remains the hard upper bound. Worker failover shares
# one deadline, so a slow worker cannot restart this budget for every candidate.
VIT_WORKER_RPC_TIMEOUT_MARGIN_SECONDS = 5.0
DEFAULT_PROXY_RPC_TIMEOUT_SECONDS = (
    VitConfig.DEFAULT_MM_TIMEOUT_MS / 1000.0 + VIT_WORKER_RPC_TIMEOUT_MARGIN_SECONDS
)
# The proxy owns fast health decisions for its concrete child workers. FlexLB
# tracks this proxy as one aggregate VIT endpoint and intentionally tolerates a
# transient proxy status timeout until its VIT expiration window elapses.
# These two layers therefore use different health deadlines by design.
STATUS_CHECK_TIMEOUT_SEC = 1.0
# Only UNAVAILABLE triggers worker failover (and marking the worker unhealthy).
# RESOURCE_EXHAUSTED (scheduler queue backpressure) is deliberately NOT here: by
# design it is returned directly to the client (no failover to another worker),
# and — because it is not in this set — the overloaded worker is NOT marked
# unhealthy, so it keeps receiving traffic once it drains. The client decides
# whether to retry or back off.
RETRYABLE_WORKER_RPC_CODES = {
    grpc.StatusCode.UNAVAILABLE,
}


def resolve_default_rpc_timeout_seconds(
    configured_timeout_ms: Optional[int],
) -> float:
    if configured_timeout_ms is not None and configured_timeout_ms > 0:
        return configured_timeout_ms / 1000.0 + VIT_WORKER_RPC_TIMEOUT_MARGIN_SECONDS
    return DEFAULT_PROXY_RPC_TIMEOUT_SECONDS


def _resolve_rpc_timeout_seconds(
    request: "MultimodalInputsPB",
    default_timeout_seconds: float = DEFAULT_PROXY_RPC_TIMEOUT_SECONDS,
) -> float:
    """Resolve each input's complete RPC budget, then use the longest one."""
    max_timeout_seconds = 0.0
    for mm_input in request.multimodal_inputs:
        cfg_ms = mm_input.mm_preprocess_config.mm_timeout_ms
        resolved_timeout_seconds = (
            cfg_ms / 1000.0 + VIT_WORKER_RPC_TIMEOUT_MARGIN_SECONDS
            if cfg_ms > 0
            else default_timeout_seconds
        )
        max_timeout_seconds = max(max_timeout_seconds, resolved_timeout_seconds)
    return max_timeout_seconds or default_timeout_seconds


def _now_us() -> int:
    return time.monotonic_ns() // 1000


def _get_context_time_remaining_seconds(context) -> Optional[float]:
    if context is None or not hasattr(context, "time_remaining"):
        return None
    try:
        return context.time_remaining()
    except Exception as e:
        logging.warning("Failed to read gRPC context time remaining: %s", e)
        return None


def _resolve_forwarding_deadline_seconds(timeout_seconds: float, context) -> float:
    context_remaining_s = _get_context_time_remaining_seconds(context)
    if context_remaining_s is not None:
        timeout_seconds = min(timeout_seconds, max(context_remaining_s, 0.0))
    return time.monotonic() + max(timeout_seconds, 0.0)


def _resolve_forwarding_timeout_seconds(
    deadline_seconds: float,
) -> Optional[float]:
    remaining_seconds = deadline_seconds - time.monotonic()
    if remaining_seconds <= 0:
        return None
    return remaining_seconds


def _resolve_status_check_deadline_seconds(context) -> Optional[float]:
    context_remaining_s = _get_context_time_remaining_seconds(context)
    if context_remaining_s is not None and context_remaining_s <= 0:
        return None
    timeout_s = STATUS_CHECK_TIMEOUT_SEC
    if context_remaining_s is not None:
        timeout_s = min(timeout_s, context_remaining_s)
    return time.monotonic() + timeout_s


def _resolve_status_check_timeout_seconds(deadline_s: float) -> Optional[float]:
    remaining_s = deadline_s - time.monotonic()
    if remaining_s <= 0:
        return None
    return remaining_s


def _set_worker_status_role(worker_status: WorkerStatusPB) -> WorkerStatusPB:
    if not worker_status.role:
        worker_status.role = "VIT"
    return worker_status


def _log_worker_status_rpc_error(worker_address: str, error: grpc.RpcError):
    logging.warning(
        "VIT worker %s status check failed: %s - %s",
        worker_address,
        error.code(),
        error.details(),
    )


def _log_worker_status_error(worker_address: str, error: Exception):
    logging.warning(
        "VIT worker %s status check failed: %s",
        worker_address,
        error,
    )


def _get_status_call_result(
    worker_address: str, status_call: grpc.Future
) -> tuple[Optional[WorkerStatusPB], bool]:
    try:
        worker_status = status_call.result(timeout=0)
        if worker_status.alive:
            return _set_worker_status_role(worker_status), False
        logging.warning(
            "VIT worker %s reported not alive during proxy status check",
            worker_address,
        )
    except grpc.RpcError as e:
        _log_worker_status_rpc_error(worker_address, e)
        try:
            return None, e.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        except Exception:
            return None, False
    except Exception as e:
        _log_worker_status_error(worker_address, e)
    return None, False


def _is_retryable_worker_rpc_error(error: grpc.RpcError) -> bool:
    try:
        return error.code() in RETRYABLE_WORKER_RPC_CODES
    except Exception:
        return False


class StatusProbeResult(NamedTuple):
    worker_address: str
    worker_status: Optional[WorkerStatusPB]
    timed_out: bool


class _WorkerStatusProbe:
    """One in-flight worker probe shared by concurrent proxy status requests."""

    def __init__(self, future: grpc.Future):
        self.future = future
        self._lock = threading.Lock()
        self._completed = False
        self._result: Optional[StatusProbeResult] = None
        self._waiters: list[queue.Queue] = []

    def subscribe(self, completed_status_calls: queue.Queue):
        result = None
        with self._lock:
            if self._completed:
                result = self._result
            else:
                self._waiters.append(completed_status_calls)
        if result is not None:
            completed_status_calls.put(result)

    def complete(self, result: StatusProbeResult):
        with self._lock:
            if self._completed:
                return
            self._completed = True
            self._result = result
            waiters = self._waiters
            self._waiters = []
        for completed_status_calls in waiters:
            completed_status_calls.put(result)

    def cancel(self):
        if not self.future.done():
            self.future.cancel()


class LoadBalancer:
    """负载均衡器，支持轮询和最少连接算法"""

    def __init__(self, worker_addresses: list[str], strategy: str = "round_robin"):
        """
        Args:
            worker_addresses: 工作进程地址列表，格式如 ['localhost:9202', 'localhost:9203']
            strategy: 负载均衡策略，'round_robin' 或 'least_connections'
        """
        self.worker_addresses = worker_addresses
        self.strategy = strategy
        self.current_index = 0
        self.connection_counts = defaultdict(int)  # 记录每个工作进程的连接数
        self.worker_alive = {addr: True for addr in worker_addresses}
        self.lock = threading.Lock()

    def get_worker_address(self) -> str:
        """获取工作进程地址"""
        return self.worker_addresses

    def set_worker_alive(self, worker_address: str, alive: bool):
        with self.lock:
            if worker_address in self.worker_alive:
                self.worker_alive[worker_address] = alive

    def get_alive_worker_addresses(self) -> list[str]:
        with self.lock:
            return [
                addr
                for addr in self.worker_addresses
                if self.worker_alive.get(addr, True)
            ]

    def _candidate_workers(
        self, excluded_workers: Optional[set[str]] = None
    ) -> list[str]:
        excluded_workers = excluded_workers or set()
        return [
            addr
            for addr in self.worker_addresses
            if self.worker_alive.get(addr, True) and addr not in excluded_workers
        ]

    def get_worker(self, excluded_workers: Optional[set[str]] = None) -> str:
        """获取下一个工作进程地址"""
        with self.lock:
            worker_addresses = self._candidate_workers(excluded_workers)
            if not worker_addresses:
                raise RuntimeError("No healthy worker addresses available")

            if self.strategy == "round_robin":
                worker = worker_addresses[self.current_index % len(worker_addresses)]
                self.current_index += 1
                return worker
            elif self.strategy == "least_connections":
                # 选择连接数最少的工作进程
                min_connections = min(
                    self.connection_counts[addr] for addr in worker_addresses
                )
                candidates = [
                    addr
                    for addr in worker_addresses
                    if self.connection_counts[addr] == min_connections
                ]
                # 如果有多个候选，使用轮询选择。current_index 保持单调递增，
                # 仅在选择时对 candidates 取模，避免候选集合变化时偏向头部。
                worker = candidates[self.current_index % len(candidates)]
                self.current_index += 1
                return worker
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

    def increment_connections(self, worker_address: str):
        """增加工作进程的连接计数"""
        with self.lock:
            self.connection_counts[worker_address] += 1

    def decrement_connections(self, worker_address: str):
        """减少工作进程的连接计数"""
        with self.lock:
            if worker_address in self.connection_counts:
                self.connection_counts[worker_address] = max(
                    0, self.connection_counts[worker_address] - 1
                )


class WorkerConnectionPool:
    """工作进程连接池，管理到各个工作进程的 gRPC 连接"""

    def __init__(self, worker_addresses: list[str]):
        self.worker_addresses = worker_addresses
        self.channels: dict[str, grpc.Channel] = {}
        self.stubs: dict[str, MultimodalRpcServiceStub] = {}
        self.lock = threading.Lock()

    def get_stub(self, worker_address: str) -> MultimodalRpcServiceStub:
        """获取工作进程的 stub，如果不存在则创建"""
        with self.lock:
            if worker_address not in self.channels:
                channel = grpc.insecure_channel(
                    worker_address,
                    options=[
                        ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                    ],
                )
                stub = MultimodalRpcServiceStub(channel)
                self.channels[worker_address] = channel
                self.stubs[worker_address] = stub
                logging.info(f"Created connection to worker: {worker_address}")
            return self.stubs[worker_address]

    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for address, channel in self.channels.items():
                try:
                    channel.close()
                    logging.info(f"Closed connection to worker: {address}")
                except Exception as e:
                    logging.warning(f"Error closing connection to {address}: {e}")
            self.channels.clear()
            self.stubs.clear()


class VitProxyRpcServer(MultimodalRpcServiceServicer):
    """VIT 代理 RPC 服务器，将请求转发到工作进程"""

    def __init__(
        self,
        load_balancer: LoadBalancer,
        connection_pool: WorkerConnectionPool,
        default_rpc_timeout_seconds: float = DEFAULT_PROXY_RPC_TIMEOUT_SECONDS,
        transport_config=None,
    ):
        self.load_balancer = load_balancer
        self.connection_pool = connection_pool
        self.default_rpc_timeout_seconds = default_rpc_timeout_seconds
        self.profiler = MMProfiler()
        self._transport_router = MMOutputProxyRouter(
            connection_pool, transport_config
        )
        kmonitor.init()
        self._status_probes: dict[str, _WorkerStatusProbe] = {}
        self._status_probes_lock = threading.Lock()
        self._worker_count_metric_lock = threading.Lock()

    @staticmethod
    def _abort_unavailable(context, details: str):
        abort = getattr(context, "abort", None) if context is not None else None
        if abort is not None:
            abort(grpc.StatusCode.UNAVAILABLE, details)
        if context is not None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(details)
        raise RuntimeError(details)

    @staticmethod
    def _abort_deadline_exceeded(context, details: str):
        abort = getattr(context, "abort", None) if context is not None else None
        if abort is not None:
            abort(grpc.StatusCode.DEADLINE_EXCEEDED, details)
        if context is not None:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(details)
        raise RuntimeError(details)

    @staticmethod
    def _abort_with_worker_status(context, error: grpc.RpcError):
        """Propagate a worker's non-retryable status (e.g. RESOURCE_EXHAUSTED
        overload) to the proxy's caller verbatim. A bare re-raise of the
        client-side RpcError would surface as UNKNOWN. Mirrors _abort_unavailable
        so both real and test contexts carry the exact code/details."""
        code = error.code()
        details = error.details()
        abort = getattr(context, "abort", None) if context is not None else None
        if abort is not None:
            abort(code, details)
        if context is not None:
            context.set_code(code)
            context.set_details(details)
        raise error

    def RemoteMultimodalEmbedding(
        self, request: MultimodalInputsPB, context
    ) -> MultimodalOutputPB:
        """将请求转发到工作进程"""
        tags = {"source": "vit_proxy"}
        rpc_start_us = _now_us()
        lifecycle_reported = False

        def _report_lifecycle():
            nonlocal lifecycle_reported
            if lifecycle_reported:
                return
            lifecycle_reported = True
            kmonitor.report(
                GaugeMetrics.VIT_RPC_PROXY_LIFECYCLE_RT_US_METRIC,
                _now_us() - rpc_start_us,
                tags,
            )

        callback_added = False
        if hasattr(context, "add_callback"):
            callback_added = context.add_callback(_report_lifecycle)

        kmonitor.report(AccMetrics.VIT_QPS_METRIC, 1, tags)
        kmonitor.report(
            GaugeMetrics.VIT_RPC_REQUEST_BYTES_METRIC, request.ByteSize(), tags
        )

        attempted_workers: set[str] = set()
        last_error: Optional[Exception] = None
        exhausted_workers = False
        try:
            request_timeout_s = _resolve_rpc_timeout_seconds(
                request, self.default_rpc_timeout_seconds
            )
            forwarding_deadline_s = _resolve_forwarding_deadline_seconds(
                request_timeout_s, context
            )
            timeout_exhausted = False
            while len(attempted_workers) < len(self.load_balancer.worker_addresses):
                attempt_timeout_s = _resolve_forwarding_timeout_seconds(
                    forwarding_deadline_s
                )
                if attempt_timeout_s is None:
                    timeout_exhausted = True
                    break
                worker_address = None
                try:
                    worker_address = self.load_balancer.get_worker(attempted_workers)
                except RuntimeError as e:
                    last_error = e
                    break

                attempted_workers.add(worker_address)
                self.load_balancer.increment_connections(worker_address)
                try:
                    try:
                        stub = self.connection_pool.get_stub(worker_address)
                    except Exception as e:
                        last_error = e
                        logging.error(
                            "Error getting stub for worker %s: %s",
                            worker_address,
                            e,
                        )
                        kmonitor.report(
                            AccMetrics.VIT_RPC_PROXY_ERROR_QPS_METRIC,
                            1,
                            {
                                "source": "vit_proxy",
                                "reason": "exception",
                                "worker": worker_address,
                            },
                        )
                        self.load_balancer.set_worker_alive(worker_address, False)
                        continue

                    logging.debug(
                        f"Forwarding request to worker {worker_address}, "
                        "connections: "
                        f"{self.load_balancer.connection_counts[worker_address]}, "
                        f"timeout: {attempt_timeout_s}s"
                    )
                    worker_rpc_start_us = _now_us()
                    response = stub.RemoteMultimodalEmbedding(
                        request, timeout=attempt_timeout_s
                    )
                    self.load_balancer.set_worker_alive(worker_address, True)
                    self._transport_router.record_receipt(worker_address, response)
                    kmonitor.report(
                        GaugeMetrics.VIT_RPC_PROXY_TO_WORKER_RT_US_METRIC,
                        _now_us() - worker_rpc_start_us,
                        {"source": "vit_proxy", "worker": worker_address},
                    )
                    kmonitor.report(
                        GaugeMetrics.VIT_RPC_RESPONSE_BYTES_METRIC,
                        response.ByteSize(),
                        tags,
                    )

                    kmonitor.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)
                    self.profiler.on_request_complete()

                    return response
                except grpc.RpcError as e:
                    last_error = e
                    logging.error(
                        "RPC error when forwarding to worker %s: %s - %s",
                        worker_address,
                        e.code(),
                        e.details(),
                    )
                    kmonitor.report(
                        AccMetrics.VIT_RPC_PROXY_ERROR_QPS_METRIC,
                        1,
                        {
                            "source": "vit_proxy",
                            "reason": "grpc_error",
                            "grpc_code": str(e.code()),
                            "worker": worker_address or "unknown",
                        },
                    )
                    if worker_address and _is_retryable_worker_rpc_error(e):
                        self.load_balancer.set_worker_alive(worker_address, False)
                        continue
                    # Non-retryable (e.g. RESOURCE_EXHAUSTED overload): propagate the
                    # worker's exact status to our caller instead of a bare re-raise
                    # (which the framework downgrades to UNKNOWN). Do NOT retry
                    # another worker or mark this one unhealthy.
                    self._abort_with_worker_status(context, e)
                except Exception as e:
                    logging.error(
                        "Error forwarding request to worker %s: %s",
                        worker_address,
                        e,
                    )
                    kmonitor.report(
                        AccMetrics.VIT_RPC_PROXY_ERROR_QPS_METRIC,
                        1,
                        {
                            "source": "vit_proxy",
                            "reason": "exception",
                            "worker": worker_address or "unknown",
                        },
                    )
                    raise
                finally:
                    if worker_address:
                        self.load_balancer.decrement_connections(worker_address)

            if timeout_exhausted:
                self._abort_deadline_exceeded(
                    context, "VIT proxy forwarding timeout exhausted"
                )

            exhausted_workers = True
            details = "No healthy VIT worker behind proxy"
            if last_error:
                details += f": {last_error}"
            self._abort_unavailable(context, details)
        except grpc.RpcError as e:
            kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            raise
        except Exception as e:
            logging.error("Error forwarding request after proxy retries: %s", e)
            kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            if exhausted_workers:
                kmonitor.report(
                    AccMetrics.VIT_RPC_PROXY_ERROR_QPS_METRIC,
                    1,
                    {"source": "vit_proxy", "reason": "all_workers_unavailable"},
                )
            raise
        finally:
            if not callback_added:
                _report_lifecycle()

    def _report_worker_counts(self):
        with self._worker_count_metric_lock:
            healthy_worker_count = len(self.load_balancer.get_alive_worker_addresses())
            total_worker_count = len(self.load_balancer.worker_addresses)
        tags = {"source": "vit_proxy"}
        kmonitor.report(
            GaugeMetrics.VIT_RPC_PROXY_HEALTHY_WORKER_COUNT_METRIC,
            healthy_worker_count,
            tags,
        )
        kmonitor.report(
            GaugeMetrics.VIT_RPC_PROXY_TOTAL_WORKER_COUNT_METRIC,
            total_worker_count,
            tags,
        )

    def _complete_status_probe(
        self,
        worker_address: str,
        probe: _WorkerStatusProbe,
        status_call: grpc.Future,
    ):
        worker_status, worker_timed_out = _get_status_call_result(
            worker_address, status_call
        )
        self.load_balancer.set_worker_alive(worker_address, worker_status is not None)
        with self._status_probes_lock:
            if self._status_probes.get(worker_address) is probe:
                self._status_probes.pop(worker_address, None)
        probe.complete(
            StatusProbeResult(worker_address, worker_status, worker_timed_out)
        )
        self._report_worker_counts()

    def _fail_status_probe_subscription(
        self,
        worker_address: str,
        probe: Optional[_WorkerStatusProbe],
        new_probe: bool,
        subscribed: bool,
        completed_status_calls: queue.Queue,
        worker_timed_out: bool,
    ):
        result = StatusProbeResult(worker_address, None, worker_timed_out)
        if new_probe and probe is not None:
            with self._status_probes_lock:
                if self._status_probes.get(worker_address) is probe:
                    self._status_probes.pop(worker_address, None)

            # A callback registration failure can happen after other requests
            # subscribed to this probe. Complete them all before cancellation.
            if not subscribed:
                completed_status_calls.put(result)
            probe.complete(result)
            probe.cancel()
            return

        # Failures before creating a probe, or while subscribing to an existing
        # probe, apply only to the current status request.
        completed_status_calls.put(result)

    def _subscribe_status_probe(
        self,
        worker_address: str,
        request: StatusVersionPB,
        timeout_s: float,
        completed_status_calls: queue.Queue,
    ):
        probe: Optional[_WorkerStatusProbe] = None
        new_probe = False
        subscribed = False
        try:
            with self._status_probes_lock:
                # GetWorkerStatus currently ignores StatusVersionPB fields, so an
                # in-flight probe is reusable across callers and cache versions.
                # If worker responses become request-dependent, include a request
                # fingerprint in this key instead of reusing by address alone.
                probe = self._status_probes.get(worker_address)
                if probe is None:
                    stub = self.connection_pool.get_stub(worker_address)
                    status_call = stub.GetWorkerStatus.future(
                        request, timeout=timeout_s
                    )
                    probe = _WorkerStatusProbe(status_call)
                    self._status_probes[worker_address] = probe
                    new_probe = True
                probe.subscribe(completed_status_calls)
                subscribed = True

            if new_probe:
                probe.future.add_done_callback(
                    lambda done_call: self._complete_status_probe(
                        worker_address, probe, done_call
                    )
                )
        except grpc.RpcError as e:
            self.load_balancer.set_worker_alive(worker_address, False)
            _log_worker_status_rpc_error(worker_address, e)
            try:
                worker_timed_out = e.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            except Exception:
                worker_timed_out = False
            self._fail_status_probe_subscription(
                worker_address,
                probe,
                new_probe,
                subscribed,
                completed_status_calls,
                worker_timed_out,
            )
            self._report_worker_counts()
        except Exception as e:
            self.load_balancer.set_worker_alive(worker_address, False)
            _log_worker_status_error(worker_address, e)
            self._fail_status_probe_subscription(
                worker_address,
                probe,
                new_probe,
                subscribed,
                completed_status_calls,
                False,
            )
            self._report_worker_counts()

    def cancel_status_probes(self):
        with self._status_probes_lock:
            probes = list(self._status_probes.values())
            self._status_probes.clear()
        for probe in probes:
            probe.cancel()

    def ReleaseRdmaLease(
        self, request: ReleaseLeasePB, context
    ) -> EmptyPB:
        self._transport_router.release(request, context)
        return EmptyPB()

    def _get_alive_worker_status(
        self, request: StatusVersionPB, context=None
    ) -> tuple[Optional[WorkerStatusPB], bool]:
        deadline_s = _resolve_status_check_deadline_seconds(context)
        if deadline_s is None:
            logging.warning(
                "VIT proxy status check stopped before probing workers: no "
                "status-check deadline remains"
            )
            return None, True

        completed_status_calls = queue.Queue()
        subscribed_status_call_count = 0
        status_check_timed_out = False
        for worker_address in list(self.load_balancer.worker_addresses):
            self._subscribe_status_probe(
                worker_address,
                request,
                STATUS_CHECK_TIMEOUT_SEC,
                completed_status_calls,
            )
            subscribed_status_call_count += 1

        pending_status_call_count = subscribed_status_call_count
        while pending_status_call_count > 0:
            timeout_s = _resolve_status_check_timeout_seconds(deadline_s)
            if timeout_s is None:
                status_check_timed_out = True
                break
            try:
                worker_address, worker_status, worker_timed_out = (
                    completed_status_calls.get(timeout=timeout_s)
                )
            except queue.Empty:
                status_check_timed_out = True
                break

            pending_status_call_count -= 1
            status_check_timed_out |= worker_timed_out
            if worker_status:
                return worker_status, status_check_timed_out

        if pending_status_call_count > 0:
            logging.warning(
                "VIT proxy status check timed out waiting for %s/%s workers; "
                "in-flight probes will continue in the background",
                pending_status_call_count,
                subscribed_status_call_count,
            )
            status_check_timed_out = True
        return None, status_check_timed_out

    @staticmethod
    def _set_no_alive_worker_status(context):
        context.set_code(grpc.StatusCode.UNAVAILABLE)
        context.set_details("No alive VIT worker behind proxy")

    @staticmethod
    def _set_status_check_timeout(context):
        context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
        context.set_details("VIT proxy status check timed out")

    def GetWorkerStatus(self, request: StatusVersionPB, context) -> WorkerStatusPB:
        worker_status, status_check_timed_out = self._get_alive_worker_status(
            request, context
        )
        if worker_status:
            return worker_status
        if status_check_timed_out:
            self._set_status_check_timeout(context)
        else:
            self._set_no_alive_worker_status(context)
        return WorkerStatusPB(role="VIT", alive=False)

    def GetCacheStatus(self, request: CacheVersionPB, context) -> CacheStatusPB:
        status_request = StatusVersionPB(
            latest_cache_version=request.latest_cache_version
        )
        worker_status, status_check_timed_out = self._get_alive_worker_status(
            status_request, context
        )
        if worker_status:
            return CacheStatusPB()
        if status_check_timed_out:
            self._set_status_check_timeout(context)
        else:
            self._set_no_alive_worker_status(context)
        return CacheStatusPB()


class VitProxyServer:
    """VIT 代理服务器主类"""

    def __init__(
        self,
        worker_addresses: list[str],
        external_grpc_port: int,
        load_balance_strategy: str = "round_robin",
        default_rpc_timeout_seconds: float = DEFAULT_PROXY_RPC_TIMEOUT_SECONDS,
        transport_config=None,
    ):
        """
        Args:
            worker_addresses: 工作进程地址列表，格式如 ['localhost:9202', 'localhost:9203']
            external_grpc_port: 外部 gRPC 端口，代理服务器监听此端口
            load_balance_strategy: 负载均衡策略，'round_robin' 或 'least_connections'
            default_rpc_timeout_seconds: 请求未指定超时时间时的默认值
            transport_config: ViT 输出通信配置
        """
        self.worker_addresses = worker_addresses
        self.external_grpc_port = external_grpc_port
        self.load_balancer = LoadBalancer(worker_addresses, load_balance_strategy)
        self.connection_pool = WorkerConnectionPool(worker_addresses)
        self.default_rpc_timeout_seconds = default_rpc_timeout_seconds
        self.transport_config = transport_config
        self.rpc_server = None
        self.proxy_servicer: Optional[VitProxyRpcServer] = None

    def start(self):
        """启动代理服务器"""
        self.rpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=200),
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                ("grpc.max_concurrent_streams", -1),
                ("grpc.http2.min_ping_interval_without_data_ms", 1000),
                ("grpc.http2.max_ping_strikes", 1000),
            ],
        )

        self.proxy_servicer = VitProxyRpcServer(
            self.load_balancer,
            self.connection_pool,
            self.default_rpc_timeout_seconds,
            self.transport_config,
        )
        add_MultimodalRpcServiceServicer_to_server(self.proxy_servicer, self.rpc_server)

        self.rpc_server.add_insecure_port(f"0.0.0.0:{self.external_grpc_port}")
        self.rpc_server.start()

        logging.info(
            f"VIT Proxy Server started on gRPC port {self.external_grpc_port}, "
            f"forwarding to {len(self.worker_addresses)} workers: {self.worker_addresses}"
        )

    def stop(self):
        """停止代理服务器"""
        if self.rpc_server:
            self.rpc_server.stop(grace=None)
            logging.info("VIT Proxy Server stopped")
        if self.proxy_servicer:
            self.proxy_servicer.cancel_status_probes()
        self.connection_pool.close_all()

    def wait_for_termination(self):
        """等待服务器终止"""
        if self.rpc_server:
            self.rpc_server.wait_for_termination()
