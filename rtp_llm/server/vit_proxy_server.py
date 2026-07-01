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
from typing import Optional

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
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

# Default per-request gRPC timeout for proxy → worker forwarding. Prevents a slow or
# hung worker from exhausting the 200-thread proxy pool (see RemoteMultimodalEmbedding).
# Per-request override comes from MMPreprocessConfigPB.mm_timeout_ms if set (>0).
DEFAULT_PROXY_RPC_TIMEOUT_SECONDS = 30.0
STATUS_CHECK_TIMEOUT_SEC = 1.0
RETRYABLE_WORKER_RPC_CODES = {
    grpc.StatusCode.UNAVAILABLE,
}


def _resolve_rpc_timeout_seconds(
    request: "MultimodalInputsPB",
    default_timeout_seconds: float = DEFAULT_PROXY_RPC_TIMEOUT_SECONDS,
) -> float:
    """Pick per-request gRPC timeout. Uses the max mm_timeout_ms across the request's
    multimodal inputs (the deadline that should bound the longest preprocess); falls
    back to *default_timeout_seconds* when none is configured."""
    max_timeout_ms = 0
    for mm_input in request.multimodal_inputs:
        cfg_ms = mm_input.mm_preprocess_config.mm_timeout_ms
        if cfg_ms > max_timeout_ms:
            max_timeout_ms = cfg_ms
    return (
        max_timeout_ms / 1000.0
        if max_timeout_ms > 0
        else default_timeout_seconds
    )


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


def _cancel_status_calls(status_calls: list[tuple[str, grpc.Future]]):
    for _, status_call in status_calls:
        if not status_call.done():
            status_call.cancel()


def _get_status_call_result(
    worker_address: str, status_call: grpc.Future
) -> Optional[WorkerStatusPB]:
    try:
        worker_status = status_call.result(timeout=0)
        if worker_status.alive:
            return _set_worker_status_role(worker_status)
        logging.warning(
            "VIT worker %s reported not alive during proxy status check",
            worker_address,
        )
    except grpc.RpcError as e:
        _log_worker_status_rpc_error(worker_address, e)
    except Exception as e:
        _log_worker_status_error(worker_address, e)
    return None


def _is_retryable_worker_rpc_error(error: grpc.RpcError) -> bool:
    try:
        return error.code() in RETRYABLE_WORKER_RPC_CODES
    except Exception:
        return False


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
    ):
        self.load_balancer = load_balancer
        self.connection_pool = connection_pool
        self.default_rpc_timeout_seconds = default_rpc_timeout_seconds
        self.profiler = MMProfiler()
        kmonitor.init()

    @staticmethod
    def _abort_unavailable(context, details: str):
        abort = getattr(context, "abort", None) if context is not None else None
        if abort is not None:
            abort(grpc.StatusCode.UNAVAILABLE, details)
        if context is not None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(details)
        raise RuntimeError(details)

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
            timeout_s = _resolve_rpc_timeout_seconds(
                request, self.default_rpc_timeout_seconds
            )
            while len(attempted_workers) < len(self.load_balancer.worker_addresses):
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
                        f"timeout: {timeout_s}s"
                    )
                    worker_rpc_start_us = _now_us()
                    response = stub.RemoteMultimodalEmbedding(
                        request, timeout=timeout_s
                    )
                    self.load_balancer.set_worker_alive(worker_address, True)
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
                    raise
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

    def _get_alive_worker_status(
        self, request: StatusVersionPB, context=None
    ) -> Optional[WorkerStatusPB]:
        deadline_s = _resolve_status_check_deadline_seconds(context)
        if deadline_s is None:
            logging.warning(
                "VIT proxy status check stopped before probing workers: no "
                "status-check deadline remains"
            )
            return None

        completed_status_calls = queue.Queue()
        status_calls: list[tuple[str, grpc.Future]] = []

        for worker_address in list(self.load_balancer.worker_addresses):
            timeout_s = _resolve_status_check_timeout_seconds(deadline_s)
            if timeout_s is None:
                break
            try:
                stub = self.connection_pool.get_stub(worker_address)
                status_call = stub.GetWorkerStatus.future(request, timeout=timeout_s)
                status_call.add_done_callback(
                    lambda done_call, addr=worker_address: completed_status_calls.put(
                        (addr, done_call)
                    )
                )
                status_calls.append((worker_address, status_call))
            except grpc.RpcError as e:
                self.load_balancer.set_worker_alive(worker_address, False)
                _log_worker_status_rpc_error(worker_address, e)
            except Exception as e:
                self.load_balancer.set_worker_alive(worker_address, False)
                _log_worker_status_error(worker_address, e)

        pending_status_call_count = len(status_calls)
        alive_worker_status = None
        try:
            while pending_status_call_count > 0:
                timeout_s = _resolve_status_check_timeout_seconds(deadline_s)
                if timeout_s is None:
                    break
                try:
                    worker_address, status_call = completed_status_calls.get(
                        timeout=timeout_s
                    )
                except queue.Empty:
                    break

                pending_status_call_count -= 1
                worker_status = _get_status_call_result(worker_address, status_call)
                self.load_balancer.set_worker_alive(
                    worker_address, worker_status is not None
                )
                if worker_status and alive_worker_status is None:
                    alive_worker_status = worker_status

            if pending_status_call_count > 0:
                logging.warning(
                    "VIT proxy status check timed out waiting for %s/%s workers",
                    pending_status_call_count,
                    len(status_calls),
                )
                for worker_address, status_call in status_calls:
                    if not status_call.done():
                        self.load_balancer.set_worker_alive(worker_address, False)
        finally:
            _cancel_status_calls(status_calls)
        return alive_worker_status

    @staticmethod
    def _set_no_alive_worker_status(context):
        context.set_code(grpc.StatusCode.UNAVAILABLE)
        context.set_details("No alive VIT worker behind proxy")

    def GetWorkerStatus(self, request: StatusVersionPB, context) -> WorkerStatusPB:
        worker_status = self._get_alive_worker_status(request, context)
        if worker_status:
            return worker_status
        self._set_no_alive_worker_status(context)
        return WorkerStatusPB(role="VIT", alive=False)

    def GetCacheStatus(self, request: CacheVersionPB, context) -> CacheStatusPB:
        status_request = StatusVersionPB(
            latest_cache_version=request.latest_cache_version
        )
        if self._get_alive_worker_status(status_request, context):
            return CacheStatusPB()
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
    ):
        """
        Args:
            worker_addresses: 工作进程地址列表，格式如 ['localhost:9202', 'localhost:9203']
            external_grpc_port: 外部 gRPC 端口，代理服务器监听此端口
            load_balance_strategy: 负载均衡策略，'round_robin' 或 'least_connections'
            default_rpc_timeout_seconds: 默认 gRPC 转发超时（秒），未配置单请求超时使用
        """
        self.worker_addresses = worker_addresses
        self.external_grpc_port = external_grpc_port
        self.load_balancer = LoadBalancer(worker_addresses, load_balance_strategy)
        self.connection_pool = WorkerConnectionPool(worker_addresses)
        self.default_rpc_timeout_seconds = default_rpc_timeout_seconds
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
        self.connection_pool.close_all()

    def wait_for_termination(self):
        """等待服务器终止"""
        if self.rpc_server:
            self.rpc_server.wait_for_termination()
