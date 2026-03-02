"""
VIT Proxy Server - 主进程代理服务器
负责接收外部请求并分发到工作进程，解决 SO_REUSEPORT 流量打偏问题
"""

import logging
import threading
from collections import defaultdict
from concurrent import futures
from typing import List

import grpc

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
    StatusVersionPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics


class LoadBalancer:
    """负载均衡器，支持轮询和最少连接算法"""

    def __init__(self, worker_addresses: List[str], strategy: str = "round_robin"):
        """
        Args:
            worker_addresses: 工作进程地址列表，格式如 ['localhost:9202', 'localhost:9203']
            strategy: 负载均衡策略，'round_robin' 或 'least_connections'
        """
        self.worker_addresses = worker_addresses
        self.strategy = strategy
        self.current_index = 0
        self.connection_counts = defaultdict(int)  # 记录每个工作进程的连接数
        self.lock = threading.Lock()

    def get_worker(self) -> str:
        """获取下一个工作进程地址"""
        with self.lock:
            if not self.worker_addresses:
                raise RuntimeError("No worker addresses available")

            if self.strategy == "round_robin":
                worker = self.worker_addresses[self.current_index]
                self.current_index = (self.current_index + 1) % len(
                    self.worker_addresses
                )
                return worker
            elif self.strategy == "least_connections":
                # 选择连接数最少的工作进程
                min_connections = min(
                    self.connection_counts[addr] for addr in self.worker_addresses
                )
                candidates = [
                    addr
                    for addr in self.worker_addresses
                    if self.connection_counts[addr] == min_connections
                ]
                # 如果有多个候选，使用轮询选择
                worker = candidates[self.current_index % len(candidates)]
                self.current_index = (self.current_index + 1) % len(candidates)
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

    def __init__(self, worker_addresses: List[str]):
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
    ):
        self.load_balancer = load_balancer
        self.connection_pool = connection_pool
        kmonitor.init()

    def RemoteMultimodalEmbedding(
        self, request: MultimodalInputsPB, context
    ) -> MultimodalOutputPB:
        """将请求转发到工作进程"""
        # 在 proxy 层记录 QPS
        kmonitor.report(AccMetrics.VIT_QPS_METRIC, 1, {"source": "vit_proxy"})

        worker_address = None
        try:
            # 选择工作进程
            worker_address = self.load_balancer.get_worker()
            self.load_balancer.increment_connections(worker_address)

            # 获取工作进程的 stub
            stub = self.connection_pool.get_stub(worker_address)

            # 转发请求
            logging.debug(
                f"Forwarding request to worker {worker_address}, "
                f"connections: {self.load_balancer.connection_counts[worker_address]}"
            )
            response = stub.RemoteMultimodalEmbedding(request)

            # 在 proxy 层记录成功 QPS
            kmonitor.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)
            return response
        except grpc.RpcError as e:
            logging.error(
                f"RPC error when forwarding to worker {worker_address}: {e.code()} - {e.details()}"
            )
            # 在 proxy 层记录错误 QPS
            kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            raise
        except Exception as e:
            logging.error(f"Error forwarding request to worker {worker_address}: {e}")
            # 在 proxy 层记录错误 QPS
            kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            raise
        finally:
            if worker_address:
                self.load_balancer.decrement_connections(worker_address)


class VitProxyServer:
    """VIT 代理服务器主类"""

    def __init__(
        self,
        worker_addresses: List[str],
        external_grpc_port: int,
        load_balance_strategy: str = "round_robin",
    ):
        """
        Args:
            worker_addresses: 工作进程地址列表，格式如 ['localhost:9202', 'localhost:9203']
            external_grpc_port: 外部 gRPC 端口，代理服务器监听此端口
            load_balance_strategy: 负载均衡策略，'round_robin' 或 'least_connections'
        """
        self.worker_addresses = worker_addresses
        self.external_grpc_port = external_grpc_port
        self.load_balancer = LoadBalancer(worker_addresses, load_balance_strategy)
        self.connection_pool = WorkerConnectionPool(worker_addresses)
        self.rpc_server = None

    def start(self):
        """启动代理服务器"""
        # 创建 gRPC 服务器
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

        # 添加服务
        proxy_servicer = VitProxyRpcServer(self.load_balancer, self.connection_pool)
        add_MultimodalRpcServiceServicer_to_server(proxy_servicer, self.rpc_server)

        # 绑定端口（不使用 SO_REUSEPORT，因为只有一个主进程）
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
