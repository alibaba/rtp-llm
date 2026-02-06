"""
VIT Proxy Server 启动模块
主进程：负责接收外部请求并分发到工作进程
"""

import asyncio
import logging
import socket
import threading
from typing import List, Optional

import grpc
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from setproctitle import setproctitle
from typing_extensions import override
from uvicorn import Config, Server
from uvicorn.loops.auto import auto_loop_setup

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.uvicorn_config import get_uvicorn_logging_config
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import StatusVersionPB
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
)
from rtp_llm.distribute.worker_info import update_worker_info
from rtp_llm.server.vit_proxy_server import VitProxyServer

setup_logging()


class GracefulShutdownProxyServer(Server):
    """支持优雅关闭的代理服务器"""

    def set_proxy_server(self, proxy_server: VitProxyServer):
        self.proxy_server = proxy_server

    @override
    async def shutdown(self, sockets: Optional[List[socket.socket]] = None) -> None:
        if hasattr(self, "proxy_server") and self.proxy_server:
            self.proxy_server.stop()
        await super().shutdown(sockets)


def vit_proxy_start_server(
    py_env_configs: PyEnvConfigs,
    worker_addresses: list[str],
    grpc_port: int,
    http_port: int,
):
    """
    启动 VIT 代理服务器（主进程）

    Args:
        py_env_configs: 配置对象
        worker_addresses: 工作进程地址列表，格式如 ['localhost:19202', 'localhost:19203']
        grpc_port: gRPC 端口号（从外部传入）
        http_port: HTTP 端口号（从外部传入）
    """
    setproctitle("rtp_llm_vit_proxy_server")

    logging.info(
        f"[VIT_PROXY] Starting proxy server with {len(worker_addresses)} workers, "
        f"grpc_port={grpc_port}, http_port={http_port}"
    )

    update_worker_info(
        py_env_configs.server_config.start_port,
        py_env_configs.server_config.worker_info_port_num,
        py_env_configs.distribute_config.remote_server_port,
    )

    # 创建并启动代理服务器（gRPC）
    load_balance_strategy = py_env_configs.vit_config.vit_proxy_load_balance_strategy
    proxy_server = VitProxyServer(
        worker_addresses=worker_addresses,
        external_grpc_port=grpc_port,
        load_balance_strategy=load_balance_strategy,
    )
    logging.info(f"[VIT_PROXY] Using load balance strategy: {load_balance_strategy}")

    try:
        # 启动 gRPC 服务器
        proxy_server.start()
        logging.info(
            f"[VIT_PROXY] Proxy gRPC server started successfully on port {grpc_port}"
        )

        # 创建并启动 HTTP 服务器
        app = create_proxy_app(proxy_server, worker_addresses)
        start_http_server(app, http_port, py_env_configs, proxy_server)

    except KeyboardInterrupt:
        logging.info("[VIT_PROXY] Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error(f"[VIT_PROXY] Proxy server error: {e}", exc_info=True)
        raise
    finally:
        proxy_server.stop()
        logging.info("[VIT_PROXY] Proxy server stopped")


def create_proxy_app(
    proxy_server: VitProxyServer, worker_addresses: list[str]
) -> FastAPI:
    """创建 FastAPI 应用，提供 HTTP 接口"""
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
    app = FastAPI(middleware=middleware)

    @app.get("/health")
    @app.post("/health")
    @app.post("/health_check")
    async def health():
        """
        健康检查接口
        只有当所有 VIT worker 进程都准备就绪时才返回健康状态
        """
        if not worker_addresses:
            return "ok"

        request = StatusVersionPB()
        healthy_count = 0

        for worker_address in worker_addresses:
            try:
                # 创建临时连接检查 worker 状态
                channel = grpc.insecure_channel(
                    worker_address,
                    options=[
                        ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                    ],
                )
                stub = MultimodalRpcServiceStub(channel)
                worker_status_response = stub.GetWorkerStatus(request, timeout=2)
                channel.close()

                if worker_status_response.alive:
                    healthy_count += 1
                    logging.debug(
                        f"[VIT_PROXY_HEALTH] Worker {worker_address} is ready"
                    )
                else:
                    logging.warning(
                        f"[VIT_PROXY_HEALTH] Worker {worker_address} is not alive"
                    )

            except Exception as e:
                logging.warning(
                    f"[VIT_PROXY_HEALTH] Failed to check worker {worker_address}: {e}"
                )
                continue

        # 只有当所有 worker 都健康时才返回 ok
        if healthy_count == len(worker_addresses):
            logging.debug(
                f"[VIT_PROXY_HEALTH] All {len(worker_addresses)} workers are healthy"
            )
            return "ok"
        else:
            logging.warning(
                f"[VIT_PROXY_HEALTH] Only {healthy_count}/{len(worker_addresses)} workers are healthy"
            )
            from fastapi.responses import ORJSONResponse

            return ORJSONResponse(
                status_code=503,
                content={
                    "error": f"Only {healthy_count}/{len(worker_addresses)} VIT workers are healthy"
                },
            )

    @app.get("/worker_status")
    @app.post("/worker_status")
    async def worker_status():
        """工作进程状态接口"""
        total_workers = len(proxy_server.worker_addresses)
        return {
            "status": "proxy",
            "proxy_alive": True,
            "total_workers": total_workers,
            "workers": worker_statuses,
        }

    return app


def start_http_server(
    app: FastAPI,
    http_port: int,
    py_env_configs: PyEnvConfigs,
    proxy_server: VitProxyServer,
) -> None:
    """启动 HTTP 服务器"""
    timeout_keep_alive = py_env_configs.server_config.timeout_keep_alive

    loop = "auto"
    if threading.current_thread() != threading.main_thread():
        # NOTE: asyncio
        loop = "none"
        auto_loop_setup()
        asyncio.set_event_loop(asyncio.new_event_loop())

    logging.info(f"[VIT_PROXY] Starting HTTP server on port {http_port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 代理服务器不使用 SO_REUSEPORT（只有一个主进程）
    sock.bind(("0.0.0.0", http_port))
    sock.listen()
    fd = sock.fileno()

    config = Config(
        app,
        fd=fd,
        loop=loop,
        log_config=get_uvicorn_logging_config(),
        timeout_keep_alive=timeout_keep_alive,
        h11_max_incomplete_event_size=1024 * 1024,
    )

    try:
        server = GracefulShutdownProxyServer(config)
        server.set_proxy_server(proxy_server)
        server.run()
    except BaseException as e:
        proxy_server.stop()
        raise e
