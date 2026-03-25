"""
VIT Proxy Server 启动模块
主进程：负责接收外部请求并分发到工作进程
"""

import asyncio
import json
import logging
import socket
import threading
import urllib.request
from functools import partial
from typing import Optional

import grpc
from fastapi import FastAPI
from fastapi import Request as RawRequest
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
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
from rtp_llm.server.vit_proxy_server import VitProxyServer

setup_logging()


class GracefulShutdownProxyServer(Server):
    """支持优雅关闭的代理服务器"""

    def set_proxy_server(self, proxy_server: VitProxyServer):
        self.proxy_server = proxy_server

    @override
    async def shutdown(self, sockets: Optional[list[socket.socket]] = None) -> None:
        if hasattr(self, "proxy_server") and self.proxy_server:
            self.proxy_server.stop()
        await super().shutdown(sockets)


def vit_proxy_start_server(
    py_env_configs: PyEnvConfigs,
    worker_addresses: list[str],
    grpc_port: int,
    http_port: int,
    worker_http_addresses: Optional[list[str]] = None,
):
    """
    启动 VIT 代理服务器（主进程）

    Args:
        py_env_configs: 配置对象
        worker_addresses: 工作进程 gRPC 地址列表
        grpc_port: gRPC 端口号（从外部传入）
        http_port: HTTP 端口号（从外部传入）
        worker_http_addresses: 工作进程 HTTP 地址列表（用于 profile 转发）
    """
    setproctitle("rtp_llm_vit_proxy_server")

    logging.info(
        f"[VIT_PROXY] Starting proxy server with {len(worker_addresses)} workers, "
        f"grpc_port={grpc_port}, http_port={http_port}"
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
        app = create_proxy_app(
            proxy_server, worker_addresses, worker_http_addresses or []
        )
        start_http_server(app, http_port, py_env_configs, proxy_server)

    except KeyboardInterrupt:
        logging.info("[VIT_PROXY] Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error(f"[VIT_PROXY] Proxy server error: {e}", exc_info=True)
        raise
    finally:
        proxy_server.stop()
        logging.info("[VIT_PROXY] Proxy server stopped")


def _forward_http(addr: str, path: str, body: Optional[dict] = None) -> dict:
    """Send a single HTTP request to a worker and return the parsed JSON."""
    url = f"http://{addr}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if path != "/profile_status" else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _forward_to_workers(
    worker_http_addresses: list[str],
    path: str,
    bodies: Optional[list[Optional[dict]]] = None,
) -> dict:
    """Forward HTTP requests to all workers concurrently.

    Args:
        worker_http_addresses: Worker HTTP address list.
        path: HTTP path to call on each worker.
        bodies: Per-worker request bodies.  ``None`` sends no body to any
            worker.  A single-element list broadcasts the same body to all.
            Otherwise the length must match *worker_http_addresses*.
    """
    n = len(worker_http_addresses)
    if bodies is None:
        per_worker = [None] * n
    elif len(bodies) == 1:
        per_worker = bodies * n
    else:
        per_worker = bodies

    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(None, partial(_forward_http, addr, path, body))
        for addr, body in zip(worker_http_addresses, per_worker)
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    results = {}
    for i, resp in enumerate(responses):
        key = f"rank_{i}"
        if isinstance(resp, Exception):
            results[key] = {"status": "error", "message": str(resp)}
        else:
            results[key] = resp
    return results


def create_proxy_app(
    proxy_server: VitProxyServer,
    worker_addresses: list[str],
    worker_http_addresses: list[str],
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

        if healthy_count == len(worker_addresses):
            logging.debug(
                f"[VIT_PROXY_HEALTH] All {len(worker_addresses)} workers are healthy"
            )
            return "ok"
        else:
            logging.warning(
                f"[VIT_PROXY_HEALTH] Only {healthy_count}/{len(worker_addresses)} workers are healthy"
            )
            return ORJSONResponse(
                status_code=503,
                content={
                    "error": f"Only {healthy_count}/{len(worker_addresses)} VIT workers are healthy"
                },
            )

    # ------------------------------------------------------------------
    #  Profile forwarding: proxy → all workers
    # ------------------------------------------------------------------

    @app.post("/start_profile")
    async def start_profile(request: RawRequest):
        if not worker_http_addresses:
            return ORJSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "No worker HTTP addresses configured",
                },
            )
        body = await request.json()
        base_output_path = body.get("output_path", "./vit_profile")
        per_worker_bodies = []
        for i in range(len(worker_http_addresses)):
            wb = dict(body)
            wb["output_path"] = f"{base_output_path}/rank_{i}"
            per_worker_bodies.append(wb)
        results = await _forward_to_workers(
            worker_http_addresses,
            "/start_profile",
            per_worker_bodies,
        )
        return ORJSONResponse({"status": "forwarded", "workers": results})

    @app.post("/end_profile")
    async def end_profile():
        if not worker_http_addresses:
            return ORJSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "No worker HTTP addresses configured",
                },
            )
        results = await _forward_to_workers(worker_http_addresses, "/end_profile")
        return ORJSONResponse({"status": "forwarded", "workers": results})

    @app.get("/profile_status")
    async def profile_status():
        if not worker_http_addresses:
            return ORJSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "No worker HTTP addresses configured",
                },
            )
        results = await _forward_to_workers(worker_http_addresses, "/profile_status")
        return ORJSONResponse({"status": "forwarded", "workers": results})

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
