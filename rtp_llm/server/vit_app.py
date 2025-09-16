import asyncio
import logging
import socket
import threading
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi import Request as RawRequest
from fastapi import status
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from typing_extensions import override
from uvicorn import Config, Server
from uvicorn.loops.auto import auto_loop_setup

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.uvicorn_config import get_uvicorn_logging_config
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.ops import RoleType
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer, create_rpc_server


class GracefulShutdownServer(Server):
    def set_server(self, vit_endpoint_server):
        self.vit_endpoint_server = vit_endpoint_server

    @override
    async def shutdown(self, sockets: Optional[List[socket.socket]] = None) -> None:
        self.vit_endpoint_server.stop()
        await super().shutdown(sockets)


class VitEndpointApp:
    def __init__(self, py_env_configs: PyEnvConfigs):
        self.py_env_configs = py_env_configs
        self.vit_endpoint_server = VitEndpointServer(self.py_env_configs)

    def start(self, worker_info: WorkerInfo):
        self.vit_endpoint_server.start(worker_info)

        app = self.create_app(worker_info)

        timeout_keep_alive = self.py_env_configs.server_config.timeout_keep_alive

        loop = "auto"
        if threading.current_thread() != threading.main_thread():
            # NOTE: asyncio
            loop = "none"
            auto_loop_setup()
            asyncio.set_event_loop(asyncio.new_event_loop())

        http_port = (
            worker_info.server_port
            if self.py_env_configs.role_config.role_type == RoleType.VIT
            else worker_info.vit_http_server_port
        )

        logging.info(f"Vit App start in http port {http_port}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(("0.0.0.0", http_port))
        sock.listen()
        fd = sock.fileno()
        timeout_keep_alive = self.py_env_configs.server_config.timeout_keep_alive

        # use http port, as vit endpoint server is only constructed when vit sep
        config = Config(
            app,
            fd=fd,
            loop=loop,
            log_config=get_uvicorn_logging_config(),
            timeout_keep_alive=timeout_keep_alive,
            h11_max_incomplete_event_size=1024 * 1024,
        )

        try:
            server = GracefulShutdownServer(config)
            server.set_server(self.vit_endpoint_server)
            server.run()
        except BaseException as e:
            self.vit_endpoint_server.stop()
            raise e

    def create_app(self, worker_info: WorkerInfo):
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
            return "ok"

        @app.get("/worker_status")
        @app.post("/worker_status")
        async def worker_status():
            return self.vit_endpoint_server.worker_status()

        return app


class VitEndpointServer:
    def __init__(self, py_env_configs: PyEnvConfigs):
        self.rpc_server = None
        self.mm_rpc_server = None
        self.py_env_configs = py_env_configs

        engine_config = EngineConfig.create(py_env_configs)

        model_config = ModelFactory.create_model_config(
            model_args=py_env_configs.model_args,
            lora_config=py_env_configs.lora_config,
            kv_cache_config=engine_config.kv_cache_config,
            profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
            generate_env_config=py_env_configs.generate_env_config,
            embedding_config=py_env_configs.embedding_config,
            quantization_config=py_env_configs.quantization_config,
            render_config=py_env_configs.render_config,
            eplb_config=py_env_configs.eplb_config,
        )

        self.mm_process_engine = ModelFactory.create_vit_from_env(
            model_config, engine_config, py_env_configs.vit_config, py_env_configs
        )
        if self.mm_process_engine is None:
            return

        self.mm_rpc_server = MultimodalRpcServer(self.mm_process_engine)
        self.rpc_server = create_rpc_server()
        add_MultimodalRpcServiceServicer_to_server(self.mm_rpc_server, self.rpc_server)
        kmonitor.init()

    def start(self, worker_info: WorkerInfo):
        if self.mm_process_engine is None:
            return
        grpc_port = (
            worker_info.rpc_server_port
            if self.py_env_configs.role_config.role_type == RoleType.VIT
            else worker_info.vit_grpc_server_port
        )
        self.rpc_server.add_insecure_port(f"0.0.0.0:{grpc_port}")
        self.rpc_server.start()
        logging.info(f"Vit Server start in grpc port {grpc_port}")

    def stop(self):
        if self.rpc_server is not None:
            self.rpc_server.stop(grace=None)
        if self.mm_rpc_server is not None:
            self.mm_rpc_server.stop()

    def worker_status(self):
        return {}
