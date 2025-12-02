import asyncio
import logging
import socket
import threading
from typing import Any, Dict, List, Optional, Union

from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import override
from uvicorn import Config, Server
from uvicorn.loops.auto import auto_loop_setup

from rtp_llm.config.py_config_modules import PyEnvConfigs, StaticConfig
from rtp_llm.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.models.base_model import BaseModel
from rtp_llm.server.backend_server import BackendServer
from rtp_llm.utils.util import AtomicCounter

# make buffer larger to avoid throw exception "RemoteProtocolError Receive buffer too long"
MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024

StreamObjectType = Union[Dict[str, Any], BaseModel]

active_requests = AtomicCounter()
server_shutdown = False


class GracefulShutdownServer(Server):
    def set_server(self, backend_server):
        self.backend_server = backend_server

    @override
    async def shutdown(self, sockets: Optional[List[socket.socket]] = None) -> None:
        global server_shutdown
        server_shutdown = True
        global active_requests
        while active_requests.get() > 0:
            logging.info(f"wait {active_requests.get()} requests finish for 1s")
            await asyncio.sleep(1)
        self.backend_server.stop()
        await super().shutdown(sockets)


class BackendApp(object):
    def __init__(self, py_env_configs: PyEnvConfigs = StaticConfig):
        self.py_env_configs = py_env_configs
        self.backend_server = BackendServer(py_env_configs)

    def start(self, worker_info: WorkerInfo):
        self.backend_server.start(self.py_env_configs)
        app = self.create_app(worker_info)
        self.backend_server.wait_all_worker_ready()

        timeout_keep_alive = self.py_env_configs.server_config.timeout_keep_alive

        loop = "auto"
        if threading.current_thread() != threading.main_thread():
            # NOTE: asyncio
            loop = "none"
            auto_loop_setup()
            asyncio.set_event_loop(asyncio.new_event_loop())

        config = Config(
            app,
            host="0.0.0.0",
            loop=loop,
            port=worker_info.backend_server_port,
            log_config=UVICORN_LOGGING_CONFIG,
            timeout_keep_alive=timeout_keep_alive,
            h11_max_incomplete_event_size=MAX_INCOMPLETE_EVENT_SIZE,
        )

        try:
            server = GracefulShutdownServer(config)
            server.set_server(self.backend_server)
            server.run()
        except BaseException as e:
            self.backend_server.stop()
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

        def check_shutdown():
            global server_shutdown
            detail = ""
            ready = True
            if server_shutdown:
                detail = "this server has been shutdown"
                ready = False

            if not ready:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail
                )

        @app.on_event("startup")
        async def startup():
            RunVar("_default_thread_limiter").set(
                CapacityLimiter(
                    self.backend_server._global_controller.max_concurrency * 2
                )
            )

        @app.get("/health")
        @app.post("/health")
        @app.get("/GraphService/cm2_status")
        @app.post("/GraphService/cm2_status")
        @app.get("/SearchService/cm2_status")
        @app.post("/SearchService/cm2_status")
        @app.get("/status")
        @app.post("/status")
        @app.post("/health_check")
        async def health_check():
            check_shutdown()
            return "ok"

        @app.get("/")
        async def health():
            check_shutdown()
            return {"status": "home"}
        return app
