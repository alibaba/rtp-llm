import asyncio
import json
import logging
import logging.config
import os
import socket
import sys
import threading
import time
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

import uvicorn
from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI, HTTPException
from fastapi import Request as RawRequest
from fastapi import status
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from typing_extensions import override
from uvicorn import Config, Server
from uvicorn.loops.auto import auto_loop_setup

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import PyEnvConfigs, StaticConfig
from rtp_llm.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info, g_worker_info
from rtp_llm.embedding.backend_embedding_app import register_backend_embedding_api
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import BaseModel
from rtp_llm.server.backend_server import BackendServer
from rtp_llm.server.misc import check_is_master, check_is_worker
from rtp_llm.server.worker_status import CacheStatus, TaskInfo, WorkStatus
from rtp_llm.utils.util import AtomicCounter
from rtp_llm.utils.version_info import VersionInfo

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
            elif self.backend_server.ready() == False:
                detail = "inference server is not ready"
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

        @app.post("/worker_status")
        def worker_status(req: Dict[str, Any]):
            check_shutdown()
            latest_cache_version: int = int(req.get("latest_cache_version", -1))
            latest_finised_version: int = int(req.get("latest_finised_version", -1))
            worker_status = self.backend_server.get_worker_status(latest_cache_version, latest_finised_version)
            worker_status.server_port=worker_info.server_port
            worker_status.http_port=worker_info.http_port
            worker_status.grpc_port=worker_info.rpc_server_port
            
            return ORJSONResponse(content=worker_status.model_dump(exclude_none=True))

        # entry for worker RANK != 0
        @app.post("/inference_internal")
        @check_is_worker()
        async def inference_internal(
            req: Union[str, Dict[Any, Any]], raw_request: RawRequest
        ):
            global active_requests
            active_requests.increment()
            try:
                return await self.backend_server.inference(req, raw_request)
            finally:
                active_requests.decrement()

        @app.post("/add_lora_internal")
        @check_is_worker()
        def add_lora_internal(req: Dict[str, str]):
            self.backend_server.add_lora(req)

        @app.post("/remove_lora_internal")
        @check_is_worker()
        def remove_lora_internal(req: Dict[str, str]):
            self.backend_server.remove_lora(req)

        @app.post("/update_scheduler_info")
        def update_scheduler_info(req: Union[str, Dict[str, Any]]):
            self.backend_server.update_scheduler_info(req)
            return {"status": "ok"}

        # update for worker RANK == 0
        @app.post("/update")
        @check_is_master()
        def update(version_info: VersionInfo):
            try:
                return self.backend_server.update(version_info)
            except Exception as e:
                return {"error": f"Failed to update", "details": str(e)}

        # request format: {"log_level": "DEBUG"}, {"log_level": "info"}
        @app.post("/set_log_level")
        async def set_log_level(req: Union[str, Dict[Any, Any]]):
            try:
                if self.backend_server.set_log_level(req):
                    return {"status": "ok"}
                else:
                    return {"status": "set log level failed"}
            except Exception as e:
                return {"error": str(e)}

        # request format: {"mode": "NONE", "update_time": 5000}
        @app.post("/update_eplb_config")
        async def update_eplb_config(req: Dict[Any, Any]):
            # TODO(yinzhi): support manual set eplb config
            try:
                logging.info(f"update eplb config: {req}")
                if self.backend_server.update_eplb_config(req):
                    return {"status": "ok"}
                else:
                    return {"status": "set eplb config failed"}
            except Exception as e:
                return {"error": str(e)}

        register_backend_embedding_api(app, self.backend_server)
        return app
