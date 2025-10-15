import asyncio
import logging
import socket
import threading
from typing import Any, Dict, List, Optional, Union

import traceback
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

from rtp_llm.config.py_config_modules import PyEnvConfigs, StaticConfig
from rtp_llm.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.models.base_model import BaseModel
from rtp_llm.server.backend_server import BackendServer
from rtp_llm.server.misc import check_is_master, check_is_worker
from rtp_llm.server.worker_status import CacheStatus
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

        @app.post("/cache_status")
        def cache_status(req: Dict[str, Any]):
            check_shutdown()
            latest_cache_version: int = int(req.get("latest_cache_version", -1))
            cache_status_info = self.backend_server.get_cache_status(
                latest_cache_version
            )
            logging.info(
                f"cache_status info: {cache_status_info.available_kv_cache}, {cache_status_info.total_kv_cache}, {cache_status_info.block_size}, {cache_status_info.version}, {cache_status_info.cached_keys}"
            )
            cache_status = CacheStatus()
            cache_status.available_kv_cache = cache_status_info.available_kv_cache
            cache_status.total_kv_cache = cache_status_info.total_kv_cache
            cache_status.block_size = cache_status_info.block_size
            cache_status.version = cache_status_info.version
            cache_status.cached_keys = cache_status_info.cached_keys
            return ORJSONResponse(content=cache_status.model_dump(exclude_none=True))

        @app.post("/worker_status")
        def worker_status(req: Dict[str, Any]):
            check_shutdown()
            latest_finised_version: int = int(req.get("latest_finised_version", -1))
            worker_status = self.backend_server.get_worker_status(
                latest_finised_version
            )
            worker_status.server_port = worker_info.server_port
            worker_status.http_port = worker_info.http_port
            worker_status.grpc_port = worker_info.rpc_server_port
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
            return self.backend_server.update_scheduler_info(req)

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

        @app.post("/pause")
        async def pause():
            """
            Pauses the engine's execution.

            When called, this method sets the `pause_` flag to true. The engine's
            `step` method checks this flag and sleeps when it's true, effectively
            pausing execution. This is necessary for tasks like updating model weights
            or clearing GPU memory, which require the engine to be inactive. The `pause_`
            parameter is modified only by this interface, so it doesn't need to be
            thread-safe.
            """
            try:
                self.backend_server.pause()
            except Exception as e:
                # Using f-string for error details
                return {
                    "error": "Failed to pause generate engine",
                    "details": traceback.format_exc(),
                }

        @app.post("/internal_pause")
        async def internal_pause():
            try:
                self.backend_server.internal_pause()
            except Exception as e:
                # Using f-string for error details
                return {
                    "error": "Failed to pause generate engine",
                    "details": traceback.format_exc(),
                }

        @app.post("/restart")
        async def restart():
            """Restarts the engine's execution"""
            try:
                self.backend_server.restart()
            except Exception as e:
                # Using f-string for error details
                return {
                    "error": "Failed to restart generate engine",
                    "details": traceback.format_exc(),
                }

        @app.post("/internal_restart")
        async def internal_restart():
            try:
                self.backend_server.restart()
            except Exception as e:
                # Using f-string for error details
                return {
                    "error": "Failed to restart generate engine",
                    "details": traceback.format_exc(),
                }

        @app.post("/internal_update_weight")
        async def internal_update_weight(req: Dict[Any, Any]):
            """
            Internal endpoint to update model weights on a worker node.
            This endpoint is designed for internal communication within a distributed
            system, allowing a master node to direct worker nodes to update their
            model weights.
            Args:
                req (Dict[Any, Any]): A dictionary containing the weight update information,
                                     typically IPC handlers for the weights.
            Returns:
                dict: A dictionary with a "status" key set to "ok" if the update is successful.
                      On failure, it returns a dictionary with an "error" key
                      and a "details" key containing the traceback.
            """
            return self.backend_server.internal_update_weight(req)

        @app.post("/update_weight")
        async def update_weight(req: Dict[Any, Any]):
            """
            Updates the model's weights.
            This endpoint is used to update the model's weights while the server is running,
            which can be particularly useful in reinforcement learning (RL) procedures.
            It is crucial to ensure there are no active requests on the server when
            calling this interface to prevent potential inconsistencies or incorrect results.
            Args:
                req (Dict[Any, Any]): A dictionary where keys are model component names
                                     and values are IPC handlers for the corresponding weights.
            Returns:
                dict: A dictionary with a "status" key set to "ok" if the update is successful.
                      On failure, it returns a dictionary with an "error" key
                      and a "details" key containing the traceback.
            """
            return self.backend_server.update_weight(req)

        @app.post("/detach_physical_memory")
        async def detach_physical_memory():
            """
            Release physical GPU memory while retaining the virtual address space.
            This method is intended for engines that support virtual memory. It
            immediately unmaps and frees all **physical** backing memory without
            releasing the reserved **virtual** addresses.  If any requests are still
            in flight, the engine **must** wait for them to complete before
            performing the detach operation.
            Returns
            -------
            bool
                ``True``  – physical memory was successfully released.
                ``False`` – the engine does not support virtual memory **or** the
                detach operation failed.
            Notes
            -----
            After a successful detach, the virtual addresses remain valid but
            accessing them will raise a device page-fault until
            :meth:`attach_physical_memory` is called.
            """
            try:
                self.backend_server.detach_physical_memory()
                return {"status": "ok"}
            except Exception as e:
                # Using f-string for error details
                return {
                    "error": "Failed to detach physical memory",
                    "details": traceback.format_exc(),
                }

        @app.post("/internal_detach_physical_memory")
        async def internal_detach_physical_memory():
            try:
                self.backend_server.internal_detach_physical_memory()
                return {"status": "ok"}
            except Exception as e:
                # Using f-string for error details
                return {
                    "error": "Internal Failure",
                    "details": traceback.format_exc(),
                }

        @app.post("/attach_physical_memory")
        async def attach_physical_memory():
            """
            Re-attach / map physical memory to previously reserved virtual addresses.
            For every virtual address range that was **reserved but not mapped**
            (e.g., after :meth:`detach_physical_memory`), this method allocates
            physical GPU memory and binds it to those ranges.  Virtual addresses that
            already have physical backing are **not** re-allocated.
            Returns
            -------
            bool
                ``True``  – physical memory was successfully (re-)mapped.
                ``False`` – the engine lacks virtual-memory support **or** the
                mapping operation failed.
            """
            try:
                self.backend_server.attach_physical_memory()
                return {"status": "ok"}
            except Exception as e:
                # Using f-string for error details
                return {
                    "error": "Failed to attach physical memory",
                    "details": traceback.format_exc(),
                }

        @app.post("/internal_attach_physical_memory")
        async def internal_attach_physical_memory():
            try:
                self.backend_server.internal_attach_physical_memory()
                return {"status": "ok"}
            except Exception as e:
                # Using f-string for error details
                return {
                    "error": "Internal Failure",
                    "details": traceback.format_exc(),
                }

        register_backend_embedding_api(app, self.backend_server)
        return app
