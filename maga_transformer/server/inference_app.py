import os
import sys
import json
import time
import logging
import logging.config
from typing_extensions import override
import uvicorn
from uvicorn import Server, Config
import asyncio
import socket
from typing import Union, Any, Dict, Optional, List

from fastapi import FastAPI, status, HTTPException
from fastapi import Request as RawRequest
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter

from maga_transformer.distribute.worker_info import g_worker_info, g_parallel_info
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.openai.api_datatype import ChatCompletionRequest, ChatCompletionStreamResponse
from maga_transformer.embedding.embedding_app import register_embedding_api
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from maga_transformer.models.base_model import BaseModel
from maga_transformer.server.inference_server import InferenceServer
from maga_transformer.server.misc import check_is_master, check_is_worker
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.utils.util import AtomicCounter

# make buffer larger to avoid throw exception "RemoteProtocolError Receive buffer too long"
MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024

StreamObjectType = Union[Dict[str, Any], BaseModel]

active_requests = AtomicCounter()
server_shutdown = False

class GracefulShutdownServer(Server):
    @override
    async def shutdown(self, sockets: Optional[List[socket.socket]] = None) -> None:
        global server_shutdown
        server_shutdown = True
        global active_requests
        while active_requests.get() > 0:
            logging.info(f"wait {active_requests.get()} requests finish for 1s")
            await asyncio.sleep(1)
        await super().shutdown(sockets)

class InferenceApp(object):
    def __init__(self):
        self.inference_server = InferenceServer()

    def start(self):
        self.inference_server.start()
        app = self.create_app()
        self.inference_server.wait_all_worker_ready()

        timeout_keep_alive = int(os.environ.get("TIMEOUT_KEEP_ALIVE", 5))
        config = Config(
            app,
            host="0.0.0.0",
            port=g_worker_info.server_port,
            log_config=UVICORN_LOGGING_CONFIG,
            timeout_keep_alive=timeout_keep_alive,
            h11_max_incomplete_event_size=MAX_INCOMPLETE_EVENT_SIZE,
        )
        server = GracefulShutdownServer(config)
        server.run()
        # uvicorn.run(app, host="0.0.0.0", port=g_worker_info.server_port, log_config=UVICORN_LOGGING_CONFIG,
        #             timeout_keep_alive = timeout_keep_alive, h11_max_incomplete_event_size=MAX_INCOMPLETE_EVENT_SIZE)

    def create_app(self):
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=['*'],
                allow_credentials=True,
                allow_methods=['*'],
                allow_headers=['*']
            )
        ]
        app = FastAPI(middleware=middleware)

        def check_shutdown():
            global server_shutdown
            if server_shutdown:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="this server has been shutdown"
                )

        @app.on_event("startup")
        async def startup():
            RunVar("_default_thread_limiter").set(CapacityLimiter(self.inference_server._controller.max_concurrency * 2))

        @app.middleware("http")
        async def count_requests(request: RawRequest, call_next):
            global active_requests
            active_requests.increment()
            try:
                response = await call_next(request)
                return response
            finally:
                active_requests.decrement()

        @app.get("/health")
        @app.post("/health")
        @app.get("/GraphService/cm2_status")
        @app.post("/GraphService/cm2_status")
        @app.get("/SearchService/cm2_status")
        @app.post("/SearchService/cm2_status")
        @app.get("/status")
        @app.post("/status")
        @app.post("/health_check")
        async def health():
            check_shutdown()
            return "ok"

        @app.get("/")
        async def health():
            check_shutdown()
            return {"status": "home"}

        @app.get("/worker_status")
        def worker_status():
            check_shutdown()
            info = self.inference_server.get_kv_cache_info()
            return {
                "available_concurrency": self.inference_server._controller.get_available_concurrency(),
                "available_kv_cache": info.available_kv_cache,
                "total_kv_cache": info.total_kv_cache,
                "alive": True,
            }

        # entry for worker RANK != 0
        @app.post("/inference_internal")
        @check_is_worker()
        async def inference_internal(req: Union[str,Dict[Any, Any]], raw_request: RawRequest):
            return await self.inference_server.inference(req, raw_request)

        # entry for worker RANK == 0
        @app.post("/")
        @check_is_master()
        async def inference(req: Union[str,Dict[Any, Any]], raw_request: RawRequest):
            # compat for huggingface-pipeline request endpoint
            if self.inference_server.is_embedding:
                start_time = time.time()
                res = await self.inference_server.embedding(req, raw_request)                
                end_time = time.time()
                # print("total: ", end_time - start_time)
                return res
            else:
                return await self.inference_server.inference(req, raw_request)

        # update for worker RANK != 0
        @app.post("/update_internal")
        @check_is_worker()
        def update_internal(version_info: VersionInfo):
            return self.inference_server.update(version_info)

        # update for worker RANK == 0
        @app.post("/update")
        @check_is_master()
        def update(version_info: VersionInfo):
            return self.inference_server.update(version_info)

        @app.get("/v1/models")
        async def list_models():
            assert (self.inference_server._openai_endpoint != None)
            return await self.inference_server._openai_endpoint.list_models()

        # entry for worker RANK == 0
        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        @check_is_master()
        async def chat_completion(request: ChatCompletionRequest, raw_request: RawRequest):
            return await self.inference_server.chat_completion(request, raw_request)

        # entry for worker RANK == 0
        @app.post("/chat/render")
        @app.post("/v1/chat/render")
        @check_is_master()
        async def chat_render(request: ChatCompletionRequest, raw_request: RawRequest):
            return await self.inference_server.chat_render(request, raw_request)

        # entry for worker RANK == 0
        @app.post("/tokenizer/encode")
        @check_is_master()
        async def encode(req: Union[str,Dict[Any, Any]]):
            return self.inference_server.tokenizer_encode(req)
        
        @app.post("/set_debug_log")
        async def set_debug_log(req: Union[str,Dict[Any, Any]]):
            try:
                self.inference_server.set_debug_log(req)
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}
        
        @app.post("/set_debug_print")
        async def set_debug_print(req: Union[str,Dict[Any, Any]]):
            try:
                self.inference_server.set_debug_print(req)
                return {"status": "ok"}
            except Exception as e:
                return {"error": str(e)}

        register_embedding_api(app, self.inference_server)
        return app