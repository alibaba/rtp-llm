import os
import sys
import json
import time
import logging
import logging.config
from functools import cached_property
from typing_extensions import override
import uvicorn
from uvicorn import Server, Config
import asyncio
import socket
import threading
import requests
from typing import Union, Any, Dict, Optional, List

from fastapi import FastAPI, status, HTTPException
from fastapi import Request as RawRequest
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter
from uvicorn.loops.auto import auto_loop_setup

from maga_transformer.distribute.worker_info import g_worker_info
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.openai.api_datatype import ChatCompletionRequest, ChatCompletionStreamResponse
from maga_transformer.embedding.frontend_embedding_app import register_frontend_embedding_api
from maga_transformer.utils.version_info import VersionInfo
from maga_transformer.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from maga_transformer.models.base_model import BaseModel
from maga_transformer.server.frontend_server import FrontendServer
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.utils.util import AtomicCounter, request_server
from maga_transformer.utils.concurrency_controller import ConcurrencyController

# make buffer larger to avoid throw exception "RemoteProtocolError Receive buffer too long"
MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024

StreamObjectType = Union[Dict[str, Any], BaseModel]

active_requests = AtomicCounter()
server_shutdown = False

class GracefulShutdownServer(Server):
    def set_server(self, frontend_server):
        self.frontend_server = frontend_server

    @override
    async def shutdown(self, sockets: Optional[List[socket.socket]] = None) -> None:
        global server_shutdown
        server_shutdown = True
        global active_requests
        while active_requests.get() > 0:
            logging.info(f"wait {active_requests.get()} requests finish for 1s")
            await asyncio.sleep(1)
        self.frontend_server.stop()
        await super().shutdown(sockets)

class FrontendApp(object):
    def __init__(self):
        self.frontend_server = FrontendServer()

    def start(self):
        self.frontend_server.start()        
        app = self.create_app()

        loop = "auto"
        if (threading.current_thread() != threading.main_thread()):
            # NOTE: asyncio
            loop = "none"
            auto_loop_setup()
            asyncio.set_event_loop(asyncio.new_event_loop())

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(('0.0.0.0', g_worker_info.server_port))
        sock.listen()
        fd = sock.fileno()
        timeout_keep_alive = int(os.environ.get("TIMEOUT_KEEP_ALIVE", 5))

        config = Config(
            app,
            fd=fd,  
            loop=loop,
            log_config=UVICORN_LOGGING_CONFIG,
            timeout_keep_alive=timeout_keep_alive,
            h11_max_incomplete_event_size=MAX_INCOMPLETE_EVENT_SIZE,
        )

        try:
            server = GracefulShutdownServer(config)
            server.set_server(self.frontend_server)
            server.run()
        except BaseException as e:
            self.frontend_server.stop()
            raise e

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

        @app.on_event("startup")
        async def startup():
            RunVar("_default_thread_limiter").set(CapacityLimiter(self.frontend_server._global_controller.max_concurrency * 2))

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
            return request_server("post", g_worker_info.backend_server_port, "health_check", {})

        @app.get("/")
        async def health():
            return request_server("get", g_worker_info.backend_server_port, "", {})

        @app.get("/worker_status")
        def worker_status():
            response = request_server("get", g_worker_info.backend_server_port, "worker_status", {})
            if "error" not in response:
                response["frontend_available_concurrency"] = self.frontend_server._global_controller.get_available_concurrency()
            return response

        # example : {"peft_info": {"lora_info": {"lora_0": "/lora/llama-lora-test/""}}}
        @app.post("/update")
        def update(version_info: VersionInfo):
            return request_server("post", g_worker_info.backend_server_port, "update", version_info.model_dump())

        @app.get("/v1/models")
        async def list_models():
            assert (self.frontend_server._openai_endpoint != None)
            return await self.frontend_server._openai_endpoint.list_models()

        # request format: {"log_level": "DEBUG"}, {"log_level": "info"}
        @app.post("/set_log_level")
        async def set_log_level(req: Union[str, Dict[Any, Any]]):
            return request_server("post", g_worker_info.backend_server_port, "set_log_level", req)

        @app.post("/")
        async def inference(req: Union[str,Dict[Any, Any]], raw_request: RawRequest):
            # compat for huggingface-pipeline request endpoint
            global active_requests
            active_requests.increment()
            try:
                if self.frontend_server.is_embedding:
                    return request_server("post", g_worker_info.backend_server_port, "v1/embeddings", req)
                else:
                    return await self.frontend_server.inference(req, raw_request)
            finally:
                active_requests.decrement()

        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        async def chat_completion(request: ChatCompletionRequest, raw_request: RawRequest):
            global active_requests
            active_requests.increment()
            try:
                return await self.frontend_server.chat_completion(request, raw_request)
            finally:
                active_requests.decrement()

        @app.post("/chat/render")
        @app.post("/v1/chat/render")
        async def chat_render(request: ChatCompletionRequest, raw_request: RawRequest):
            global active_requests
            active_requests.increment()
            try:
                return await self.frontend_server.chat_render(request, raw_request)
            finally:
                active_requests.decrement()

        # example {"prompt": "abcde"}
        @app.post("/tokenizer/encode")
        async def encode(req: Union[str,Dict[Any, Any]]):
            return self.frontend_server.tokenizer_encode(req)


        # example {"prompt": "abcde"}
        # example openai_request
        @app.post("/tokenize")
        async def encode(req: Union[str,Dict[Any, Any]]):
            return self.frontend_server.tokenize(req)

        register_frontend_embedding_api(app, g_worker_info.backend_server_port)
        return app