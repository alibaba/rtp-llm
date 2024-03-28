import os
import sys
import json
import time
import logging
import logging.config
import uvicorn
import traceback
from typing import Generator, Union, Any, Dict, List, AsyncGenerator, Callable, Coroutine

from fastapi import FastAPI
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
from maga_transformer.config.exceptions import ExceptionType

# make buffer larger to avoid throw exception "RemoteProtocolError Receive buffer too long"
MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024

StreamObjectType = Union[Dict[str, Any], BaseModel]

class InferenceApp(object):
    def __init__(self):
        self.inference_server = InferenceServer()

    def start(self):
        self.inference_server.start()
        app = self.create_app()
        self.inference_server.wait_all_worker_ready()

        timeout_keep_alive = int(os.environ.get("TIMEOUT_KEEP_ALIVE", 5))
        uvicorn.run(app, host="0.0.0.0", port=g_worker_info.server_port, log_config=UVICORN_LOGGING_CONFIG,
                    timeout_keep_alive = timeout_keep_alive, h11_max_incomplete_event_size=MAX_INCOMPLETE_EVENT_SIZE)

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
            RunVar("_default_thread_limiter").set(CapacityLimiter(self.inference_server._controller.max_concurrency * 2))

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
            return "ok"

        @app.get("/")
        async def health():
            return {"status": "home"}

        @app.get("/worker_status")
        def worker_status():
            return {
                "available_concurrency": self.inference_server._controller.get_available_concurrency(),
                "alive": True,
            }

        # entry for worker RANK != 0
        @app.post("/inference_internal")
        async def inference_internal(req: Union[str,Dict[Any, Any]], raw_request: RawRequest):
            if g_parallel_info.is_master:
                return InferenceServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION,
                                    "gang cluster is None or role is master, should not access /inference_internal!")
            return await self.inference_server.inference(req, raw_request)

        # entry for worker RANK == 0
        @app.post("/")
        async def inference(req: Union[str,Dict[Any, Any]], raw_request: RawRequest):
            if not g_parallel_info.is_master:
                return InferenceServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION,
                                    "gang worker should not access this / api directly!")
            # compat for huggingface-pipeline request endpoint
            if self.inference_server.is_embedding:
                return await self.inference_server.embedding(req, raw_request)
            else:
                return await self.inference_server.inference(req, raw_request)

        # update for worker RANK != 0
        @app.post("/update_internal")
        def update_internal(version_info: VersionInfo):
            if g_parallel_info.is_master:
                return InferenceServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION,
                                    "gang cluster is None or role is master, should not access /update_internal!")
            return self.inference_server.update(version_info)

        # update for worker RANK == 0
        @app.post("/update")
        def update(version_info: VersionInfo):
            if not g_parallel_info.is_master:
                return InferenceServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION,
                                    "gang worker should not access /update api directly!")
            return self.inference_server.update(version_info)

        @app.get("/v1/models")
        async def list_models():
            assert (self.inference_server._openai_endpoint != None)
            return await self.inference_server._openai_endpoint.list_models()

        # entry for worker RANK == 0
        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        async def chat_completion(request: ChatCompletionRequest, raw_request: RawRequest):
            if not g_parallel_info.is_master:
                return InferenceServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION,
                                    "gang worker should not access this completions api directly!")
            return await self.inference_server.chat_completion(request, raw_request)

        # entry for worker RANK == 0
        @app.post("/tokenizer/encode")
        async def encode(req: Union[str,Dict[Any, Any]]):
            if not g_parallel_info.is_master:
                return InferenceServer.format_exception(ExceptionType.UNSUPPORTED_OPERATION,
                                    "gang worker should not access this completions api directly!")
            return self.inference_server.tokenizer_encode(req)        

        register_embedding_api(app, self.inference_server)
        return app
