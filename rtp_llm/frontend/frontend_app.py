import asyncio
import logging
import socket
import threading
from typing import Any, Dict, List, Optional, Union

from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import Body, FastAPI, HTTPException
from fastapi import Request
from fastapi import Request as RawRequest
from fastapi import status
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import override
from uvicorn import Config, Server
from uvicorn.loops.auto import auto_loop_setup

from rtp_llm.config.py_config_modules import PyEnvConfigs, StaticConfig
from rtp_llm.config.uvicorn_config import UVICORN_LOGGING_CONFIG
from rtp_llm.distribute.worker_info import WorkerInfo, g_worker_info
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.utils.util import AtomicCounter, async_request_server
from rtp_llm.utils.version_info import VersionInfo

# make buffer larger to avoid throw exception "RemoteProtocolError Receive buffer too long"
MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024

active_requests = AtomicCounter()
server_shutdown = False


class GracefulShutdownServer(Server):
    def set_server(self, frontend_server: FrontendServer):
        self.frontend_server = frontend_server

    @override
    async def shutdown(self, sockets: Optional[List[socket.socket]] = None) -> None:
        global server_shutdown
        server_shutdown = True
        global active_requests
        while active_requests.get() > 0:
            logging.info(f"wait {active_requests.get()} requests finish for 1s")
            await asyncio.sleep(1)
        await super().shutdown(sockets)


class FrontendApp(object):
    def __init__(
        self,
        py_env_configs: PyEnvConfigs = StaticConfig,
        separated_frontend: bool = False,
    ):
        self.py_env_configs = py_env_configs
        self.frontend_server = FrontendServer(
            separated_frontend,
            py_env_configs.server_config.rank_id,
            py_env_configs.server_config.frontend_server_id,
        )
        self.separated_frontend = separated_frontend
        g_worker_info.server_port = WorkerInfo.server_port_offset(
            self.py_env_configs.server_config.rank_id, g_worker_info.server_port
        )
        g_worker_info.backend_server_port = WorkerInfo.server_port_offset(
            self.py_env_configs.server_config.rank_id, g_worker_info.backend_server_port
        )
        logging.info(
            f"rank_id = {self.py_env_configs.server_config.rank_id}, "
            f"server_port = {g_worker_info.server_port}, backend_server_port = {g_worker_info.backend_server_port}, frontend_server_id = {py_env_configs.server_config.frontend_server_id}"
        )

    def start(self):
        self.frontend_server.start()
        app = self.create_app()

        loop = "auto"
        if threading.current_thread() != threading.main_thread():
            # NOTE: asyncio
            loop = "none"
            auto_loop_setup()
            asyncio.set_event_loop(asyncio.new_event_loop())

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(("0.0.0.0", g_worker_info.server_port))
        sock.listen()
        fd = sock.fileno()
        timeout_keep_alive = self.py_env_configs.server_config.timeout_keep_alive

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
            raise e

    def create_app(self):
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

        @app.on_event("startup")
        async def startup():
            RunVar("_default_thread_limiter").set(
                CapacityLimiter(
                    self.frontend_server._global_controller.max_concurrency * 2
                )
            )

        async def check_all_health():
            if not self.frontend_server.check_health():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="inference service is not ready",
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
            if self.separated_frontend:
                await check_all_health()
                return "ok"
            return await async_request_server(
                "post", g_worker_info.backend_server_port, "health_check", {}
            )

        @app.get("/")
        async def health():
            if self.separated_frontend:
                await check_all_health()
                return {"status": "home"}
            return await async_request_server(
                "get", g_worker_info.backend_server_port, "", {}
            )

        @app.get("/cache_status")
        @app.post("/cache_status")
        @app.get("/rtp_llm/cache_status")
        @app.post("/rtp_llm/cache_status")
        async def cache_status(
            request: Request, data: Optional[Dict[Any, Any]] = Body(None)
        ):
            query_params = (
                dict(request.query_params) if request.method == "GET" else (data or {})
            )

            logging.info(f"cache_status request {data}")
            response = await async_request_server(
                "post", g_worker_info.backend_server_port, "cache_status", query_params
            )
            if "error" not in response:
                response["frontend_available_concurrency"] = (
                    self.frontend_server._global_controller.get_available_concurrency()
                )
            logging.info(f"cache_status response {response}")
            return response

        @app.get("/worker_status")
        @app.post("/worker_status")
        @app.get("/rtp_llm/worker_status")
        @app.post("/rtp_llm/worker_status")
        async def worker_status(
            request: Request, data: Optional[Dict[Any, Any]] = Body(None)
        ):
            query_params = (
                dict(request.query_params) if request.method == "GET" else (data or {})
            )
            response = await async_request_server(
                "post", g_worker_info.backend_server_port, "worker_status", query_params
            )
            if "error" not in response:
                response["frontend_available_concurrency"] = (
                    self.frontend_server._global_controller.get_available_concurrency()
                )
            return response

        # example : {"peft_info": {"lora_info": {"lora_0": "/lora/llama-lora-test/""}}}
        @app.post("/update")
        async def update(version_info: VersionInfo):
            return await async_request_server(
                "post",
                g_worker_info.backend_server_port,
                "update",
                version_info.model_dump(),
            )

        @app.get("/v1/models")
        async def list_models():
            assert self.frontend_server._openai_endpoint != None
            return await self.frontend_server._openai_endpoint.list_models()

        # request format: {"log_level": "DEBUG"}, {"log_level": "info"}
        @app.post("/set_log_level")
        async def set_log_level(req: Union[str, Dict[Any, Any]]):
            return await async_request_server(
                "post", g_worker_info.backend_server_port, "set_log_level", req
            )

        # request format: {"mode": "NONE", "update_time": 5000}
        @app.post("/update_eplb_config")
        async def update_eplb_config(req: Dict[Any, Any]):
            return await async_request_server(
                "post", g_worker_info.backend_server_port, "update_eplb_config", req
            )

        @app.post("/")
        async def inference(req: Union[str, Dict[Any, Any]], raw_request: RawRequest):
            # compat for huggingface-pipeline request endpoint
            global active_requests
            active_requests.increment()
            try:
                if self.frontend_server.is_embedding:
                    return await async_request_server(
                        "post", g_worker_info.backend_server_port, "v1/embeddings", req
                    )
                else:
                    return await self.frontend_server.inference(req, raw_request)
            finally:
                active_requests.decrement()

        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        async def chat_completion(
            request: ChatCompletionRequest, raw_request: RawRequest
        ):
            global active_requests
            active_requests.increment()
            try:
                return await self.frontend_server.chat_completion(request, raw_request)
            finally:
                active_requests.decrement()

        @app.post("/update_scheduler_info")
        async def update_scheduler_info(req: Union[str, Dict[Any, Any]]):
            return await async_request_server(
                "post", g_worker_info.backend_server_port, "update_scheduler_info", req
            )

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
        async def tokenizer_encode(req: Union[str, Dict[Any, Any]]):
            return self.frontend_server.tokenizer_encode(req)

        # example {"prompt": "abcde"}
        # example openai_request
        @app.post("/tokenize")
        async def encode(req: Union[str, Dict[Any, Any]]):
            return self.frontend_server.tokenize(req)

        @app.post("/update_weight")
        async def update_weight(req: Union[str, Dict[Any, Any]]):
            return await async_request_server(
                "post", g_worker_info.backend_server_port, "update_weight", req
            )

        if self.frontend_server.is_embedding:
            # embedding
            @app.post("/v1/embeddings")
            async def embedding(request: Dict[str, Any], raw_request: RawRequest):
                return await async_request_server(
                    "post", g_worker_info.backend_server_port, "v1/embeddings", request
                )

            @app.post("/v1/embeddings/dense")
            async def embedding_dense(request: Dict[str, Any], raw_request: RawRequest):
                return await async_request_server(
                    "post",
                    g_worker_info.backend_server_port,
                    "v1/embeddings/dense",
                    request,
                )

            @app.post("/v1/embeddings/sparse")
            async def embedding_sparse(
                request: Dict[str, Any], raw_request: RawRequest
            ):
                return await async_request_server(
                    "post",
                    g_worker_info.backend_server_port,
                    "v1/embeddings/sparse",
                    request,
                )

            @app.post("/v1/embeddings/colbert")
            async def embedding_colbert(
                request: Dict[str, Any], raw_request: RawRequest
            ):
                return await async_request_server(
                    "post",
                    g_worker_info.backend_server_port,
                    "v1/embeddings/colbert",
                    request,
                )

            @app.post("/v1/embeddings/similarity")
            async def similarity(request: Dict[str, Any], raw_request: RawRequest):
                return await async_request_server(
                    "post",
                    g_worker_info.backend_server_port,
                    "v1/embeddings/similarity",
                    request,
                )

            @app.post("/v1/classifier")
            async def classifier(request: Dict[str, Any], raw_request: RawRequest):
                return await async_request_server(
                    "post", g_worker_info.backend_server_port, "v1/classifier", request
                )

            @app.post("/v1/reranker")
            async def reranker(request: Dict[str, Any], raw_request: RawRequest):
                return await async_request_server(
                    "post", g_worker_info.backend_server_port, "v1/reranker", request
                )

        return app
