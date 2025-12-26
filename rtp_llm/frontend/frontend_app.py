import asyncio
import json
import logging
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import Body, FastAPI, HTTPException
from fastapi import Request
from fastapi import Request as RawRequest
from fastapi import status
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from typing_extensions import override
from uvicorn import Config, Server
from uvicorn.loops.auto import auto_loop_setup

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import (
    update_stop_words_from_env,
    update_tokenizer_special_tokens,
)
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.uvicorn_config import get_uvicorn_logging_config
from rtp_llm.distribute.distributed_server import (
    WorldInfo,
    get_dp_addrs_from_world_info,
    get_world_info,
)
from rtp_llm.distribute.worker_info import WorkerInfo, g_worker_info
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint
from rtp_llm.embedding.embedding_type import TYPE_STR, EmbeddingType
from rtp_llm.frontend.frontend_worker import FrontendWorker, TokenizerEncodeResponse
from rtp_llm.frontend.generation.orchestrator import GenerationOrchestrator
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.model_factory import ModelFactory
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import ParallelismConfig, SpecialTokens, TaskType, VitSeparation
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.misc import format_exception
from rtp_llm.utils.concurrency_controller import get_global_controller
from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper
from rtp_llm.utils.util import AtomicCounter, async_request_server
from rtp_llm.utils.version_info import VersionInfo

# make buffer larger to avoid throw exception "RemoteProtocolError Receive buffer too long"
MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024

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


class FrontendApp(object):
    def __init__(
        self,
        py_env_configs: PyEnvConfigs,
        separated_frontend: bool = False,
    ):
        self.py_env_configs = py_env_configs
        self.rank_id = py_env_configs.server_config.rank_id
        self.server_id = py_env_configs.server_config.frontend_server_id
        self.server_config = py_env_configs.server_config

        self.separated_frontend = separated_frontend
        self.grpc_client = GrpcClientWrapper(g_worker_info.rpc_server_port)
        g_worker_info.server_port = WorkerInfo.server_port_offset(
            self.rank_id, g_worker_info.server_port
        )
        g_worker_info.backend_server_port = WorkerInfo.server_port_offset(
            self.rank_id, g_worker_info.backend_server_port
        )
        logging.info(
            f"rank_id = {self.rank_id}, "
            f"server_port = {g_worker_info.server_port}, backend_server_port = {g_worker_info.backend_server_port}, server_id = {self.server_id}"
        )
        self.is_embedding = False

        self._frontend_worker = None
        self._openai_endpoint = None
        self._embedding_endpoint = None
        self._tokenizer = None
        self._backend_rpc_server_visitor = None
        self._global_controller = get_global_controller()

    def start(self):
        if (
            self.py_env_configs.profiling_debug_logging_config.debug_start_fake_process
            == 1
        ):
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake server")
            return

        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=self.py_env_configs.kv_cache_config,
            profiling_debug_logging_config=self.py_env_configs.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
        )

        special_tokens = SpecialTokens()
        if self.py_env_configs.generate_env_config:
            update_stop_words_from_env(
                special_tokens, self.py_env_configs.generate_env_config
            )

        # Build shared tokenizer and backend visitor once
        self._tokenizer = TokenizerFactory.create(
            model_config.ckpt_path,
            model_config.tokenizer_path,
            model_config.model_type,
        )
        update_tokenizer_special_tokens(special_tokens, self._tokenizer)

        engine_config = EngineConfig.create(self.py_env_configs)
        world_info = get_world_info(
            server_config=self.py_env_configs.server_config,
            distribute_config=self.py_env_configs.distribute_config,
        )
        addresses = get_dp_addrs_from_world_info(
            world_info=world_info,
            parallelism_config=engine_config.parallelism_config,
        )
        vit_separation = None
        if self.py_env_configs.vit_config:
            vit_separation = self.py_env_configs.vit_config.vit_separation

        self._backend_rpc_server_visitor = BackendRPCServerVisitor(
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.attn_config.tokens_per_block,
            pd_sep_config=engine_config.pd_sep_config,
            addresses=addresses,
            sp_config=self.py_env_configs.sp_config,
            vit_separation=vit_separation,
        )

        orchestrator = GenerationOrchestrator(
            special_tokens=special_tokens,
            pd_sep_config=engine_config.pd_sep_config,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.attn_config.tokens_per_block,
            tokenizer=self._tokenizer,
            sp_config=self.py_env_configs.sp_config,
            mm_related_params=None,
            vit_separation=vit_separation,
            backend_rpc_server_visitor=self._backend_rpc_server_visitor,
        )

        self._frontend_worker = FrontendWorker(
            tokenizer=self._tokenizer,
            backend_rpc_server_visitor=self._backend_rpc_server_visitor,
            generate_env_config=self.py_env_configs.generate_env_config,
            orchestrator=orchestrator,
            model_config=model_config,
            global_controller=self._global_controller,
            rank_id=int(self.rank_id),
            server_id=int(self.server_id),
        )

        # Only initialize OpenaiEndpoint for LANGUAGE_MODEL task type
        if model_config.task_type == TaskType.LANGUAGE_MODEL:
            # Update model_config with the latest values
            model_config.special_tokens = special_tokens
            model_config.generate_env_config = self.py_env_configs.generate_env_config
            model_config.render_config = self.py_env_configs.render_config
            model_config.model_name = self.py_env_configs.model_args.model_type
            model_config.template_type = None

            self._openai_endpoint = OpenaiEndpoint(
                model_config=model_config,
                misc_config=self.py_env_configs.misc_config,
                vit_config=self.py_env_configs.vit_config,
                tokenizer=self._tokenizer,
                backend_rpc_server_visitor=self._backend_rpc_server_visitor,
                global_controller=self._global_controller,
                rank_id=int(self.rank_id),
                server_id=int(self.server_id),
            )
        else:
            self._embedding_endpoint = EmbeddingEndpoint(
                model_config=model_config,
                grpc_config=self.py_env_configs.grpc_config,
                tokenizer=self._tokenizer,
                global_controller=self._global_controller,
                rank_id=int(self.rank_id),
                server_id=int(self.server_id),
            )
            self.is_embedding = True
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
        timeout_keep_alive = self.server_config.timeout_keep_alive

        config = Config(
            app,
            fd=fd,
            loop=loop,
            log_config=get_uvicorn_logging_config(),
            timeout_keep_alive=timeout_keep_alive,
            h11_max_incomplete_event_size=MAX_INCOMPLETE_EVENT_SIZE,
        )

        try:
            server = GracefulShutdownServer(config)
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
                CapacityLimiter(self._global_controller.max_concurrency * 2)
            )

        async def check_all_health():
            assert self._frontend_worker is not None
            if not (
                self._backend_rpc_server_visitor.is_backend_service_ready(refresh=False)
            ):
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
            response = await self.grpc_client.post_request("cache_status", query_params)
            if "error" not in response:
                response["frontend_available_concurrency"] = (
                    self._global_controller.get_available_concurrency()
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
            response = await self.grpc_client.post_request(
                "worker_status", query_params
            )
            if "error" not in response:
                response["frontend_available_concurrency"] = (
                    self._global_controller.get_available_concurrency()
                )
            return response

        @app.get("/v1/models")
        async def list_models():
            assert self._openai_endpoint != None
            return await self._openai_endpoint.list_models()

        # request format: {"log_level": "DEBUG"}, {"log_level": "info"}
        @app.post("/set_log_level")
        async def set_log_level(req: Union[str, Dict[Any, Any]]):
            result = await self.grpc_client.post_request("set_log_level", req)
            return result

        # request format: {"mode": "NONE", "update_time": 5000}
        @app.post("/update_eplb_config")
        async def update_eplb_config(req: Union[str, Dict[Any, Any]]):
            result = await self.grpc_client.post_request("update_eplb_config", req)
            return result

        @app.post("/")
        async def inference(req: Union[str, Dict[Any, Any]], raw_request: RawRequest):
            # compat for huggingface-pipeline request endpoint
            global active_requests
            active_requests.increment()
            try:
                if self.is_embedding:
                    assert self._embedding_endpoint is not None
                    return await self._embedding_endpoint.handle_request(
                        req, raw_request
                    )
                else:
                    assert self._frontend_worker is not None
                    return await self._frontend_worker.handle_request(req, raw_request)
            except Exception as e:
                exception_json = format_exception(e)
                return ORJSONResponse(exception_json, status_code=500)

            finally:
                active_requests.decrement()

        @app.post("/chat/completions")
        @app.post("/v1/chat/completions")
        async def chat_completion(
            request: ChatCompletionRequest, raw_request: RawRequest
        ):
            assert self._openai_endpoint is not None
            global active_requests
            active_requests.increment()
            try:
                return await self._openai_endpoint.handle_request(request, raw_request)
            except Exception as e:
                exception_json = format_exception(e)
                return ORJSONResponse(exception_json, status_code=500)
            finally:
                active_requests.decrement()

        @app.post("/update_scheduler_info")
        async def update_scheduler_info(req: Union[str, Dict[Any, Any]]):
            result = await self.grpc_client.post_request("update_scheduler_info", req)
            return result

        @app.post("/chat/render")
        @app.post("/v1/chat/render")
        async def chat_render(request: ChatCompletionRequest, raw_request: RawRequest):
            assert self._openai_endpoint != None
            global active_requests
            active_requests.increment()
            try:
                return self._openai_endpoint.chat_render(request)
            except Exception as e:
                return ORJSONResponse(format_exception(e), status_code=500)
            finally:
                active_requests.decrement()

        # example {"prompt": "abcde"}
        @app.post("/tokenizer/encode")
        async def tokenizer_encode(req: Union[str, Dict[Any, Any]]):
            try:
                if isinstance(req, str):
                    req = json.loads(req)
                assert isinstance(req, dict)
                prompt = req.pop("prompt")
                if req.get("return_offsets_mapping", None) == True:
                    mapping = self._tokenizer(
                        prompt, return_offsets_mapping=True, return_attention_mask=False
                    )
                    response = TokenizerEncodeResponse(
                        offset_mapping=mapping["offset_mapping"],
                        token_ids=mapping["input_ids"],
                    )
                else:
                    token_ids = self._tokenizer.encode(prompt)
                    tokens = self._tokenizer.decode([int(id) for id in token_ids])
                    response = TokenizerEncodeResponse(
                        token_ids=token_ids, tokens=tokens
                    )
                return ORJSONResponse(content=response.model_dump(exclude_none=True))
            except Exception as e:
                return ORJSONResponse(format_exception(e), status_code=500)

        # example {"prompt": "abcde"}
        # example openai_request
        @app.post("/tokenize")
        async def encode(req: Union[str, Dict[Any, Any]]):
            try:
                if isinstance(req, str):
                    req = json.loads(req)
                if ChatCompletionRequest.is_openai_request(req):
                    chat_request = ChatCompletionRequest(**req)
                    token_ids = self._openai_endpoint.render_chat(
                        chat_request
                    ).input_ids
                else:
                    prompt = req.pop("prompt")
                    token_ids = self._tokenizer.encode(prompt)
                return ORJSONResponse({"token_ids": token_ids})
            except Exception as e:
                return ORJSONResponse(format_exception(e), status_code=500)

        @app.post("/update_weight")
        async def update_weight(req: Union[str, Dict[Any, Any]]):
            return await async_request_server(
                "post", g_worker_info.backend_server_port, "update_weight", req
            )

        if self.is_embedding:
            # embedding
            @app.post("/v1/embeddings/similarity")
            @app.post("/v1/reranker")
            @app.post("/v1/classifier")
            @app.post("/v1/embeddings")
            async def embedding(request: Dict[str, Any], raw_request: RawRequest):
                return await self._embedding_endpoint.handle_request(
                    request, raw_request
                )

            @app.post("/v1/embeddings/dense")
            async def embedding_dense(request: Dict[str, Any], raw_request: RawRequest):
                request[TYPE_STR] = EmbeddingType.DENSE
                return await self._embedding_endpoint.handle_request(
                    request, raw_request
                )

            @app.post("/v1/embeddings/sparse")
            async def embedding_sparse(
                request: Dict[str, Any], raw_request: RawRequest
            ):
                request[TYPE_STR] = EmbeddingType.SPARSE
                return await self._embedding_endpoint.handle_request(
                    request, raw_request
                )

            @app.post("/v1/embeddings/colbert")
            async def embedding_colbert(
                request: Dict[str, Any], raw_request: RawRequest
            ):
                request[TYPE_STR] = EmbeddingType.COLBERT
                return await self._embedding_endpoint.handle_request(
                    request, raw_request
                )

        return app
