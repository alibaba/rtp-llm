import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional, Union

from fastapi import HTTPException
from fastapi import Request
from fastapi import Request as RawRequest
from fastapi import status
from fastapi.responses import ORJSONResponse

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import (
    update_stop_words_from_env,
    update_tokenizer_special_tokens,
)
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint
from rtp_llm.frontend.frontend_worker import FrontendWorker, TokenizerEncodeResponse
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.metrics import AccMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import (
    OpenaiEndpoint,
    prepare_model_config_for_openai,
)
from rtp_llm.ops import SpecialTokens, TaskType
from rtp_llm.server.misc import format_exception
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper
from rtp_llm.utils.util import AtomicCounter


class FrontendServer(object):
    def __init__(
        self,
        rank_id: int,
        server_id: int,
        py_env_configs=None,
    ):
        self.py_env_configs = py_env_configs
        self.server_config = py_env_configs.server_config
        self.active_requests = AtomicCounter()
        self.access_logger = AccessLogger(
            get_log_path(),
            py_env_configs.profiling_debug_logging_config.log_file_backup_count,
            rank_id,
            server_id,
        )
        self.frontend_worker = None
        self.openai_endpoint = None
        self.embedding_endpoint = None
        self.tokenizer = None
        self.is_embedding = False
        self.thread_lock = threading.Lock()
        self.global_controller = get_global_controller()
        self.rank_id = str(rank_id)
        self.server_id = str(server_id)
        self.grpc_client = None
        kmonitor.init()

    def start(self):
        if (
            self.py_env_configs.profiling_debug_logging_config.debug_start_fake_process
            == 1
        ):
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake server")
            self.frontend_worker = None
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
        if model_config.task_type != TaskType.LANGUAGE_MODEL:
            self.is_embedding = True

        self.grpc_client = GrpcClientWrapper(
            self.server_config.rpc_server_port,
            self.server_config.embedding_rpc_server_port,
            self.is_embedding,
        )
        # Create tokenizer in FrontendServer and pass to worker/endpoints
        self.tokenizer = TokenizerFactory.create(
            model_config.ckpt_path,
            model_config.tokenizer_path,
            model_config.model_type,
        )
        update_tokenizer_special_tokens(special_tokens, self.tokenizer)
        # Only initialize OpenaiEndpoint for LANGUAGE_MODEL task type
        if model_config.task_type == TaskType.LANGUAGE_MODEL:
            # Create FrontendWorker with special_tokens and config (and endpoint params for LANGUAGE_MODEL)
            self.frontend_worker = FrontendWorker(
                self.py_env_configs,
                model_config,
                special_tokens,
                global_controller=self.global_controller,
                access_logger=self.access_logger,
                tokenizer=self.tokenizer,
                rank_id=self.rank_id,
                server_id=self.server_id,
                active_requests=self.active_requests,
            )
            prepare_model_config_for_openai(
                model_config,
                special_tokens,
                self.py_env_configs.generate_env_config,
                self.py_env_configs.render_config,
                self.py_env_configs.model_args.model_type,
                template_type=None,
            )
            self.openai_endpoint = OpenaiEndpoint(
                model_config=model_config,
                misc_config=self.py_env_configs.misc_config,
                vit_config=self.py_env_configs.vit_config,
                tokenizer=self.tokenizer,
                backend_rpc_server_visitor=self.frontend_worker.backend_rpc_server_visitor,
                global_controller=self.global_controller,
                access_logger=self.access_logger,
                rank_id=self.rank_id,
                server_id=self.server_id,
                active_requests=self.active_requests,
            )
        else:
            self.embedding_endpoint = EmbeddingEndpoint(
                model_config=model_config,
                grpc_config=self.py_env_configs.grpc_config,
                server_config=self.py_env_configs.server_config,
                tokenizer=self.tokenizer,
                global_controller=self.global_controller,
                access_logger=self.access_logger,
                rank_id=self.rank_id,
                server_id=self.server_id,
                active_requests=self.active_requests,
            )
            self.grpc_client.set_embedding_port(
                self.server_config.embedding_rpc_server_port
            )

    def stop(self):
        if self.frontend_worker is not None:
            self.frontend_worker.stop()

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        """Delegate to EmbeddingEndpoint (BaseEndpoint pipeline via handle_request)."""
        assert self.embedding_endpoint is not None
        return await self.embedding_endpoint.handle_request(request, raw_request)

    async def inference(self, req: Union[str, Dict[Any, Any]], raw_request: RawRequest):
        """Delegate to FrontendWorker (BaseEndpoint pipeline via handle_request)."""
        assert self.frontend_worker is not None
        return await self.frontend_worker.handle_request(req, raw_request)

    async def chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """Delegate to OpenaiEndpoint (BaseEndpoint pipeline with inference_request)."""
        assert self.openai_endpoint is not None
        return await self.openai_endpoint.handle_request(request, raw_request)

    async def chat_render(self, request: ChatCompletionRequest, raw_request: Request):
        self.active_requests.increment()
        try:
            assert self.openai_endpoint != None
            return self.openai_endpoint.chat_render(request)
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)
        finally:
            self.active_requests.decrement()

    def tokenize(self, req: str | Dict[str, Any]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            if ChatCompletionRequest.is_openai_request(req):
                chat_request = ChatCompletionRequest(**req)
                token_ids = self.openai_endpoint.render_chat(chat_request).input_ids
            else:
                prompt = req.pop("prompt")
                token_ids = self.tokenizer.encode(prompt)
            return ORJSONResponse({"token_ids": token_ids})
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def tokenizer_encode(self, req: Union[str, Dict[Any, Any]]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            prompt = req.pop("prompt")
            assert self.tokenizer is not None
            if req.get("return_offsets_mapping", None) == True:
                mapping = self.tokenizer(
                    prompt,
                    return_offsets_mapping=True,
                    return_attention_mask=False,
                )
                response = TokenizerEncodeResponse(
                    offset_mapping=mapping["offset_mapping"],
                    token_ids=mapping["input_ids"],
                )
            else:
                token_ids = self.tokenizer.encode(prompt)
                token_ids = [int(id) for id in token_ids]
                tokens = [self.tokenizer.decode([id]) for id in token_ids]
                response = TokenizerEncodeResponse(token_ids=token_ids, tokens=tokens)
            return ORJSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def is_backend_service_ready(self):
        """Return True if the backend RPC service is ready (used when separated_frontend)."""
        assert self.frontend_worker is not None
        return self.frontend_worker.backend_rpc_server_visitor.is_backend_service_ready(
            refresh=False
        )

    async def health_check(
        self, separated_frontend: bool
    ) -> Union[str, ORJSONResponse]:
        if separated_frontend:
            if not self.is_backend_service_ready():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="inference service is not ready",
                )
            return "ok"
        response = await self.grpc_client.post_request("health_check", {})
        if response.get("status", "") != "ok":
            return ORJSONResponse(
                status_code=400,
                content={"error": " HTTP health check failed"},
            )
        return "ok"

    async def health(
        self, separated_frontend: bool
    ) -> Union[Dict[str, str], ORJSONResponse]:
        if separated_frontend:
            if not self.is_backend_service_ready():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="inference service is not ready",
                )
            return {"status": "home"}
        response = await self.grpc_client.post_request("health_check", {})
        if response.get("status", "") != "ok":
            return ORJSONResponse(
                status_code=400,
                content={"error": " HTTP health check failed"},
            )
        return {"status": "home"}

    async def cache_status(
        self, request: Request, data: Optional[Dict[Any, Any]] = None
    ) -> Union[Dict, ORJSONResponse]:
        query_params = (
            dict(request.query_params) if request.method == "GET" else (data or {})
        )
        logging.info(f"cache_status request {data}")
        response = await self.grpc_client.post_request("cache_status", query_params)
        if "error" not in response:
            response["frontend_available_concurrency"] = (
                self.global_controller.get_available_concurrency()
            )
        logging.info(f"cache_status response {response}")
        if "error" in response:
            return ORJSONResponse(status_code=500, content=response)
        return response

    async def worker_status(
        self, request: Request, data: Optional[Dict[Any, Any]] = None
    ) -> Union[Dict, ORJSONResponse]:
        query_params = (
            dict(request.query_params) if request.method == "GET" else (data or {})
        )
        response = await self.grpc_client.post_request("worker_status", query_params)
        if "error" not in response:
            response["frontend_available_concurrency"] = (
                self.global_controller.get_available_concurrency()
            )
        else:
            return ORJSONResponse(status_code=500, content=response)
        return response

    async def set_log_level(self, req: Union[str, Dict[Any, Any]]):
        return await self.grpc_client.post_request("set_log_level", req)

    async def update_eplb_config(self, req: Union[str, Dict[Any, Any]]):
        return await self.grpc_client.post_request("update_eplb_config", req)

    async def update_scheduler_info(self, req: Union[str, Dict[Any, Any]]):
        return await self.grpc_client.post_request("update_scheduler_info", req)
