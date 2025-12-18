import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Union

from fastapi import Request
from fastapi import Request as RawRequest
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.config.generate_config import RoleType
from rtp_llm.config.gpt_init_model_parameters import ConfigMode, GptInitModelParameters
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.config.task_type import TaskType
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint
from rtp_llm.frontend.frontend_worker import FrontendWorker, TokenizerEncodeResponse
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.misc import format_exception
from rtp_llm.structure.request_extractor import (
    Request,
    RequestExtractor,
    request_id_field_name,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import check_with_info

USAGE_HEADER = "USAGE"


class FrontendServer(object):
    def __init__(
        self, separated_frontend: bool = False, rank_id: int = 0, server_id: int = 0
    ):
        self._access_logger = AccessLogger(rank_id, server_id)
        self._frontend_worker = None
        self._openai_endpoint = None
        self._embedding_endpoint = None
        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        self.separated_frontend = separated_frontend
        self.rank_id = str(rank_id)
        self.server_id = str(server_id)
        self.is_embedding = False

        # Initialize dependencies
        self._model_config = ModelFactory.create_frontend_config(
            ModelFactory.create_normal_model_config()
        )
        self._tokenizer = TokenizerFactory.create_from_env()
        self._model_config.update_task_prompt_tokens_id(self._tokenizer)
        self._model_config.update_tokenizer_special_tokens(self._tokenizer)
        self._backend_rpc_server_visitor = BackendRPCServerVisitor(
            self._model_config, separated_frontend
        )

        kmonitor.init()

    def start(self):
        if StaticConfig.profiling_debug_config.debug_start_fake_process == 1:
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake server")
            self._frontend_worker = None
        else:
            # Create frontend worker with dependencies
            self._frontend_worker = FrontendWorker(
                self._model_config,
                self._tokenizer,
                self._backend_rpc_server_visitor,
                self.rank_id,
                self.server_id,
            )

            # Initialize endpoints based on task type
            if (
                self._model_config is not None
                and self._model_config.task_type != TaskType.LANGUAGE_MODEL
            ):
                self.is_embedding = True
                model_config = ModelFactory.create_normal_model_config()
                global _model_factory
                if model_config.model_type not in _model_factory:
                    raise Exception(
                        f"model type {model_config.model_type} not registered!"
                    )
                model_cls = _model_factory[model_config.model_type]
                config: GptInitModelParameters = model_cls.create_config(model_config)
                self._embedding_endpoint = EmbeddingEndpoint(config, self._tokenizer)
            else:
                self._openai_endpoint = OpenaiEndpoint(
                    self._model_config,
                    self._tokenizer,
                    self._backend_rpc_server_visitor,
                    self.rank_id,
                    self.server_id,
                )

    def stop(self):
        if self._frontend_worker is not None:
            self._frontend_worker.stop()

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        start_time = time.time()
        try:
            if isinstance(request, str):
                request = json.loads(request)
            kmonitor.report(
                AccMetrics.QPS_METRIC, 1, {"source": request.get("source", "unknown")}
            )
            request[request_id_field_name] = self._global_controller.increment()
        except Exception as e:
            return self._handle_exception(request, e)

        try:
            assert (
                self._embedding_endpoint is not None
            ), "embedding pipeline should not be None"
            result, logable_result = await self._embedding_endpoint.embedding(request)
            # do not log result since too big
            if logable_result is not None:
                self._access_logger.log_success_access(request, logable_result)
            end_time = time.time()
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, (end_time - start_time) * 1000
            )
            kmonitor.report(
                AccMetrics.SUCCESS_QPS_METRIC,
                1,
                {"source": request.get("source", "unknown")},
            )
            usage = result.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            return ORJSONResponse(result, headers={USAGE_HEADER: json.dumps(usage)})
        except BaseException as e:
            return self._handle_exception(request, e)
        finally:
            self._global_controller.decrement()

    # use asyncio.sleep(0) to correctly exit when client closed https://github.com/tiangolo/fastapi/issues/4146

    async def inference(
        self, request: Union[str, Dict[Any, Any]], raw_request: RawRequest
    ):
        assert self._frontend_worker is not None
        req_id = self._global_controller.increment()
        request = self._frontend_worker._check_request(request, req_id)

        try:
            self._frontend_worker._report_qps_metrics(request)

            if await raw_request.is_disconnected():
                raise asyncio.CancelledError("client disconnects")

            response_generator = self._frontend_worker.inference(**request)

            res = await self._frontend_worker._call_generate_with_report(
                lambda: response_generator
            )

            if self._frontend_worker._check_is_streaming(request):
                return StreamingResponse(
                    self._frontend_worker._stream_response(request, res),
                    media_type="text/event-stream",
                )

            async for x in res:
                if await raw_request.is_disconnected():
                    await res.aclose()
                    raise asyncio.CancelledError("client disconnects")

            complete_response = await self._frontend_worker._collect_complete_response_and_record_access_log(
                request, res
            )
            self._global_controller.decrement()
            return ORJSONResponse(content=complete_response)

        except BaseException as e:
            self._global_controller.decrement()
            return self._frontend_worker._handle_exception(request, e)

    async def chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        assert self._openai_endpoint != None

        req_id = self._global_controller.increment()
        request_dict = self._openai_endpoint._check_request(request, req_id)
        try:
            self._openai_endpoint._report_qps_metrics(request_dict)
            if await raw_request.is_disconnected():
                raise asyncio.CancelledError("client disconnects")

            response_generator = self._openai_endpoint.chat_completion(
                request_id=request_dict[request_id_field_name],
                chat_request=request,
                raw_request=raw_request,
            )
            assert isinstance(
                response_generator, CompleteResponseAsyncGenerator
            ), f"error type: {type(response_generator)}"
            res = await self._openai_endpoint._call_generate_with_report(
                lambda: response_generator
            )

            if self._openai_endpoint._check_is_streaming(request_dict):
                return StreamingResponse(
                    self._openai_endpoint._stream_response(request_dict, res),
                    media_type="text/event-stream",
                )

            async for x in res:
                if await raw_request.is_disconnected():
                    await res.aclose()
                    raise asyncio.CancelledError("client disconnects")

            complete_response = await self._openai_endpoint._collect_complete_response_and_record_access_log(
                request_dict, res
            )
            self._global_controller.decrement()
            return ORJSONResponse(content=complete_response)

        except BaseException as e:
            self._global_controller.decrement()
            return self._openai_endpoint._handle_exception(
                request_dict if "request_dict" in locals() else request, e
            )

    async def chat_render(self, request: ChatCompletionRequest, raw_request: Request):
        assert self._openai_endpoint != None
        try:
            return self._openai_endpoint.chat_render(request)
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def tokenize(self, req: str | Dict[str, Any]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            if ChatCompletionRequest.is_openai_request(req):
                chat_request = ChatCompletionRequest(**req)
                token_ids = self._openai_endpoint.render_chat(chat_request).input_ids
            else:
                prompt = req.pop("prompt")
                token_ids = self._tokenizer.encode(prompt)
            return ORJSONResponse({"token_ids": token_ids})
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def tokenizer_encode(self, req: Union[str, Dict[Any, Any]]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            prompt = req.pop("prompt")
            assert self._frontend_worker is not None
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
                token_ids = [int(id) for id in token_ids]
                tokens = [self._tokenizer.decode(id) for id in token_ids]
                response = TokenizerEncodeResponse(token_ids=token_ids, tokens=tokens)
            return ORJSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def check_health(self):
        assert self._frontend_worker is not None
        return self._backend_rpc_server_visitor.is_backend_service_ready(refresh=False)
