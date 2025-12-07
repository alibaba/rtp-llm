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
from rtp_llm.frontend.frontend_worker import FrontendWorker, TokenizationResponse
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
        self._access_logger = AccessLogger()
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
                self._model_config, self._tokenizer, self._backend_rpc_server_visitor
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
    async def stream_response(
        self,
        request: Dict[str, Any],
        response: CompleteResponseAsyncGenerator,
    ):
        is_openai_response = request.get("stream", False)
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            async for res in response:
                data_str = res.model_dump_json(exclude_none=True)
                yield response_data_prefix + data_str + "\r\n\r\n"
                await asyncio.sleep(0)
            if not is_openai_response:
                yield f"data:[done]\r\n\r\n"
            await self._collect_complete_response_and_record_access_log(
                request, response
            )
        except asyncio.CancelledError as e:
            self._access_logger.log_exception_access(request, e)
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unkown"),
                },
            )
        except BaseException as e:
            # 捕获非Cancel以外所有的异常,所以使用BaseException
            self._access_logger.log_exception_access(request, e)
            format_e = format_exception(e)
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unkown"),
                    "error_code": str(format_e.get("error_code_str", -1)),
                },
            )
            yield response_data_prefix + json.dumps(
                format_e, ensure_ascii=False
            ) + "\r\n\r\n"
        finally:
            self._global_controller.decrement()

    async def generate_response(
        self, req: Union[str, Dict[Any, Any]], raw_request: RawRequest
    ):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            if "master_info" in req:
                request_id = req["master_info"].get("request_id")
                check_with_info(
                    request_id != None and isinstance(request_id, int),
                    "request_id in master_info is None or not int",
                )
                req[request_id_field_name] = request_id
                self._global_controller.increment()
            else:
                req[request_id_field_name] = self._global_controller.increment()
        except Exception as e:
            return self._handle_exception(req, e)

        def generate_call():
            assert self._frontend_worker is not None
            return self._frontend_worker.generate_response(**req)

        try:
            rep = await self._infer_wrap(req, raw_request, generate_call)
        except Exception as e:
            self._global_controller.decrement()
            raise e

        if not isinstance(rep, StreamingResponse):
            self._global_controller.decrement()

        return rep

    async def _infer_wrap(
        self,
        req: Dict[str, Any],
        raw_request: RawRequest,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
    ):
        try:
            rep = await self._infer_impl(req, raw_request, generate_call)
        except BaseException as e:
            rep = self._handle_exception(req, e)
        return rep

    async def chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        try:
            if request.master_info is not None:
                request_id = request.master_info.get("request_id")
                check_with_info(
                    request_id != None and isinstance(request_id, int),
                    "request_id in master_info is None or not int",
                )
                self._global_controller.increment()
            else:
                request_id = self._global_controller.increment()
        except Exception as e:
            return self._handle_exception(request, e)

        def generate_call():
            assert self._openai_endpoint != None
            response = self._openai_endpoint.chat_completion(
                request_id, request, raw_request
            )
            assert isinstance(
                response, CompleteResponseAsyncGenerator
            ), f"error type: {type(response)}"
            return response

        try:
            request_dict = request.model_dump(exclude_none=True)
            request_dict[request_id_field_name] = request_id
            rep = await self._infer_wrap(request_dict, raw_request, generate_call)
        except Exception as e:
            self._global_controller.decrement()
            raise e

        if not isinstance(rep, StreamingResponse):
            self._global_controller.decrement()

        return rep

    async def chat_render(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            assert self._openai_endpoint != None
            return self._openai_endpoint.chat_render(request)
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def _handle_exception(self, request: Dict[str, Any], e: BaseException):
        exception_json = format_exception(e)
        error_code_str = exception_json.get("error_code_str", "")
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
        elif isinstance(e, asyncio.CancelledError):
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                },
            )
            self._access_logger.log_exception_access(request, e)
        else:
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                    "error_code": error_code_str,
                },
            )
            self._access_logger.log_exception_access(request, e)

        rep = ORJSONResponse(exception_json, status_code=500)
        return rep

    async def _call_generate_with_report(
        self, generate_call: Callable[[], CompleteResponseAsyncGenerator]
    ):
        async def __gen_response_with_report(start_time: float, response_generator):
            last_iterate_time = current_time_ms()
            first_token = True
            iter_count = 0
            async for response in response_generator:
                end_time = current_time_ms()
                if first_token:
                    first_token = False
                    kmonitor.report(
                        GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC,
                        end_time - last_iterate_time,
                    )
                else:
                    step_output_len = 1
                    if hasattr(response, "aux_info"):
                        if isinstance(response.aux_info, list):
                            step_output_len = 0
                            for info in response.aux_info:
                                step_output_len += info.get("step_output_len", 1)
                        elif isinstance(response.aux_info, dict):
                            step_output_len = max(
                                response.aux_info.get("step_output_len", 1),
                                step_output_len,
                            )

                    kmonitor.report(
                        GaugeMetrics.RESPONSE_ITER_RT_METRIC,
                        (end_time - last_iterate_time) / step_output_len,
                    )
                kmonitor.report(
                    AccMetrics.ITER_QPS_METRIC,
                    1,
                    {
                        "rank_id": self.rank_id,
                        "server_id": self.server_id,
                    },
                )
                last_iterate_time = end_time
                iter_count += 1
                yield response
            kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, current_time_ms() - start_time
            )
            kmonitor.report(
                AccMetrics.SUCCESS_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                },
            )

        assert self._frontend_worker is not None
        start_time = current_time_ms()
        response_generator = generate_call()
        return CompleteResponseAsyncGenerator(
            __gen_response_with_report(start_time, response_generator),
            response_generator._collect_complete_response_func,
        )

    async def _collect_complete_response_and_record_access_log(
        self, req: Dict[Any, Any], res: Any
    ):
        complete_response = await res.gen_complete_response_once()
        complete_response = (
            complete_response.model_dump(exclude_none=True)
            if isinstance(complete_response, BaseModel)
            else complete_response
        )
        self._access_logger.log_success_access(req, complete_response)

        return complete_response

    async def _infer_impl(
        self,
        req: Dict[Any, Any],
        raw_request: RawRequest,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
    ):
        assert self._frontend_worker is not None
        kmonitor.report(
            AccMetrics.QPS_METRIC,
            1,
            {
                "rank_id": self.rank_id,
                "server_id": self.server_id,
                "source": req.get("source", "unkown"),
            },
        )
        self._access_logger.log_query_access(req)
        is_streaming = RequestExtractor.is_streaming(req) or req.get("stream", False)
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        res = await self._call_generate_with_report(generate_call)

        if is_streaming:
            return StreamingResponse(
                self.stream_response(req, res), media_type="text/event-stream"
            )
        async for x in res:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await res.aclose()
                raise asyncio.CancelledError("client disconnects")

        complete_response = await self._collect_complete_response_and_record_access_log(
            req, res
        )
        return ORJSONResponse(content=complete_response)

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
                response = TokenizationResponse(
                    offset_mapping=mapping["offset_mapping"],
                    token_ids=mapping["input_ids"],
                )
            else:
                token_ids = self._tokenizer.encode(prompt)
                token_ids = [int(id) for id in token_ids]
                tokens = [self._tokenizer.decode(id) for id in token_ids]
                response = TokenizationResponse(token_ids=token_ids, tokens=tokens)
            return ORJSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def check_health(self):
        assert self._frontend_worker is not None
        return self._backend_rpc_server_visitor.is_backend_service_ready(refresh=False)
