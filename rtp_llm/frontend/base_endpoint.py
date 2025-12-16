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
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.openai.api_datatype import ChatCompletionRequest
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
    ConcurrencyController,
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import check_with_info

USAGE_HEADER = "USAGE"


class BaseEndpoint(object):
    def __init__(
        self,
        model_config: GptInitModelParameters,
        tokenizer,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
        rank_id,
        server_id,
    ):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.backend_rpc_server_visitor = backend_rpc_server_visitor
        self._access_logger = AccessLogger()
        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        self.rank_id = rank_id
        self.server_id = server_id

        # Initialize dependencies
        self._model_config = ModelFactory.create_frontend_config(
            ModelFactory.create_normal_model_config()
        )
        self._tokenizer = TokenizerFactory.create_from_env()
        self._model_config.update_task_prompt_tokens_id(self._tokenizer)
        self._model_config.update_tokenizer_special_tokens(self._tokenizer)

        kmonitor.init()

    def _check_request(self, request: Dict[str, Any]) -> None:
        raise NotImplementedError

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

    def _report_qps_metrics(self, req: Dict[str, Any]):
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

    def _check_is_streaming(self, request: Dict[str, Any]) -> bool:
        return RequestExtractor.is_streaming(request) or request.get("stream", False)

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

        start_time = current_time_ms()
        response_generator = generate_call()
        return CompleteResponseAsyncGenerator(
            __gen_response_with_report(start_time, response_generator),
            response_generator._collect_complete_response_func,
        )

    async def _stream_response(
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

    async def handle_request(
        self, request: Dict[str, Any], raw_request
    ) -> Dict[str, Any]:
        req_id = self._global_controller.increment()
        request_dict = self._check_request(request, req_id)
        try:
            self._report_qps_metrics(request_dict)
            if await raw_request.is_disconnected():
                raise asyncio.CancelledError("client disconnects")

            response_generator = self.inference_request(request_dict)
            res = await self._call_generate_with_report(lambda: response_generator)

            if self._check_is_streaming(request_dict):
                return StreamingResponse(
                    self.stream_response(request_dict, res),
                    media_type="text/event-stream",
                )

            async for x in res:
                if await raw_request.is_disconnected():
                    await res.aclose()
                    raise asyncio.CancelledError("client disconnects")

            complete_response = (
                await self._collect_complete_response_and_record_access_log(
                    request_dict, res
                )
            )
            self._global_controller.decrement()
            return ORJSONResponse(content=complete_response)

        except BaseException as e:
            self._global_controller.decrement()
            return self._handle_exception(
                request_dict if "request_dict" in locals() else request, e
            )

    def inference_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
