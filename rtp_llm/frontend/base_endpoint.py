import asyncio
import json
import threading
from typing import Any, Callable, Dict, Union

from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.server.misc import format_exception
from rtp_llm.structure.request_extractor import RequestExtractor
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    ConcurrencyException,
)
from rtp_llm.utils.time_util import current_time_ms

USAGE_HEADER = "USAGE"


class BaseEndpoint(object):
    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: BaseTokenizer,
        global_controller: ConcurrencyController,
        rank_id: int = 0,
        server_id: int = 0,
    ):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.rank_id = rank_id
        self.server_id = server_id
        self._access_logger = AccessLogger(
            log_path=get_log_path(),
            backup_count=20,
            rank_id=rank_id,
            server_id=server_id,
        )
        self.thread_lock_ = threading.Lock()
        self._global_controller = global_controller

        kmonitor.init()

    def _check_request(self, request: Any) -> None:
        raise NotImplementedError

    async def handle_request(self, request, raw_request) -> Dict[str, Any]:
        try:
            req_id = self._global_controller.increment()
            request = self._check_request(request, req_id)
            self._report_metric(
                AccMetrics.QPS_METRIC, source=request.get("source", "unkown")
            )
            self._access_logger.log_query_access(request)
            if await raw_request.is_disconnected():
                raise asyncio.CancelledError("client disconnects")

            response_generator = self.inference_request(request)
            res = await self._call_generate_with_report(lambda: response_generator)

            if self._check_is_streaming(request):
                return StreamingResponse(
                    self._stream_response(request, res),
                    media_type="text/event-stream",
                )
            async for x in res:
                if await raw_request.is_disconnected():
                    await res.aclose()
                    raise asyncio.CancelledError("client disconnects")
            complete_response = (
                await self._collect_complete_response_and_record_access_log(
                    request, res
                )
            )
            return ORJSONResponse(content=complete_response)
        except BaseException as e:
            return self._handle_exception(
                request if "request_dict" in locals() else request, e
            )
        finally:
            self._global_controller.decrement()

    def inference_request(self, request: Any):
        raise NotImplementedError

    def _handle_exception(self, request: Any, e: BaseException):
        exception_json = format_exception(e)
        error_code_str = exception_json.get("error_code_str", "")
        if isinstance(e, ConcurrencyException):
            self._report_metric(AccMetrics.CONFLICT_QPS_METRIC)
        elif isinstance(e, asyncio.CancelledError):
            self._report_metric(
                AccMetrics.CANCEL_QPS_METRIC, source=request.get("source", "unknown")
            )
            self._access_logger.log_exception_access(request, e)
        else:
            self._report_metric(
                AccMetrics.ERROR_QPS_METRIC,
                source=request.get("source", "unknown"),
                error_code=error_code_str,
            )
            self._access_logger.log_exception_access(request, e)

        rep = ORJSONResponse(exception_json, status_code=500)
        return rep

    def _report_metric(self, metric_type, value=1, **kwargs):
        tags = {
            "rank_id": str(self.rank_id),
            "server_id": str(self.server_id),
        }
        for key, val in kwargs.items():
            if val is not None:
                tags[key] = str(val)

        kmonitor.report(metric_type, value, tags)

    def _check_is_streaming(self, request: Any) -> bool:
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
                    self._report_metric(
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

                    self._report_metric(
                        GaugeMetrics.RESPONSE_ITER_RT_METRIC,
                        (end_time - last_iterate_time) / step_output_len,
                    )
                self._report_metric(AccMetrics.ITER_QPS_METRIC)
                last_iterate_time = end_time
                iter_count += 1
                yield response
            self._report_metric(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
            self._report_metric(
                GaugeMetrics.LANTENCY_METRIC, current_time_ms() - start_time
            )
            self._report_metric(AccMetrics.SUCCESS_QPS_METRIC)

        start_time = current_time_ms()
        response_generator = generate_call()
        return CompleteResponseAsyncGenerator(
            __gen_response_with_report(start_time, response_generator),
            response_generator._collect_complete_response_func,
        )

    async def _stream_response(
        self,
        request: Any,
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
            self._report_metric(
                AccMetrics.CANCEL_QPS_METRIC, source=request.get("source", "unknown")
            )
        except BaseException as e:
            # 捕获非Cancel以外所有的异常,所以使用BaseException
            self._access_logger.log_exception_access(request, e)
            format_e = format_exception(e)
            self._report_metric(
                AccMetrics.ERROR_QPS_METRIC,
                source=request.get("source", "unknown"),
                error_code=str(format_e.get("error_code_str", -1)),
            )

            yield response_data_prefix + json.dumps(
                format_e, ensure_ascii=False
            ) + "\r\n\r\n"

    async def _collect_complete_response_and_record_access_log(
        self, req: Any, res: Any
    ):
        complete_response = await res.gen_complete_response_once()
        complete_response = (
            complete_response.model_dump(exclude_none=True)
            if isinstance(complete_response, BaseModel)
            else complete_response
        )
        self._access_logger.log_success_access(req, complete_response)

        return complete_response
