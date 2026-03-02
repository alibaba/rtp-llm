"""Base endpoint with common request pipeline for frontend worker and OpenAI endpoint."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.server.misc import format_exception
from rtp_llm.structure.request_extractor import RequestExtractor
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import ConcurrencyException
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter


class BaseEndpoint:
    """Base class for all endpoints with common pipeline: _check_request + inference_request + handle_request."""

    def __init__(
        self,
        global_controller=None,
        access_logger: AccessLogger = None,
        rank_id: str = "0",
        server_id: str = "0",
        frontend_worker=None,
        active_requests: Optional[AtomicCounter] = None,
    ):
        self._global_controller = global_controller
        self._access_logger = access_logger
        self.rank_id = rank_id
        self.server_id = server_id
        self._frontend_worker = frontend_worker
        self._active_requests = active_requests

    def _handle_exception(
        self, request: Union[Dict[str, Any], Any], e: BaseException
    ) -> ORJSONResponse:
        """Handle exceptions and return proper error response (no stack trace to client)."""
        exception_json = format_exception(e)
        error_code_str = exception_json.get("error_code_str", "")
        request_dict = (
            request
            if isinstance(request, dict)
            else (
                request.model_dump(exclude_none=True)
                if hasattr(request, "model_dump")
                else {}
            )
        )
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
        elif isinstance(e, asyncio.CancelledError):
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request_dict.get("source", "unknown"),
                },
            )
            if self._access_logger:
                self._access_logger.log_exception_access(request_dict, e)
        else:
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request_dict.get("source", "unknown"),
                    "error_code": error_code_str,
                },
            )
            if self._access_logger:
                self._access_logger.log_exception_access(request_dict, e)
        return ORJSONResponse(exception_json, status_code=500)

    async def _call_generate_with_report(
        self, generate_call: Callable[[], CompleteResponseAsyncGenerator]
    ) -> CompleteResponseAsyncGenerator:
        """Wrap generator with performance monitoring."""

        async def __gen_response_with_report(
            start_time: float, response_generator: CompleteResponseAsyncGenerator
        ):
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

    async def _collect_complete_response_and_record_access_log(
        self, req: Dict[Any, Any], res: Any
    ):
        """Collect complete response and log access."""
        complete_response = await res.gen_complete_response_once()
        complete_response = (
            complete_response.model_dump(exclude_none=True)
            if isinstance(complete_response, BaseModel)
            else complete_response
        )
        if self._access_logger:
            self._access_logger.log_success_access(req, complete_response)
        return complete_response

    async def stream_response(
        self,
        request: Dict[str, Any],
        response: CompleteResponseAsyncGenerator,
    ):
        """Generate streaming response."""
        is_openai_response = request.get("stream", False)
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            async for res in response:
                data_str = res.model_dump_json(exclude_none=True)
                yield response_data_prefix + data_str + "\r\n\r\n"
                await asyncio.sleep(0)
            if not is_openai_response:
                yield "data:[done]\r\n\r\n"
            await self._collect_complete_response_and_record_access_log(
                request, response
            )
        except asyncio.CancelledError as e:
            if self._access_logger:
                self._access_logger.log_exception_access(request, e)
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                },
            )
        except BaseException as e:
            if self._access_logger:
                self._access_logger.log_exception_access(request, e)
            format_e = format_exception(e)
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                    "error_code": str(format_e.get("error_code_str", -1)),
                },
            )
            yield response_data_prefix + json.dumps(
                format_e, ensure_ascii=False
            ) + "\r\n\r\n"
        finally:
            if self._global_controller:
                self._global_controller.decrement()

    def _check_is_streaming(self, request_dict: Dict[str, Any]) -> bool:
        """Determine if request should use streaming mode."""
        if self._frontend_worker is not None and hasattr(
            self._frontend_worker, "is_streaming"
        ):
            return self._frontend_worker.is_streaming(request_dict)
        return RequestExtractor.is_streaming(request_dict) or request_dict.get(
            "stream", False
        )

    def _report_qps_metrics(self, request_dict: Dict[str, Any]) -> None:
        kmonitor.report(
            AccMetrics.QPS_METRIC,
            1,
            {
                "rank_id": self.rank_id,
                "server_id": self.server_id,
                "source": request_dict.get("source", "unknown"),
            },
        )

    def _log_query_access(self, request_dict: Dict[str, Any]) -> None:
        if self._access_logger:
            self._access_logger.log_query_access(request_dict)

    # ---------- Abstract: subclass implements ----------

    def _check_request(self, request: Any, req_id: int) -> Dict[str, Any]:
        """Parse/validate request and set request_id in dict. Subclass must implement."""
        raise NotImplementedError("Subclass must implement _check_request")

    def _use_non_streaming_path(self) -> bool:
        """Override to True for endpoints that only support non-streaming (e.g. embedding)."""
        return False

    def inference_request(
        self,
        request_dict: Dict[str, Any],
        raw_request: Optional[Request] = None,
    ) -> CompleteResponseAsyncGenerator:
        """Build response generator from request_dict. Subclass must implement (for streaming path)."""
        raise NotImplementedError("Subclass must implement inference_request")

    async def inference_request_non_streaming(
        self,
        request_dict: Dict[str, Any],
        raw_request: Optional[Request] = None,
    ) -> Any:
        """Return complete response directly. Override for non-streaming-only endpoints (e.g. embedding)."""
        raise NotImplementedError(
            "Subclass must implement inference_request_non_streaming when _use_non_streaming_path() is True"
        )

    # ---------- Unified pipeline ----------

    def _convert_to_dict(self, request: Any) -> Dict[str, Any]:
        """Convert request to dict for error handling."""
        if isinstance(request, dict):
            return request
        if hasattr(request, "model_dump"):
            return request.model_dump(exclude_none=True)
        if hasattr(request, "dict"):
            return request.dict(exclude_none=True)
        return {}

    async def handle_request(
        self, request: Any, raw_request: Request
    ) -> Union[StreamingResponse, ORJSONResponse]:
        """Unified pipeline: active_requests, increment, _check_request, metrics, inference_request, stream or collect."""
        if self._active_requests is not None:
            self._active_requests.increment()
        try:
            return await self._handle_request_impl(request, raw_request)
        finally:
            if self._active_requests is not None:
                self._active_requests.decrement()

    async def _handle_request_impl(
        self, request: Any, raw_request: Request
    ) -> Union[StreamingResponse, ORJSONResponse]:
        req_id = self._global_controller.increment() if self._global_controller else 0
        try:
            request_dict = self._check_request(request, req_id)
        except Exception as e:
            if self._global_controller:
                self._global_controller.decrement()
            return self._handle_exception(request, e)

        try:
            self._report_qps_metrics(request_dict)
            self._log_query_access(request_dict)
            if await raw_request.is_disconnected():
                raise asyncio.CancelledError("client disconnects")

            # Non-streaming-only path (e.g. embedding): no generator, direct response
            if self._use_non_streaming_path():
                start_time = current_time_ms()
                complete_response = await self.inference_request_non_streaming(
                    request_dict, raw_request
                )
                complete_response = (
                    complete_response.model_dump(exclude_none=True)
                    if isinstance(complete_response, BaseModel)
                    else complete_response
                )
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
                if self._access_logger:
                    self._access_logger.log_success_access(
                        request_dict, complete_response
                    )
                if self._global_controller:
                    self._global_controller.decrement()
                return ORJSONResponse(content=complete_response)

            # Streaming path: inference_request returns generator, then stream or collect
            response_generator = self.inference_request(request_dict, raw_request)
            res = await self._call_generate_with_report(lambda: response_generator)

            if self._check_is_streaming(request_dict):
                return StreamingResponse(
                    self.stream_response(request_dict, res),
                    media_type="text/event-stream",
                )

            async for _ in res:
                if await raw_request.is_disconnected():
                    await res.aclose()
                    raise asyncio.CancelledError("client disconnects")
            complete_response = (
                await self._collect_complete_response_and_record_access_log(
                    request_dict, res
                )
            )
            if self._global_controller:
                self._global_controller.decrement()
            return ORJSONResponse(content=complete_response)

        except BaseException as e:
            if self._global_controller:
                self._global_controller.decrement()
            req_for_error = (
                request_dict
                if "request_dict" in locals()
                else self._convert_to_dict(request)
            )
            return self._handle_exception(req_for_error, e)
