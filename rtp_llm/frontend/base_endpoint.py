"""Base endpoint with common request pipeline for frontend worker and OpenAI endpoint."""

import asyncio
import json
from typing import Any, Dict, Optional, Tuple, Union

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
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    ConcurrencyException,
)
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter


class BaseEndpoint:
    """Base class for all endpoints with common pipeline: _check_request + inference_request + handle_request."""

    def __init__(
        self,
        global_controller: ConcurrencyController,
        access_logger: AccessLogger,
        rank_id: str = "0",
        server_id: str = "0",
        active_requests: Optional[AtomicCounter] = None,
    ):
        self._global_controller = global_controller
        self._access_logger = access_logger
        self.rank_id = rank_id
        self.server_id = server_id
        self._active_requests = (
            active_requests if active_requests is not None else AtomicCounter()
        )

    def _log_exception(self, request: Dict[str, Any], e: BaseException) -> None:
        """Log exception to access_logger only."""
        self._access_logger.log_exception_access(request, e)

    def _report_exception(
        self, request: Dict[str, Any], e: BaseException
    ) -> Optional[Dict[str, Any]]:
        """Report CANCEL_QPS or ERROR_QPS. Returns format_exception(e) for non-CancelledError, None for CancelledError."""
        source = request.get("source", "unknown")
        tags = {
            "rank_id": self.rank_id,
            "server_id": self.server_id,
            "source": source,
        }
        if isinstance(e, asyncio.CancelledError):
            kmonitor.report(AccMetrics.CANCEL_QPS_METRIC, 1, tags)
            return None
        format_e = format_exception(e)
        kmonitor.report(
            AccMetrics.ERROR_QPS_METRIC,
            1,
            {**tags, "error_code": str(format_e.get("error_code_str", -1))},
        )
        return format_e

    def _log_and_report_exception(
        self, request: Dict[str, Any], e: BaseException
    ) -> Optional[Dict[str, Any]]:
        """Log exception and report metrics. Returns format_exception(e) for non-CancelledError, None for CancelledError."""
        self._log_exception(request, e)
        return self._report_exception(request, e)

    def _handle_exception(
        self, request: Union[Dict[str, Any], Any], e: BaseException
    ) -> ORJSONResponse:
        """Handle exceptions and return proper error response (no stack trace to client)."""
        if isinstance(request, dict):
            request_dict = request
        elif hasattr(request, "model_dump"):
            request_dict = request.model_dump(exclude_none=True)
        else:
            request_dict = {}

        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
            exception_json = format_exception(e)
        else:
            self._log_exception(request_dict, e)
            exception_json = self._report_exception(request_dict, e)
            if exception_json is None:
                exception_json = format_exception(e)
        return ORJSONResponse(exception_json, status_code=500)

    async def _collect_complete_response_and_record_access_log(
        self, req: Dict[Any, Any], res: Any
    ):
        """Collect complete response and log access."""
        complete_response = await res.gen_complete_response_once()
        complete_response = self._normalize_complete_response(complete_response)
        self._access_logger.log_success_access(req, complete_response)
        return complete_response

    async def stream_response(
        self,
        request: Dict[str, Any],
        response_generator: CompleteResponseAsyncGenerator,
    ):
        """Generate streaming response with metrics reported during iteration."""
        response_data_prefix = "data: " if request.get("stream", False) else "data:"
        async for chunk in self._stream_with_controller_lifecycle(
            request, response_generator, response_data_prefix
        ):
            yield chunk

    @staticmethod
    def _step_output_len_from_response(response: Any) -> int:
        """Extract step_output_len from response.aux_info for iter RT metric."""
        if not hasattr(response, "aux_info"):
            return 1
        aux = response.aux_info
        if isinstance(aux, list):
            return sum(info.get("step_output_len", 1) for info in aux) or 1
        if isinstance(aux, dict):
            return max(aux.get("step_output_len", 1), 1)
        return 1

    def _report_stream_iter_metrics(
        self,
        response: Any,
        end_time: float,
        last_iterate_time: float,
        first_token: bool,
    ) -> Tuple[float, bool]:
        """Report first-token or iter RT and ITER_QPS; return (end_time, first_token after)."""
        rt = end_time - last_iterate_time
        if first_token:
            kmonitor.report(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC, rt)
            first_token = False
        else:
            step_len = self._step_output_len_from_response(response)
            kmonitor.report(GaugeMetrics.RESPONSE_ITER_RT_METRIC, rt / step_len)
        kmonitor.report(
            AccMetrics.ITER_QPS_METRIC,
            1,
            {"rank_id": self.rank_id, "server_id": self.server_id},
        )
        return (end_time, first_token)

    async def _stream_with_controller_lifecycle(
        self,
        request: Dict[str, Any],
        response_generator: CompleteResponseAsyncGenerator,
        response_data_prefix: str,
    ):
        """Stream body with try/except/finally: log+report on exception, decrement in finally."""
        is_openai_response = response_data_prefix == "data: "
        start_time = current_time_ms()
        last_iterate_time = start_time
        first_token = True
        iter_count = 0
        iter_tags = {"rank_id": self.rank_id, "server_id": self.server_id}
        try:
            async for res in response_generator:
                end_time = current_time_ms()
                last_iterate_time, first_token = self._report_stream_iter_metrics(
                    res, end_time, last_iterate_time, first_token
                )
                iter_count += 1
                yield response_data_prefix + res.model_dump_json(
                    exclude_none=True
                ) + "\r\n\r\n"
                await asyncio.sleep(0)
            kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, current_time_ms() - start_time
            )
            kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, iter_tags)
            if not is_openai_response:
                yield "data:[done]\r\n\r\n"
            await self._collect_complete_response_and_record_access_log(
                request, response_generator
            )
        except asyncio.CancelledError as e:
            self._log_and_report_exception(request, e)
        except BaseException as e:
            format_e = self._log_and_report_exception(request, e)
            if format_e is not None:
                yield response_data_prefix + json.dumps(
                    format_e, ensure_ascii=False
                ) + "\r\n\r\n"
        finally:
            self._global_controller.decrement()

    def _check_is_streaming(self, request_dict: Dict[str, Any]) -> bool:
        """Determine if request should use streaming mode."""
        return RequestExtractor.is_streaming(request_dict) or request_dict.get(
            "stream", False
        )

    # ---------- Abstract: subclass implements ----------

    def _check_request(self, request: Any, req_id: int) -> Dict[str, Any]:
        """Parse/validate request and set request_id in dict. Subclass must implement."""
        raise NotImplementedError("Subclass must implement _check_request")

    def inference_request(
        self,
        request_dict: Dict[str, Any],
        raw_request: Optional[Request] = None,
    ) -> CompleteResponseAsyncGenerator:
        """Build response generator from request_dict. Subclass must implement (for streaming path)."""
        raise NotImplementedError("Subclass must implement inference_request")

    def _finish_with_response(
        self,
        request_dict: Dict[str, Any],
        complete_response: Any,
        logable_for_log: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> ORJSONResponse:
        """Decrement, log success access, return ORJSONResponse. Shared by direct and non-streaming paths."""
        self._global_controller.decrement()
        self._access_logger.log_success_access(
            request_dict,
            logable_for_log if logable_for_log is not None else complete_response,
        )
        return ORJSONResponse(
            content=complete_response, headers=headers if headers else {}
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

    def _normalize_complete_response(self, complete_response: Any) -> Any:
        """Return dict form of complete_response if BaseModel, else unchanged."""
        if isinstance(complete_response, BaseModel):
            return complete_response.model_dump(exclude_none=True)
        return complete_response

    async def handle_request(
        self, request: Any, raw_request: Request
    ) -> Union[StreamingResponse, ORJSONResponse]:
        """Unified pipeline: active_requests, then controller + _check_request, metrics, inference_request, stream or collect."""
        self._active_requests.increment()
        try:
            return await self._handle_request_impl(request, raw_request)
        finally:
            self._active_requests.decrement()

    async def _run_non_streaming(
        self,
        request_dict: Dict[str, Any],
        response_generator: CompleteResponseAsyncGenerator,
        raw_request: Request,
    ) -> ORJSONResponse:
        """Iterate generator with metrics, collect complete response, then finish. Used by _handle_request_impl for non-streaming path."""
        start_time = current_time_ms()
        last_iterate_time = start_time
        first_token = True
        iter_count = 0
        iter_tags = {"rank_id": self.rank_id, "server_id": self.server_id}
        async for response in response_generator:
            if await raw_request.is_disconnected():
                await response_generator.aclose()
                raise asyncio.CancelledError("client disconnects")
            end_time = current_time_ms()
            last_iterate_time, first_token = self._report_stream_iter_metrics(
                response, end_time, last_iterate_time, first_token
            )
            iter_count += 1
        kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
        kmonitor.report(GaugeMetrics.LANTENCY_METRIC, current_time_ms() - start_time)
        kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, iter_tags)
        complete_response = await response_generator.gen_complete_response_once()
        complete_response = self._normalize_complete_response(complete_response)
        return self._finish_with_response(
            request_dict,
            complete_response,
            logable_for_log=complete_response,
        )

    async def _obtain_request_dict(
        self, request: Any, raw_request: Request
    ) -> Dict[str, Any]:
        """Increment controller, validate request, report QPS, log access, check disconnect. Caller must handle exceptions and decrement on error."""
        req_id = self._global_controller.increment()
        request_dict = self._check_request(request, req_id)
        kmonitor.report(
            AccMetrics.QPS_METRIC,
            1,
            {
                "rank_id": self.rank_id,
                "server_id": self.server_id,
                "source": request_dict.get("source", "unknown"),
            },
        )
        self._access_logger.log_query_access(request_dict)
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        return request_dict

    async def _handle_request_impl(
        self, request: Any, raw_request: Request
    ) -> Union[StreamingResponse, ORJSONResponse]:
        """Controller lifecycle + core: obtain request_dict, then streaming or non-streaming path. Single try/except for all errors."""
        request_dict = None
        try:
            request_dict = await self._obtain_request_dict(request, raw_request)

            response_generator = self.inference_request(request_dict, raw_request)

            if self._check_is_streaming(request_dict):
                return StreamingResponse(
                    self.stream_response(request_dict, response_generator),
                    media_type="text/event-stream",
                )
            return await self._run_non_streaming(
                request_dict, response_generator, raw_request
            )
        except BaseException as e:
            self._global_controller.decrement()
            req_for_error = (
                request_dict
                if request_dict is not None
                else self._convert_to_dict(request)
            )
            return self._handle_exception(req_for_error, e)
