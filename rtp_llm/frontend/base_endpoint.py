import asyncio
import json
import logging
from typing import Any, Callable, Dict, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.server.misc import format_exception
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import ConcurrencyException
from rtp_llm.utils.time_util import current_time_ms


class BaseEndpoint:
    """Base class for all endpoints with common pipeline processing methods"""

    def __init__(
        self,
        global_controller=None,
        access_logger: AccessLogger = None,
        rank_id: str = "0",
        server_id: str = "0",
    ):
        """
        Initialize base endpoint with common dependencies

        Args:
            frontend_worker: Frontend worker for request processing
            global_controller: Concurrency controller for request management
            access_logger: Logger for access logs
            rank_id: Rank ID for metrics reporting
            server_id: Server ID for metrics reporting
        """
        self._global_controller = global_controller
        self._access_logger = access_logger
        self.rank_id = rank_id
        self.server_id = server_id

    def _handle_exception(self, request: Union[Dict[str, Any], Any], e: BaseException):
        """Handle exceptions and return proper error response"""
        exception_json = format_exception(e)
        error_code_str = exception_json.get("error_code_str", "")

        # Convert request to dict if needed
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
    ):
        """Wrap generator with performance monitoring"""

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

    async def _collect_complete_response_and_record_access_log(
        self, req: Dict[Any, Any], res: Any
    ):
        """Collect complete response and log access"""
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
        """Generate streaming response"""
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
            if self._access_logger:
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
            if self._access_logger:
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
            if self._global_controller:
                self._global_controller.decrement()

    # ==================== Fine-Grained Pipeline Methods ====================

    # ===== Abstract Methods (Subclass must implement) =====

    async def _parse_request(
        self, request: Any, raw_request: Request
    ) -> Dict[str, Any]:
        """
        Parse request and convert to dictionary format

        Subclass must implement this to handle request-specific parsing logic.

        Args:
            request: Request object (e.g., ChatCompletionRequest or dict)
            raw_request: FastAPI Request object

        Returns:
            request_dict: Parsed request as dictionary

        Raises:
            NotImplementedError: Subclass must implement this method
        """
        raise NotImplementedError("Subclass must implement _parse_request")

    async def _create_response_generator_core(
        self,
        request: Any,
        request_id: int,
        request_dict: Dict[str, Any],
        raw_request: Request,
    ) -> CompleteResponseAsyncGenerator:
        """
        Create response generator with core generation logic

        Subclass must implement this to provide the actual generation logic.

        Args:
            request: Original request object
            request_id: Generated request ID
            request_dict: Parsed request dictionary
            raw_request: FastAPI Request object

        Returns:
            response_generator: CompleteResponseAsyncGenerator

        Raises:
            NotImplementedError: Subclass must implement this method
        """
        raise NotImplementedError(
            "Subclass must implement _create_response_generator_core"
        )

    # ===== Common Utility Methods (BaseEndpoint implements) =====

    def _convert_to_dict(self, request: Any) -> Dict[str, Any]:
        """Convert request to dictionary format"""
        if isinstance(request, dict):
            return request
        elif hasattr(request, "model_dump"):
            return request.model_dump(exclude_none=True)
        elif hasattr(request, "dict"):
            return request.dict(exclude_none=True)
        else:
            return {}

    def _extract_or_generate_request_id(self, request_dict: Dict[str, Any]) -> int:
        """Extract request_id from master_info or generate new one"""
        from rtp_llm.structure.request_extractor import request_id_field_name

        if "master_info" in request_dict:
            request_id = request_dict["master_info"].get("request_id")
            if request_id is not None and isinstance(request_id, int):
                # Use provided request_id
                return request_id

        # Generate new request_id
        if self._global_controller:
            request_id = self._global_controller.increment()
        else:
            request_id = 0

        return request_id

    def _increment_concurrency(self) -> None:
        """Increment concurrency counter"""
        if self._global_controller:
            self._global_controller.increment()

    def _decrement_concurrency(self) -> None:
        """Decrement concurrency counter"""
        if self._global_controller:
            self._global_controller.decrement()

    def _report_qps_metrics(self, request_dict: Dict[str, Any]) -> None:
        """Report QPS metrics"""
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
        """Log query access"""
        if self._access_logger:
            self._access_logger.log_query_access(request_dict)

    def _determine_streaming_mode(self, request_dict: Dict[str, Any]) -> bool:
        """Determine if request should use streaming mode"""
        if self._frontend_worker and hasattr(self._frontend_worker, "is_streaming"):
            return self._frontend_worker.is_streaming(request_dict)
        return request_dict.get("stream", False)

    async def _check_client_connection(self, raw_request: Request) -> None:
        """Check if client is still connected"""
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")

    async def _handle_response(
        self,
        request_dict: Dict[str, Any],
        response_generator: CompleteResponseAsyncGenerator,
        is_streaming: bool,
        raw_request: Request,
    ) -> Union[StreamingResponse, ORJSONResponse]:
        """
        Handle response based on streaming mode

        This method handles both streaming and non-streaming responses:
        - For streaming: return StreamingResponse with stream_response generator
        - For non-streaming: iterate response, check disconnection, return complete response

        Args:
            request_dict: Request dictionary
            response_generator: Response generator from _execute_generation
            is_streaming: Whether to use streaming response
            raw_request: FastAPI Request object

        Returns:
            StreamingResponse or ORJSONResponse
        """
        if is_streaming:
            return StreamingResponse(
                self.stream_response(request_dict, response_generator),
                media_type="text/event-stream",
            )
        else:
            async for x in response_generator:
                if await raw_request.is_disconnected():
                    await response_generator.aclose()
                    raise asyncio.CancelledError("client disconnects")

            complete_response = (
                await self._collect_complete_response_and_record_access_log(
                    request_dict, response_generator
                )
            )
            return ORJSONResponse(content=complete_response)

    def _cleanup_resources(
        self, response: Union[StreamingResponse, ORJSONResponse]
    ) -> None:
        """
        Clean up resources after request processing

        This method performs resource cleanup:
        - For non-streaming responses: decrement global_controller
        - For streaming responses: cleanup is handled in stream_response generator

        Args:
            response: Final response object
        """
        if not isinstance(response, StreamingResponse):
            if self._global_controller:
                self._global_controller.decrement()

    async def handle_request(self, request: Any, raw_request: Request):
        """
        Fine-grained unified request handling pipeline

        This method orchestrates the complete request processing flow with 11 steps:
        1. Parse request
        2. Extract or generate request ID
        3. Increment concurrency
        4. Report QPS metrics
        5. Log query access
        6. Determine streaming mode
        7. Check client connection
        8. Create response generator
        9. Wrap with monitoring
        10. Handle response
        11. Cleanup resources

        Args:
            request: Request object (e.g., ChatCompletionRequest or dict)
            raw_request: FastAPI Request object

        Returns:
            Response object (StreamingResponse or ORJSONResponse)
        """
        from rtp_llm.structure.request_extractor import request_id_field_name

        # Step 1: Parse request
        try:
            request_dict = await self._parse_request(request, raw_request)
        except Exception as e:
            return self._handle_exception(request, e)

        try:
            # Step 2: Extract or generate request ID
            request_id = self._extract_or_generate_request_id(request_dict)
            request_dict[request_id_field_name] = request_id

            # Step 3: Increment concurrency
            self._increment_concurrency()

            # Step 4: Report QPS metrics
            self._report_qps_metrics(request_dict)

            # Step 5: Log query access
            self._log_query_access(request_dict)

            # Step 6: Determine streaming mode
            is_streaming = self._determine_streaming_mode(request_dict)

            # Step 7: Check client connection
            await self._check_client_connection(raw_request)

            # Step 8: Create response generator
            response_generator = await self._create_response_generator_core(
                request, request_id, request_dict, raw_request
            )

            # Step 9: Wrap with monitoring
            response_generator = await self._call_generate_with_report(
                lambda: response_generator
            )

            # Step 10: Handle response
            response = await self._handle_response(
                request_dict, response_generator, is_streaming, raw_request
            )

        except BaseException as e:
            # Step 11a: Cleanup on error
            self._decrement_concurrency()
            return self._handle_exception(
                request_dict if "request_dict" in locals() else request, e
            )

        # Step 11b: Cleanup on success
        self._cleanup_resources(response)

        return response
