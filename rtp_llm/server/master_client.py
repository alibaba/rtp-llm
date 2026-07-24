"""FlexLB schedule client: request role addrs from the master via gRPC."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import grpc
import grpc.aio

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.config.py_config_modules import MasterConfig
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
    CANCEL_REASON_CLIENT_CANCELLED,
    FlexlbCancelRequestPB,
    FlexlbScheduleRequestPB,
)
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2_grpc import (
    FlexlbServiceStub,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import GenerateInputPB
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.server.host_service import HostService
from rtp_llm.server.worker_status import _coerce_role_type
from rtp_llm.utils.base_model_datatypes import GenerateInput

route_logger = logging.getLogger("route_logger")

SUCCESS_CODE = 200
# gRPC = HTTP + 2 for FlexLB's own servers (consistent with FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET).
# This is NOT the same as the backend engine offset (HTTP+1)—see CommonConstants.GRPC_PORT_OFFSET.
FLEXLB_GRPC_PORT_OFFSET = 2
BEARER_PREFIX = "Bearer "

# gRPC status codes that indicate a connection-phase failure: the batch was
# never delivered to the server, so retrying on a slave is safe.  Other failures
# (e.g. stream established then timed out) may mean the batch was already
# dispatched and must not be retried.
_CONNECTION_PHASE_STATUS_CODES = frozenset(
    {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNKNOWN,
    }
)


def _resolve_role_from_server_status(s) -> RoleType:
    """Determine RoleType from the stable string role field."""
    if s.role:
        try:
            return _coerce_role_type(s.role)
        except (AttributeError, ValueError):
            pass
    return RoleType.PDFUSION


@dataclass
class FlexlbResponse:
    """
    Result of a FlexLB schedule request: success or failure state.

    Success: role_addrs is set. Failure: connection_failed and/or
    error_code/error_message from scheduler. request_id is always from frontend;
    only connection_failed permits domain fallback.  When a slave is available,
    connection-phase failures (batch never delivered) are retried on the slave
    before returning connection_failed; stream-level timeouts are not retried
    because the batch may already have been dispatched.
    """

    role_addrs: Optional[List[RoleAddr]] = None
    connection_failed: bool = False
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    enqueued_by_master: bool = False

    @property
    def is_ok(self) -> bool:
        return self.role_addrs is not None

    @classmethod
    def ok(
        cls,
        role_addrs: List[RoleAddr],
        enqueued_by_master: bool = False,
    ) -> "FlexlbResponse":
        """Business success: parsed role addrs."""
        return cls(
            role_addrs=role_addrs,
            connection_failed=False,
            error_code=None,
            error_message=None,
            enqueued_by_master=enqueued_by_master,
        )

    @classmethod
    def error_response(
        cls,
        error_code: int,
        error_message: Optional[str] = None,
    ) -> "FlexlbResponse":
        """Scheduler returned an error. Domain fallback is not allowed."""
        return cls(
            role_addrs=None,
            connection_failed=False,
            error_code=error_code,
            error_message=error_message,
            enqueued_by_master=False,
        )

    @classmethod
    def connection_failed_response(cls) -> "FlexlbResponse":
        """No response (connection/timeout). Permits domain fallback."""
        return cls(
            role_addrs=None,
            connection_failed=True,
            error_code=None,
            error_message=None,
            enqueued_by_master=False,
        )


class MasterClient:
    """Client for the FlexLB master Schedule gRPC API."""

    def __init__(self, host_service=None, server_config=None, master_config=None):
        self.master_config = (
            master_config if master_config is not None else MasterConfig()
        )
        self.host_service: Optional[HostService] = host_service
        self._channels: Dict[str, grpc.aio.Channel] = {}
        self.latest_queue_length: int = 0

    def _get_grpc_target(self, addr: str) -> str:
        """Resolve gRPC target from service discovery address (ip:HTTP_PORT).

        gRPC port is always derived as HTTP port + FLEXLB_GRPC_PORT_OFFSET.
        """
        ip = addr.split(":")[0]
        try:
            http_port = int(addr.split(":")[1])
            return f"{ip}:{http_port + FLEXLB_GRPC_PORT_OFFSET}"
        except (IndexError, ValueError):
            return f"{ip}:{7001 + FLEXLB_GRPC_PORT_OFFSET}"

    def _get_channel(self, target: str) -> grpc.aio.Channel:
        if target not in self._channels:
            self._channels[target] = grpc.aio.insecure_channel(
                target,
                options=[
                    ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                    ("grpc.max_send_message_length", 16 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                ],
            )
        return self._channels[target]

    async def _close_channel(self, target: str) -> None:
        channel = self._channels.pop(target, None)
        if channel is not None:
            await channel.close()

    async def close(self) -> None:
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()

    def get_latest_queue_length(self) -> int:
        return self.latest_queue_length

    async def _send_schedule_request(
        self,
        addr: str,
        request_pb: "FlexlbScheduleRequestPB",
        timeout_s: Optional[float],
        request_id: int,
    ) -> Tuple[Optional[Any], bool]:
        """Send gRPC schedule request.

        Returns ``(response, is_connection_phase_failure)`` where *response* is
        the proto on success or ``None`` on failure.  *is_connection_phase_failure*
        is ``True`` only when the failure happened before the batch was delivered
        (i.e. the gRPC status code is in ``_CONNECTION_PHASE_STATUS_CODES``),
        making a slave retry safe.
        """
        target = self._get_grpc_target(addr)
        start = time.time()
        try:
            channel = self._get_channel(target)
            stub = FlexlbServiceStub(channel)
            response = await stub.Schedule(request_pb, timeout=timeout_s)
            return response, False
        except grpc.aio.AioRpcError as e:
            elapsed = time.time() - start
            is_conn_phase = e.code() in _CONNECTION_PHASE_STATUS_CODES
            route_logger.error(
                "gRPC schedule failed, addr=%s, request_id=%s, status=%s, "
                "detail=%s, elapsed=%.3fs",
                addr,
                request_id,
                e.code(),
                e.details(),
                elapsed,
            )
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                # P0-2: Don't cancel or close channel — leave it for
                # passive recovery retry with the same request_id.
                raise FtRuntimeException(
                    exception_type=ExceptionType.DEADLINE_EXCEEDED,
                    message=f"FlexLB schedule deadline exceeded for request {request_id}",
                ) from e
            await self._close_channel(target)
            return None, is_conn_phase
        except asyncio.CancelledError:
            if "stub" in locals():
                await self._best_effort_cancel(
                    stub, request_id, CANCEL_REASON_CLIENT_CANCELLED
                )
            raise
        except Exception as e:
            elapsed = time.time() - start
            route_logger.exception(
                "Unexpected gRPC error, addr=%s, request_id=%s, elapsed=%.3fs",
                addr,
                request_id,
                elapsed,
            )
            await self._close_channel(target)
            return None, False

    @staticmethod
    async def _best_effort_cancel(stub, request_id: int, reason: int) -> None:
        try:
            await stub.Cancel(
                FlexlbCancelRequestPB(request_id=request_id, reason=reason),
                timeout=1.0,
            )
        except Exception:
            route_logger.warning(
                "best-effort FlexLB cancel failed, request_id=%s, reason=%s",
                request_id,
                reason,
                exc_info=True,
            )

    async def get_backend_role_addrs(
        self,
        block_cache_keys: list[int],
        cache_key_block_size: int,
        input: GenerateInput,
        request_id: int,
        input_pb: Optional["GenerateInputPB"] = None,
    ) -> FlexlbResponse:
        """
        Resolve backend role addrs from the FlexLB master.

        request_id is frontend-generated and only used for logging.
        On a connection-phase failure (batch never delivered to the server) the
        request is retried on the slave when one is available.  Stream-level
        timeouts are not retried because Schedule may already have taken effect.
        """
        master_addr = self.host_service.get_master_addr() if self.host_service else None
        if not master_addr:
            return FlexlbResponse.connection_failed_response()

        slave_addr = self.host_service.get_slave_addr() if self.host_service else None

        ttft_timeout_ms = getattr(
            input.generate_config, "ttft_timeout_ms", None
        ) or getattr(input.generate_config, "timeout_ms", None)
        if ttft_timeout_ms is None or ttft_timeout_ms <= 0:
            ttft_timeout_ms = self.master_config.master_default_timeout_ms

        # Absolute-deadline propagation: if absolute_deadline_ms is set (>0),
        # compute remaining time and use it as the gRPC timeout instead of the
        # full ttft_timeout_ms. If remaining is below the minimum threshold,
        # abort immediately without making the gRPC call.
        absolute_deadline_ms = getattr(input.generate_config, "absolute_deadline_ms", 0)
        if absolute_deadline_ms > 0:
            remaining_ms = absolute_deadline_ms - int(time.time() * 1000)
            min_remaining = self.master_config.min_remaining_deadline_ms
            # ttft_timeout_ms caps the Schedule phase; absolute_deadline_ms is the
            # overall deadline.  Cap remaining by ttft_timeout_ms so the Schedule
            # gRPC call never exceeds the TTFT budget.
            if ttft_timeout_ms > 0:
                remaining_ms = min(remaining_ms, ttft_timeout_ms)
            if remaining_ms < min_remaining:
                route_logger.warning(
                    "Schedule aborted: remaining %dms < min %dms, request_id=%s",
                    remaining_ms,
                    min_remaining,
                    request_id,
                )
                raise FtRuntimeException(
                    exception_type=ExceptionType.DEADLINE_EXCEEDED,
                    message=f"FlexLB schedule deadline exceeded (remaining {remaining_ms}ms < min {min_remaining}ms) for request {request_id}",
                )
            timeout_s = remaining_ms / 1000.0
            # generate_timeout passes remaining so that the Java side computes
            # absoluteDeadlineMs = request_time_ms + remaining = original deadline,
            # rather than request_time_ms + full ttft_timeout_ms which would extend
            # the deadline by the frontend processing time.
            effective_generate_timeout = remaining_ms
        else:
            # Fallback: absolute_deadline_ms not set (old client), use full timeout
            timeout_s = ttft_timeout_ms / 1000.0 if ttft_timeout_ms > 0 else None
            effective_generate_timeout = ttft_timeout_ms

        gc = input.generate_config
        api_key = self._extract_api_key(input)
        request_pb = FlexlbScheduleRequestPB(
            request_id=request_id,
            block_cache_keys=block_cache_keys,
            seq_len=input.prompt_length,
            generate_timeout=effective_generate_timeout,
            request_time_ms=int(time.time() * 1000),
            max_new_tokens=gc.max_new_tokens,
            num_beams=gc.num_beams,
            force_disable_sp_run=gc.force_disable_sp_run,
            model="engine_service",
            api_key=api_key,
            cache_key_block_size=cache_key_block_size,
        )
        if input_pb is not None:
            request_pb.generate_input = input_pb.SerializeToString()

        # P0-2: On deadline exceeded, retry the same master with the same
        # request_id (passive recovery).  Master-side duplicate detection
        # returns inflight routing info.
        route_logger.debug(
            "Schedule attempt starting, master=%s, request_id=%s, timeout=%.3fs",
            master_addr,
            request_id,
            timeout_s if timeout_s is not None else -1.0,
        )
        for schedule_attempt in range(2):
            try:
                schedule_start = time.monotonic()
                response, is_conn_phase = await self._send_schedule_request(
                    master_addr, request_pb, timeout_s, request_id
                )
                if response is not None:
                    route_logger.info(
                        "Schedule succeeded on attempt %d, request_id=%s, master=%s",
                        schedule_attempt + 1,
                        request_id,
                        master_addr,
                    )
                    break
                # response is None: connection-phase failure, fall through to slave retry
            except FtRuntimeException as e:
                if (
                    e.exception_type == ExceptionType.DEADLINE_EXCEEDED
                    and schedule_attempt == 0
                ):
                    elapsed = time.monotonic() - schedule_start
                    route_logger.warning(
                        "Schedule deadline exceeded after %.3fs, retrying same master "
                        "(passive recovery), request_id=%s, master=%s, attempt=%d/2",
                        elapsed,
                        request_id,
                        master_addr,
                        schedule_attempt + 1,
                    )
                    await asyncio.sleep(0.1)
                    continue
                if e.exception_type == ExceptionType.DEADLINE_EXCEEDED:
                    route_logger.warning(
                        "Schedule retry exhausted (deadline exceeded on attempt %d/2), "
                        "request_id=%s, master=%s",
                        schedule_attempt + 1,
                        request_id,
                        master_addr,
                    )
                raise

        if response is None and is_conn_phase and slave_addr:
            route_logger.info(
                "Master connection failed, retrying slave, master=%s, slave=%s, "
                "request_id=%s",
                master_addr,
                slave_addr,
                request_id,
            )
            response, _ = await self._send_schedule_request(
                slave_addr, request_pb, timeout_s, request_id
            )

        if response is None:
            return FlexlbResponse.connection_failed_response()

        self.latest_queue_length = response.queue_length

        if response.code != SUCCESS_CODE:
            try:
                exception_type = ExceptionType(response.code)
            except ValueError:
                exception_type = ExceptionType.MASTER_NO_AVAILABLE_WORKER
            message = response.error_message or "master schedule error"
            route_logger.error(
                "Master schedule error, request_id=%s, error_code=%s, "
                "error_message=%s",
                request_id,
                response.code,
                message,
            )
            kmonitor.report(
                AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
                1,
                {"error_code": str(response.code)},
            )
            raise FtRuntimeException(
                exception_type=exception_type,
                message=message,
            )

        role_addrs = [
            RoleAddr(
                role=_resolve_role_from_server_status(s),
                ip=s.server_ip,
                http_port=s.http_port,
                grpc_port=s.grpc_port,
            )
            for s in response.server_status
        ]
        return FlexlbResponse.ok(
            role_addrs,
            enqueued_by_master=response.enqueued_by_master,
        )

    @staticmethod
    def _extract_api_key(input: GenerateInput) -> str:
        headers = getattr(input, "headers", None)
        if not headers:
            return ""
        api_key = headers.get("x-api-key") or headers.get("api-key")
        if api_key:
            return api_key
        auth = headers.get("authorization", "")
        if auth.startswith(BEARER_PREFIX):
            return auth[len(BEARER_PREFIX) :].strip()
        return ""
