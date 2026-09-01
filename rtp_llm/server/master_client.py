"""FlexLB schedule client: request role addrs from master/slave via gRPC."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import grpc
import grpc.aio

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.config.py_config_modules import MasterConfig
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
    CANCEL_REASON_CLIENT_CANCELLED,
    CANCEL_REASON_DEADLINE_EXCEEDED,
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
    only connection_failed triggers slave retry and domain fallback.
    """

    role_addrs: Optional[List[RoleAddr]] = None
    connection_failed: bool = False
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    admission_reject_reason: AdmissionRejectReason = AdmissionRejectReason.UNSPECIFIED
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
            admission_reject_reason=AdmissionRejectReason.UNSPECIFIED,
            enqueued_by_master=enqueued_by_master,
        )

    @classmethod
    def error_response(
        cls,
        error_code: int,
        error_message: Optional[str] = None,
        admission_reject_reason: AdmissionRejectReason = (
            AdmissionRejectReason.UNSPECIFIED
        ),
    ) -> "FlexlbResponse":
        """Scheduler returned error (e.g. non-200 body). No slave retry / no domain fallback."""
        return cls(
            role_addrs=None,
            connection_failed=False,
            error_code=error_code,
            error_message=error_message,
            admission_reject_reason=admission_reject_reason,
            enqueued_by_master=False,
        )

    @classmethod
    def connection_failed_response(cls) -> "FlexlbResponse":
        """No response (connection/timeout). Triggers slave retry and domain fallback."""
        return cls(
            role_addrs=None,
            connection_failed=True,
            error_code=None,
            error_message=None,
            admission_reject_reason=AdmissionRejectReason.UNSPECIFIED,
            enqueued_by_master=False,
        )


def _admission_reject_reason_from_response(response) -> AdmissionRejectReason:
    """Read field 9 without interpreting scheduler diagnostic text.

    Old peers that do not yet send the field naturally yield UNSPECIFIED.
    Preserve an unknown numeric value as the local INVALID sentinel so the
    centralized Dash contract cannot mistake it for a legal UNSPECIFIED value.
    """

    raw_reason = getattr(response, "admission_reject_reason", 0)
    try:
        return AdmissionRejectReason(int(raw_reason))
    except (TypeError, ValueError):
        route_logger.error("Unknown FlexLB admission rejection reason: %r", raw_reason)
        return AdmissionRejectReason.INVALID


class MasterClient:
    """Client for FlexLB schedule gRPC API (master and optional slave)."""

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
    ):
        """Send gRPC schedule request. Returns proto response on success, None on transport failure."""
        target = self._get_grpc_target(addr)
        start = time.time()
        try:
            channel = self._get_channel(target)
            stub = FlexlbServiceStub(channel)
            route_logger.debug(
                "gRPC Schedule sending, request_id=%s, proto_priority=%d",
                request_id,
                request_pb.priority,
            )
            response = await stub.Schedule(request_pb, timeout=timeout_s)
            return response
        except grpc.aio.AioRpcError as e:
            elapsed = time.time() - start
            route_logger.error(
                "gRPC schedule failed, addr=%s, request_id=%s, status=%s, detail=%s, elapsed=%.3fs",
                addr,
                request_id,
                e.code(),
                e.details(),
                elapsed,
            )
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                await self._best_effort_cancel(
                    stub, request_id, CANCEL_REASON_DEADLINE_EXCEEDED
                )
                await self._close_channel(target)
                raise FtRuntimeException(
                    exception_type=ExceptionType.DEADLINE_EXCEEDED,
                    message=f"FlexLB schedule deadline exceeded for request {request_id}",
                ) from e
            await self._close_channel(target)
            return None
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
            return None

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
        seq_len_hint: Optional[int] = None,
    ) -> FlexlbResponse:
        """
        Resolve backend role addrs from FlexLB scheduler (master, then slave on connection failure).

        request_id is frontend-generated and only used for logging.
        Only connection_failed triggers slave retry and domain fallback.
        seq_len_hint overrides the reported seq_len when one routing call stands in for
        more work than this single input — a batch routed as one scheduling unit reports
        its aggregate prompt length so the master's load accounting sees the true weight.
        """
        master_addr = self.host_service.get_master_addr() if self.host_service else None
        if not master_addr:
            return FlexlbResponse.connection_failed_response()

        slave_addr = None
        if self.host_service:
            slave_addr = self.host_service.get_slave_addr()

        ttft_timeout_ms = (
            input.generate_config.ttft_timeout_ms
            or input.generate_config.timeout_ms
        )
        if ttft_timeout_ms is None or ttft_timeout_ms <= 0:
            ttft_timeout_ms = self.master_config.master_default_timeout_ms
        timeout_s = ttft_timeout_ms / 1000.0 if ttft_timeout_ms > 0 else None

        gc = input.generate_config
        api_key = self._extract_api_key(input)
        priority = self._extract_priority(input)
        request_pb = FlexlbScheduleRequestPB(
            request_id=request_id,
            block_cache_keys=block_cache_keys,
            seq_len=(
                seq_len_hint
                if seq_len_hint is not None
                else input.prompt_length
            ),
            generate_timeout=ttft_timeout_ms,
            request_time_ms=int(time.time() * 1000),
            max_new_tokens=gc.max_new_tokens,
            num_beams=gc.num_beams,
            force_disable_sp_run=gc.force_disable_sp_run,
            model="engine_service",
            api_key=api_key,
            cache_key_block_size=cache_key_block_size,
            priority=priority,
        )
        if input_pb is not None:
            request_pb.generate_input = input_pb.SerializeToString()

        response = await self._send_schedule_request(
            master_addr, request_pb, timeout_s, request_id
        )

        if response is None and slave_addr:
            route_logger.info(
                "Master connection failed, retrying slave, slave=%s, request_id=%s",
                slave_addr,
                request_id,
            )
            response = await self._send_schedule_request(
                slave_addr, request_pb, timeout_s, request_id
            )

        if response is None:
            return FlexlbResponse.connection_failed_response()

        self.latest_queue_length = response.queue_length

        if response.code != SUCCESS_CODE:
            admission_reject_reason = _admission_reject_reason_from_response(response)
            try:
                exception_type = ExceptionType(response.code)
            except ValueError:
                exception_type = ExceptionType.MASTER_NO_AVAILABLE_WORKER
            message = response.error_message or "master schedule error"
            route_logger.error(
                "Master schedule error, request_id=%s, error_code=%s, "
                "error_message=%s, admission_reject_reason=%s",
                request_id,
                response.code,
                message,
                admission_reject_reason.name,
            )
            kmonitor.report(
                AccMetrics.MASTER_ROUTE_ERROR_QPS_METRIC,
                1,
                {"error_code": str(response.code)},
            )
            raise FtRuntimeException(
                exception_type=exception_type,
                message=message,
                admission_reject_reason=admission_reject_reason,
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

    @staticmethod
    def _extract_priority(input: GenerateInput) -> int:
        """QoS priority from x-dashscope-inner-qos-level header; returns 50
        (default priority) when the header is absent so FlexLB participates in
        Auto-TPM scheduling instead of opting out via NO_PRIORITY. Pure
        passthrough, no range validation here."""
        # 1. Try GenerateInput.headers (available when enqueue runs in the
        #    same process that received the HTTP request).
        headers = getattr(input, "headers", None)
        if headers:
            value = headers.get("x-dashscope-inner-qos-level")
            if value is not None:
                try:
                    return int(str(value).strip())
                except (TypeError, ValueError):
                    pass  # fall through to generate_config fallback
        # 2. Fallback: generate_config.qos_priority survives IPC to the
        #    dash_sc enqueue loop where GenerateInput.headers may be absent.
        gc = getattr(input, "generate_config", None)
        if gc is not None:
            qos_priority = getattr(gc, "qos_priority", None)
            if qos_priority is not None and qos_priority > 0:
                return qos_priority
        return 50
