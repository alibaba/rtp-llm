"""FlexLB schedule client: request role addrs from master/slave via gRPC."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ErrorDetailsPB,
    GenerateInputPB,
    MultimodalHashRequestPB,
    MultimodalInputsPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
from rtp_llm.server.host_service import HostService
from rtp_llm.server.mm_cache_metadata import MAX_METADATA_BYTES, metadata_from_proto
from rtp_llm.server.request_headers import (
    dashscope_greennet_metadata,
    normalize_request_headers,
)
from rtp_llm.server.worker_status import _coerce_role_type
from rtp_llm.telemetry import attributes as trace_attrs
from rtp_llm.telemetry import start_client_span
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool

DEFAULT_REQUEST_TIMEOUT_SEC = 0.5
VIT_ROUTE_STALE_CODE = 8408

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
    result: Optional[Dict[str, Any]] = None

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
        self._vit_channels = GrpcHostChannelPool(
            options=[
                ("grpc.max_receive_message_length", MAX_METADATA_BYTES),
                ("grpc.max_send_message_length", MAX_METADATA_BYTES),
            ]
        )
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
        await self._vit_channels.close()

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
        trace_metadata = []
        try:
            channel = self._get_channel(target)
            stub = FlexlbServiceStub(channel)
            route_logger.debug(
                "gRPC Schedule sending, request_id=%s, proto_priority=%d",
                request_id,
                request_pb.priority,
            )
            client_span, trace_metadata = start_client_span(
                "rtp_llm.flexlb.schedule", target
            )
            if client_span is not None:
                # The platform indexes the internal ID's string form for span search.
                # Same contract as the generate_stream_call / fetch_response spans.
                client_span.set_attribute(trace_attrs.REQUEST_ID, str(request_id))
            try:
                response = await stub.Schedule(
                    request_pb,
                    timeout=timeout_s,
                    metadata=trace_metadata or None,
                )
            except BaseException as error:
                if client_span is not None:
                    if isinstance(error, grpc.aio.AioRpcError):
                        client_span.set_attribute(
                            trace_attrs.RPC_RESPONSE_STATUS_CODE, error.code().name
                        )
                        client_span.finish(error=error, error_type="RpcError")
                    elif isinstance(error, asyncio.CancelledError):
                        client_span.finish(error=error, error_type="Cancelled")
                    else:
                        client_span.finish(error=error, error_type="RpcError")
                raise
            if client_span is not None:
                # The transport succeeded whatever the business code says; the
                # sibling fetch_response CLIENT span already ships this key, so it
                # is known not to disturb the platform's token aggregation.
                client_span.set_attribute(
                    trace_attrs.RPC_RESPONSE_STATUS_CODE, grpc.StatusCode.OK.name
                )
                client_span.set_attribute(
                    trace_attrs.RTP_LLM_SCHEDULE_CODE, int(response.code)
                )
                # A rejecting business code on an otherwise successful transport is
                # still a failed schedule: the caller raises FtRuntimeException on
                # it (see the SUCCESS_CODE branch below). Closing this span as OK
                # would contradict the code just recorded above and hide the
                # rejection from status-based filtering on the CLIENT span.
                if int(response.code) != SUCCESS_CODE:
                    client_span.finish(error_type="FlexlbBusinessRejected")
                else:
                    client_span.finish()
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
                if not request_pb.vit_route_only:
                    await self._best_effort_cancel(
                        stub,
                        request_id,
                        CANCEL_REASON_DEADLINE_EXCEEDED,
                        trace_metadata,
                    )
                await self._close_channel(target)
                raise FtRuntimeException(
                    exception_type=ExceptionType.DEADLINE_EXCEEDED,
                    message=f"FlexLB schedule deadline exceeded for request {request_id}",
                ) from e
            await self._close_channel(target)
            return None
        except asyncio.CancelledError:
            if "stub" in locals() and not request_pb.vit_route_only:
                await self._best_effort_cancel(
                    stub, request_id, CANCEL_REASON_CLIENT_CANCELLED, trace_metadata
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
    async def _best_effort_cancel(
        stub, request_id: int, reason: int, metadata=None
    ) -> None:
        try:
            await stub.Cancel(
                FlexlbCancelRequestPB(request_id=request_id, reason=reason),
                timeout=1.0,
                # Retain the failed Schedule's ancestry even after its span ends.
                metadata=metadata or None,
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
        *,
        media_keys: Optional[List[str]] = None,
        selected_vit: Optional[Dict[str, Any]] = None,
        seq_len: Optional[int] = None,
        vit_only: bool = False,
    ) -> FlexlbResponse:
        """
        Resolve backend role addrs from FlexLB scheduler (master, then slave on connection failure).

        request_id is frontend-generated and only used for logging.
        Only connection_failed triggers slave retry and domain fallback.
        """
        master_addr = self.host_service.get_master_addr() if self.host_service else None
        if not master_addr:
            return FlexlbResponse.connection_failed_response()

        slave_addr = None
        if self.host_service:
            slave_addr = getattr(self.host_service, "get_slave_addr", lambda: None)()

        ttft_timeout_ms = getattr(
            input.generate_config, "ttft_timeout_ms", None
        ) or getattr(input.generate_config, "timeout_ms", None)
        if ttft_timeout_ms is None or ttft_timeout_ms <= 0:
            ttft_timeout_ms = self.master_config.master_default_timeout_ms
        timeout_s = ttft_timeout_ms / 1000.0 if ttft_timeout_ms > 0 else None

        gc = input.generate_config
        api_key = self._extract_api_key(input)
        priority = self._extract_priority(input)
        request_pb = FlexlbScheduleRequestPB(
            request_id=request_id,
            block_cache_keys=block_cache_keys,
            seq_len=input.prompt_length if seq_len is None else seq_len,
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
        request_pb.media_keys.extend(media_keys or [])
        request_pb.vit_route_only = vit_only
        if selected_vit is not None:
            for name in (
                "role",
                "server_ip",
                "http_port",
                "grpc_port",
                "group",
                "worker_instance",
                "worker_generation",
            ):
                value = selected_vit.get(name)
                if value is not None:
                    setattr(request_pb.selected_vit, name, value)
        if vit_only:
            timeout_s = min(timeout_s, 0.5) if timeout_s is not None else 0.5
        if input_pb is not None and not vit_only:
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
            # Only an unadmitted stale selection can be retried with the same request id.
            if (
                selected_vit is not None
                and response.code == VIT_ROUTE_STALE_CODE
                and not response.HasField("lifecycle")
            ):
                return FlexlbResponse.error_response(
                    response.code, response.error_message
                )
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
        result = FlexlbResponse.ok(
            role_addrs, enqueued_by_master=response.enqueued_by_master
        )
        result.result = {
            "server_status": [
                {
                    "role": s.role,
                    "server_ip": s.server_ip,
                    "http_port": s.http_port,
                    "grpc_port": s.grpc_port,
                    "group": s.group or None,
                    "worker_instance": s.worker_instance or None,
                    "worker_generation": s.worker_generation,
                }
                for s in response.server_status
            ]
        }
        return result

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

    async def get_vit_cache_metadata(
        self, address: RoleAddr, keys: List[str], input: Optional[GenerateInput] = None
    ):
        """Probe hashes, then submit only missing media when routing requires them."""
        started = time.monotonic()
        unique_keys = list(dict.fromkeys(keys))
        if input is not None:
            input.greennet_verified_vit = None
        metadata = await self._get_vit_metadata(
            address,
            MultimodalHashRequestPB(keys=unique_keys),
            DEFAULT_REQUEST_TIMEOUT_SEC,
            headers=input.headers if input is not None else None,
        )
        if input is None:
            return metadata
        entries = {
            e["key"]: e
            for e in (metadata or {}).get("entries", [])
            if isinstance(e, dict) and isinstance(e.get("key"), str)
        }
        missing = {
            key
            for key in unique_keys
            if not entries.get(key, {}).get(
                "hash_hit", entries.get(key, {}).get("hit", False)
            )
        }
        if not missing:
            self._record_greennet_approval(input, address, unique_keys, entries)
            return metadata

        from rtp_llm.cpp.model_rpc.model_rpc_client import iter_multimodal_inputs

        # The cache-hit probe carries no URLs. On a miss serialize each distinct
        # missing input once, without copying image/video data into embeddings.
        inputs = MultimodalInputsPB(request_id=input.request_id)
        submitted = set()
        for key, item in zip(
            keys, iter_multimodal_inputs(input, input.generate_config)
        ):
            if key in missing and key not in submitted:
                inputs.multimodal_inputs.add().CopyFrom(item)
                submitted.add(key)
        if submitted != missing:
            raise FtRuntimeException(
                ExceptionType.MM_PROCESS_ERROR, "Missing ViT submission inputs"
            )
        configured_timeout = input.generate_config.mm_timeout_ms
        if not configured_timeout or configured_timeout <= 0:
            configured_timeout = max(
                (
                    i.mm_preprocess_config.mm_timeout_ms
                    for i in input.mm_inputs
                    if i.mm_preprocess_config.mm_timeout_ms > 0
                ),
                default=120000,
            )
        limits = [configured_timeout]
        for name in ("ttft_timeout_ms", "timeout_ms"):
            limit = getattr(input.generate_config, name, None)
            if limit and limit > 0:
                limits.append(limit)
        remaining = min(limits) / 1000.0 - (time.monotonic() - started)
        if remaining <= 0:
            raise FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT, "ViT hash acquisition timed out"
            )
        filled = await self._get_vit_metadata(
            address,
            MultimodalHashRequestPB(
                keys=[key for key in unique_keys if key in missing],
                inputs=inputs,
                timeout_ms=max(1, int(remaining * 1000)),
            ),
            remaining,
            required=True,
            headers=input.headers,
        )
        if metadata and filled.get("worker_instance") != metadata.get(
            "worker_instance"
        ):
            raise FtRuntimeException(
                ExceptionType.ROUTE_ERROR, "ViT restarted during hash acquisition"
            )
        entries.update({e["key"]: e for e in filled.get("entries", [])})
        if any(
            not entries.get(key, {}).get(
                "hash_hit", entries.get(key, {}).get("hit", False)
            )
            for key in unique_keys
        ):
            raise FtRuntimeException(
                ExceptionType.MM_PROCESS_ERROR, "ViT returned incomplete feature hashes"
            )
        filled["entries"] = [entries[key] for key in unique_keys]
        self._record_greennet_approval(input, address, unique_keys, entries)
        return filled

    @staticmethod
    def _record_greennet_approval(
        input: GenerateInput,
        address: RoleAddr,
        keys: List[str],
        entries: Dict[str, Any],
    ) -> None:
        if keys and all(entries[key].get("greennet_passed") is True for key in keys):
            input.greennet_verified_vit = (address.ip, address.grpc_port, tuple(keys))

    async def _get_vit_metadata(
        self,
        address,
        request,
        timeout_sec,
        required=False,
        headers=None,
    ):
        started = time.monotonic()
        try:
            channel = await self._vit_channels.get(f"{address.ip}:{address.grpc_port}")
            response = await MultimodalRpcServiceStub(channel).GetMultimodalHashes(
                request,
                timeout=timeout_sec,
                metadata=dashscope_greennet_metadata(headers),
            )
            return metadata_from_proto(response)
        except grpc.RpcError as error:
            if required:
                code = (
                    ExceptionType.GENERATE_TIMEOUT
                    if error.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                    else ExceptionType.MM_PROCESS_ERROR
                )
                message = error.details() or "ViT hash acquisition failed"
                for key, value in error.trailing_metadata() or ():
                    if key == "grpc-status-details-bin":
                        details = ErrorDetailsPB.FromString(value)
                        code = ExceptionType(details.error_code)
                        message = details.error_message
                        break
                raise FtRuntimeException(code, message) from error
            route_logger.warning(
                "ViT hash probe unavailable, address=%s:%s, status=%s",
                address.ip,
                address.grpc_port,
                error.code(),
            )
            return None
        except (TimeoutError, OSError, ValueError) as error:
            if required:
                code = (
                    ExceptionType.GENERATE_TIMEOUT
                    if isinstance(error, TimeoutError)
                    else ExceptionType.MM_PROCESS_ERROR
                )
                raise FtRuntimeException(
                    code, f"ViT hash acquisition failed: {type(error).__name__}"
                ) from error
            route_logger.warning("ViT hash probe failed: %s", error)
            return None
        finally:
            route_logger.debug(
                "ViT hash RPC elapsed_ms=%.3f",
                (time.monotonic() - started) * 1000,
            )
