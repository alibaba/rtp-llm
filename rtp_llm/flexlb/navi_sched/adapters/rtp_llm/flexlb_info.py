import asyncio
import time
from dataclasses import dataclass, field, fields
from typing import Any

import grpc

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
    CANCEL_REASON_CLIENT_CANCELLED,
    CANCEL_REASON_DEADLINE_EXCEEDED,
    FlexlbScheduleRequestPB,
)
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2_grpc import FlexlbServiceStub
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import GenerateInputPB
from rtp_llm.server.master_client import (
    FLEXLB_GRPC_PORT_OFFSET, FlexlbResponse, MasterClient, SUCCESS_CODE,
    _resolve_role_from_server_status,
)

from .RtpLLMAdapterUtils import split_address

NAVI_MASTER_GRPC_PORT_OFFSET = 3
_now = time.monotonic


@dataclass(frozen=True)
class FlexlbInfo:
    ip: str
    port: int
    request: Any = field(repr=False)
    deadline: float
    metadata: Any = field(default=(), repr=False)
    _on_response: Any = field(default=None, repr=False)

    def __post_init__(self):
        if not self.ip or not 0 < self.port <= 65535 - NAVI_MASTER_GRPC_PORT_OFFSET:
            raise ValueError("flexlb_info requires an IP and a valid gRPC port")

    @property
    def address(self):
        return f"{self.ip}:{self.port}"

    @property
    def navi_master_address(self):
        return f"{self.ip}:{self.port + NAVI_MASTER_GRPC_PORT_OFFSET}"


@dataclass
class FlexlbPrefillResponse(FlexlbResponse):
    flexlb_response: Any = field(default=None, repr=False)
    schedule_error: Any = field(default=None, repr=False)

    def with_flexlb_result(self, parsed):
        if self.schedule_error is not None:
            raise self.schedule_error
        if not self.is_ok:
            return self
        self.result = parsed.result
        if self.result is not None:
            for status, wire in zip(self.result["server_status"],
                                    self.flexlb_response.server_status):
                if wire.HasField("engine_index"):
                    status["engine_index"] = wire.engine_index
                if wire.preempt_request_ids:
                    status["preempt_request_ids"] = list(wire.preempt_request_ids)
        return self


def _wrap(route, response, error=None):
    return FlexlbPrefillResponse(
        **{item.name: getattr(route, item.name) for item in fields(FlexlbResponse)},
        flexlb_response=response, schedule_error=error,
    )


def _deadline_error():
    return FtRuntimeException(
        ExceptionType.DEADLINE_EXCEEDED,
        "FlexLB/NaviSched scheduling deadline exceeded",
    )


async def _cancel_first_stage(info, response, request_id, reason):
    target = info.address
    if response.real_master_host:
        ip, port = split_address(response.real_master_host)
        target = f"{ip}:{port + FLEXLB_GRPC_PORT_OFFSET}"
    async with grpc.aio.insecure_channel(target) as channel:
        await MasterClient._best_effort_cancel(
            FlexlbServiceStub(channel), request_id, reason, metadata=info.metadata)


async def schedule_with_flexlb(client, info, *, block_cache_keys,
                               cache_key_block_size, input, request_id, input_pb):
    if input_pb is None:
        raise ValueError("FlexLB Prefill scheduling requires the complete request")
    if client.config.get_target_role() != RoleType.PREFILL:
        raise ValueError("FlexLB integration requires NaviSched to schedule only Prefill")
    remaining = info.deadline - _now()
    if remaining <= 0:
        raise _deadline_error()
    request = FlexlbScheduleRequestPB()
    request.CopyFrom(info.request)
    request.skip_prefill = True
    request.seq_len = len(input_pb.token_ids)
    request.ClearField("generate_input")
    response = await client.schedule_flexlb(
        info.address, request, remaining, info.metadata)
    if response.code != SUCCESS_CODE:
        if info._on_response is not None:
            info._on_response(response)
            await asyncio.sleep(0)
        return _wrap(FlexlbResponse.error_response(response.code, response.error_message),
                     response)

    original_addrs = input.generate_config.role_addrs
    reason = CANCEL_REASON_CLIENT_CANCELLED
    try:
        if info._on_response is not None:
            info._on_response(response)
            await asyncio.sleep(0)
        if response.enqueued_by_master or not response.HasField("prefill_delegation"):
            raise ValueError("FlexLB did not delegate Prefill without enqueueing")
        addresses = [
            RoleAddr(role=_resolve_role_from_server_status(wire), ip=wire.server_ip,
                     http_port=wire.http_port, grpc_port=wire.grpc_port)
            for wire in response.server_status
        ]
        if any(address.role in (RoleType.PREFILL, RoleType.PDFUSION)
               for address in addresses):
            raise ValueError("FlexLB selected Prefill in a delegated request")
        payload = GenerateInputPB()
        payload.CopyFrom(input_pb)
        payload.flexlb_schedule_response = response.SerializeToString()
        del payload.generate_config.role_addrs[:]
        for address in addresses:
            wire = payload.generate_config.role_addrs.add()
            wire.role = int(address.role)
            wire.role_str = address.role.name
            wire.ip = address.ip
            wire.http_port = address.http_port
            wire.grpc_port = address.grpc_port
        input.generate_config.role_addrs = addresses
        remaining_ms = int((info.deadline - _now()) * 1000)
        if remaining_ms <= 0:
            reason = CANCEL_REASON_DEADLINE_EXCEEDED
            raise _deadline_error()
        route = await client.get_backend_role_addrs(
            block_cache_keys=block_cache_keys,
            cache_key_block_size=cache_key_block_size,
            input=input, request_id=request_id, input_pb=payload,
            prompt_token_count=len(payload.token_ids), timeout_ms=remaining_ms,
            master_address=info.navi_master_address,
            master_biz_name=response.prefill_delegation.master_biz_name,
        )
        if not route.is_ok:
            await _cancel_first_stage(info, response, request_id, reason)
            input.generate_config.role_addrs = original_addrs
        return _wrap(route, response)
    except asyncio.CancelledError:
        input.generate_config.role_addrs = original_addrs
        await asyncio.shield(_cancel_first_stage(info, response, request_id, reason))
        raise
    except Exception as error:
        input.generate_config.role_addrs = original_addrs
        await _cancel_first_stage(info, response, request_id, reason)
        return _wrap(FlexlbResponse.error_response(
            ExceptionType.MASTER_NO_AVAILABLE_WORKER, str(error)), response, error)
