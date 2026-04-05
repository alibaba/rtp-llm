"""ActionCache gRPC client for REAPI v2.

Provides GetActionResult / UpdateActionResult RPCs for the test result cache.
Since we can't regenerate _pb2 stubs without protoc, this module constructs
the two trivial request messages (GetActionResultRequest, UpdateActionResultRequest)
using raw protobuf wire format and reuses the existing ActionResult from _pb2.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import grpc

from . import remote_execution_pb2 as re_pb2

log = logging.getLogger(__name__)

# Protobuf wire-format helpers for constructing request messages.
# These avoid needing generated code for the two simple request types.

def _encode_varint(value: int) -> bytes:
    pieces: list = []
    while value > 0x7F:
        pieces.append((value & 0x7F) | 0x80)
        value >>= 7
    pieces.append(value & 0x7F)
    return bytes(pieces)


def _encode_length_delimited(field_number: int, data: bytes) -> bytes:
    tag = _encode_varint((field_number << 3) | 2)
    return tag + _encode_varint(len(data)) + data


def _build_get_action_result_request(
    instance_name: str, action_digest: re_pb2.Digest
) -> bytes:
    """Serialize GetActionResultRequest {instance_name=1, action_digest=2}."""
    parts = b""
    if instance_name:
        parts += _encode_length_delimited(1, instance_name.encode("utf-8"))
    parts += _encode_length_delimited(2, action_digest.SerializeToString())
    return parts


def _build_update_action_result_request(
    instance_name: str,
    action_digest: re_pb2.Digest,
    action_result: re_pb2.ActionResult,
) -> bytes:
    """Serialize UpdateActionResultRequest {instance_name=1, action_digest=2, action_result=3}."""
    parts = b""
    if instance_name:
        parts += _encode_length_delimited(1, instance_name.encode("utf-8"))
    parts += _encode_length_delimited(2, action_digest.SerializeToString())
    parts += _encode_length_delimited(3, action_result.SerializeToString())
    return parts


GRPC_MAX_MSG_SIZE = 16 * 1024 * 1024

_AC_SERVICE = "build.bazel.remote.execution.v2.ActionCache"
_GET_METHOD = f"/{_AC_SERVICE}/GetActionResult"
_UPDATE_METHOD = f"/{_AC_SERVICE}/UpdateActionResult"


class ActionCacheClient:
    """Thin gRPC client for the REAPI v2 ActionCache service.

    Falls back gracefully when the server does not implement ActionCache
    (returns UNIMPLEMENTED) by returning None / False and logging a warning.
    """

    def __init__(
        self,
        endpoint: str,
        metadata: Optional[List[tuple]] = None,
        *,
        instance_name: str = "",
    ):
        addr = endpoint.replace("grpc://", "")
        self.channel = grpc.insecure_channel(
            addr,
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MSG_SIZE),
                ("grpc.max_receive_message_length", GRPC_MAX_MSG_SIZE),
            ],
        )
        self.metadata = metadata or []
        self.instance_name = instance_name
        self._unimplemented_warned = False

        self._get_rpc = self.channel.unary_unary(
            _GET_METHOD,
            request_serializer=lambda req: req,
            response_deserializer=re_pb2.ActionResult.FromString,
        )
        self._update_rpc = self.channel.unary_unary(
            _UPDATE_METHOD,
            request_serializer=lambda req: req,
            response_deserializer=re_pb2.ActionResult.FromString,
        )

    def _warn_unimplemented(self, method: str) -> None:
        if not self._unimplemented_warned:
            log.warning(
                "REAPI server does not implement ActionCache.%s — "
                "test cache will use local-only mode",
                method,
            )
            self._unimplemented_warned = True

    def get(self, action_digest: re_pb2.Digest) -> Optional[re_pb2.ActionResult]:
        """GetActionResult: returns ActionResult or None on NOT_FOUND / error."""
        req_bytes = _build_get_action_result_request(
            self.instance_name, action_digest
        )
        try:
            return self._get_rpc(req_bytes, metadata=self.metadata, timeout=30)
        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.NOT_FOUND:
                return None
            if code == grpc.StatusCode.UNIMPLEMENTED:
                self._warn_unimplemented("GetActionResult")
                return None
            log.warning("ActionCache.GetActionResult failed: %s %s", code, e.details())
            return None

    def update(
        self,
        action_digest: re_pb2.Digest,
        action_result: re_pb2.ActionResult,
    ) -> bool:
        """UpdateActionResult: returns True on success, False on error."""
        req_bytes = _build_update_action_result_request(
            self.instance_name, action_digest, action_result
        )
        try:
            self._update_rpc(req_bytes, metadata=self.metadata, timeout=30)
            return True
        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.UNIMPLEMENTED:
                self._warn_unimplemented("UpdateActionResult")
                return False
            log.warning(
                "ActionCache.UpdateActionResult failed: %s %s", code, e.details()
            )
            return False

    def close(self) -> None:
        try:
            self.channel.close()
        except Exception:
            pass
