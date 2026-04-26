"""Pure gRPC forward servicer for predict_v2.proto.

Supports multiple downstream addresses with round-robin connection pool.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import List, Optional

import grpc

from rtp_llm.bailian.proto import predict_v2_pb2_grpc
from rtp_llm.utils.util import AtomicCounter

_FORWARD_ENV_KEY = "BAILIAN_GRPC_FORWARD_ADDR"


def _parse_forward_addrs(env_value: str) -> List[str]:
    """Parse forward addresses from env value.

    Supports formats:
    - Single address: "10.0.0.1:8096"
    - Comma separated: "10.0.0.1:8096,10.0.0.2:8096"
    - JSON array: '["10.0.0.1:8096", "10.0.0.2:8096"]'
    """
    env_value = env_value.strip()
    if not env_value:
        return []

    # Try JSON array format
    if env_value.startswith("["):
        try:
            addrs = json.loads(env_value)
            if isinstance(addrs, list):
                return [str(a).strip() for a in addrs if str(a).strip()]
        except json.JSONDecodeError:
            pass

    # Comma separated format
    return [a.strip() for a in env_value.split(",") if a.strip()]


class PureForwardServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """Pure transparent proxy with connection pool for multiple downstream addresses."""

    @staticmethod
    def has_forward_config() -> bool:
        """Check if forward addresses are configured via environment variable."""
        return bool(os.environ.get(_FORWARD_ENV_KEY, "").strip())

    def __init__(self, forward_addrs: Optional[List[str]] = None):
        if forward_addrs is None:
            env_value = os.environ.get(_FORWARD_ENV_KEY, "")
            forward_addrs = _parse_forward_addrs(env_value)

        if not forward_addrs:
            raise ValueError(
                f"No forward addresses provided. Set {_FORWARD_ENV_KEY} env or pass forward_addrs."
            )

        self._channels: List[grpc.Channel] = []
        self._stubs: List[predict_v2_pb2_grpc.GRPCInferenceServiceStub] = []
        self._lock = threading.Lock()
        self._counter = AtomicCounter()

        for addr in forward_addrs:
            channel = grpc.insecure_channel(addr)
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
            self._channels.append(channel)
            self._stubs.append(stub)

        logging.info(
            "[BailianGrpc] PureForwardServicer initialized with %d addresses: %s",
            len(forward_addrs),
            forward_addrs,
        )

    def _next_stub(self) -> predict_v2_pb2_grpc.GRPCInferenceServiceStub:
        """Round-robin selection of downstream stub."""
        idx = self._counter.increment() % len(self._stubs)
        return self._stubs[idx]

    def close(self):
        with self._lock:
            for channel in self._channels:
                if channel is not None:
                    channel.close()
            self._channels.clear()
            self._stubs.clear()

    def ModelStreamInfer(self, request_iterator, context):
        stub = self._next_stub()
        for resp in stub.ModelStreamInfer(request_iterator):
            yield resp