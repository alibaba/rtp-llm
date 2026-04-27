"""Pure gRPC forward servicer for predict_v2.proto.

Supports multiple downstream addresses with random selection.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import List, Optional

import grpc

from rtp_llm.bailian.proto import predict_v2_pb2_grpc

_FORWARD_ENV_KEY = "BAILIAN_GRPC_FORWARD_ADDR"
_LOG_DEBUG_ENV_KEY = "BAILIAN_GRPC_FORWARD_LOG_DEBUG"


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

    if env_value.startswith("["):
        try:
            addrs = json.loads(env_value)
            if isinstance(addrs, list):
                return [str(a).strip() for a in addrs if str(a).strip()]
        except json.JSONDecodeError:
            pass

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

        self._forward_addrs = forward_addrs
        self._channels: List[grpc.Channel] = []
        self._stubs: List[predict_v2_pb2_grpc.GRPCInferenceServiceStub] = []
        self._log_debug = os.environ.get(_LOG_DEBUG_ENV_KEY, "").lower() in ("true", "1", "on")

        for addr in forward_addrs:
            channel = grpc.insecure_channel(addr)
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
            self._channels.append(channel)
            self._stubs.append(stub)

        logging.info(
            "[BailianGrpc] PureForwardServicer initialized with %d addresses: %s, log_debug=%s",
            len(forward_addrs),
            forward_addrs,
            self._log_debug,
        )

    def _next_stub(self) -> tuple[predict_v2_pb2_grpc.GRPCInferenceServiceStub, int]:
        idx = random.randrange(len(self._stubs))
        return self._stubs[idx], idx

    def close(self):
        for channel in self._channels:
            if channel is not None:
                channel.close()
        self._channels.clear()
        self._stubs.clear()

    def ModelStreamInfer(self, request_iterator, context):
        stub, idx = self._next_stub()
        addr = self._forward_addrs[idx]

        if self._log_debug:
            req_count = 0
            resp_count = 0

            def logged_request_iterator():
                nonlocal req_count
                for req in request_iterator:
                    req_count += 1
                    logging.info(
                        "[BailianGrpc] Forward req #%d to %s: model=%s id=%s inputs=%d",
                        req_count,
                        addr,
                        req.model_name,
                        req.id,
                        len(req.inputs),
                    )
                    yield req

            upstream_iter = stub.ModelStreamInfer(logged_request_iterator())

            def logged_response_iter():
                nonlocal resp_count
                for resp in upstream_iter:
                    resp_count += 1
                    logging.info(
                        "[BailianGrpc] Forward resp #%d from %s: error=%s outputs=%d",
                        resp_count,
                        addr,
                        resp.error_message or "none",
                        len(resp.infer_response.outputs),
                    )
                    yield resp

            downstream_iter = logged_response_iter()
        else:
            downstream_iter = stub.ModelStreamInfer(request_iterator)

        yield from self._buffered_iter(downstream_iter)

    @staticmethod
    def _buffered_iter(downstream_iter):
        """Hold the first chunk until the second arrives, then yield both back-to-back.

        Smooths PD-disaggregation TPOT perception: in PD mode the gap between
        token 1 (emitted by prefill) and token 2 (emitted by decode, after KV cache
        handoff) is much larger than steady-state TPOT. Buffering delays token 1
        until token 2 is ready, so clients see a uniform inter-token cadence.

        Exception handling contract:
        - downstream ends after 1 chunk -> flush buffered, return cleanly;
        - downstream errors at any point -> flush buffered (best-effort), then re-raise;
        - downstream yields 0 chunks -> return cleanly.
        """
        it = iter(downstream_iter)
        buffered = None
        try:
            try:
                buffered = next(it)
            except StopIteration:
                return
            try:
                second = next(it)
            except StopIteration:
                yield buffered
                buffered = None
                return
            yield buffered
            buffered = None
            yield second
            yield from it
        except BaseException:
            if buffered is not None:
                try:
                    yield buffered
                except BaseException:
                    pass
            raise