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

from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc

_FORWARD_ENV_KEY = "DASH_SC_GRPC_FORWARD_ADDR"

# HTTP/2 keepalive for the forwarder -> real frontend channel. Production
# topology places an LBS between the two with a 100s idle timeout; without
# keepalive, any RPC whose downstream stalls >100s (cold prefill, PD handoff,
# backend GC pause, etc.) gets its TCP connection RST by the LBS and the
# access log records a mystifying ``UNAVAILABLE / recvmsg:Connection reset
# by peer / resp_count=0``. Sending a PING every 30s keeps the LBS idle
# counter pinned at ~0 so only real downstream failure shows up.
#
# ``permit_without_calls=0`` means PINGs only fire while an RPC is active —
# idle channels don't burn anything. ``max_pings_without_data=0`` removes
# the grpcio default cap of 2 consecutive data-less PINGs, which would
# otherwise GOAWAY us after 60s of PING-only traffic on a slow stream.
_FORWARD_CHANNEL_OPTS: list[tuple[str, int]] = [
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 0),
    ("grpc.http2.max_pings_without_data", 0),
]


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

        for addr in forward_addrs:
            channel = grpc.insecure_channel(addr, options=_FORWARD_CHANNEL_OPTS)
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
            self._channels.append(channel)
            self._stubs.append(stub)

        logging.info(
            "[DashScGrpc] PureForwardServicer initialized with %d addresses: %s",
            len(forward_addrs),
            forward_addrs,
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

        # Grab the access-log aggregate (installed by the access-log
        # interceptor) so forward-path diagnostics land inline on the access
        # log line — ``downstream_addr`` / ``downstream_resp_count`` /
        # ``buffered_stage`` are what lets an operator triage a
        # ``resp_count=0`` RPC without cross-grepping the forwarder debug log.
        agg = getattr(context, "_dash_sc_access_agg", None)
        if agg is not None:
            try:
                agg.downstream_addr = addr
            except Exception:
                pass

        upstream_iter = stub.ModelStreamInfer(request_iterator)
        if agg is not None:

            def counting_response_iter():
                for resp in upstream_iter:
                    try:
                        agg.downstream_resp_count += 1
                    except Exception:
                        pass
                    yield resp

            downstream_iter = counting_response_iter()
        else:
            downstream_iter = upstream_iter

        yield from self._buffered_iter(downstream_iter, agg)

    @staticmethod
    def _buffered_iter(downstream_iter, diag_agg=None):
        """Hold the first chunk until the second arrives, then yield both back-to-back.

        Smooths PD-disaggregation TPOT perception: in PD mode the gap between
        token 1 (emitted by prefill) and token 2 (emitted by decode, after KV cache
        handoff) is much larger than steady-state TPOT. Buffering delays token 1
        until token 2 is ready, so clients see a uniform inter-token cadence.

        ``diag_agg`` is the access-log aggregate (or ``None`` in tests); the
        stage labels it receives are what distinguish the three failure shapes
        at a glance on the access log line:

        - ``waiting_first`` + exception -> backend produced zero frames (issue
          is downstream / LBS, buffering is innocent);
        - ``waiting_second`` + exception -> backend produced token 1 and we
          blocked waiting for token 2; whether that token made it to the
          client follows:
        - ``flushed_first_on_exception`` -> buffered token 1 was still yielded
          to the client before re-raise (client at least saw partial output);
        - ``dropped_buffered_on_exception`` -> client went away first, token 1
          was lost — this is the "HoL blocking burned the only output" case.

        Exception handling contract:
        - downstream ends after 1 chunk -> flush buffered, return cleanly;
        - downstream errors at any point -> flush buffered (best-effort), then re-raise;
        - downstream yields 0 chunks -> return cleanly.
        """

        def _set_stage(s):
            if diag_agg is not None:
                try:
                    diag_agg.buffered_stage = s
                except Exception:
                    pass

        it = iter(downstream_iter)
        buffered = None
        _set_stage("waiting_first")
        try:
            try:
                buffered = next(it)
            except StopIteration:
                return
            _set_stage("waiting_second")
            try:
                second = next(it)
            except StopIteration:
                yield buffered
                buffered = None
                _set_stage("flushed_first")
                return
            yield buffered
            buffered = None
            yield second
            _set_stage("flushed_both")
            yield from it
        except GeneratorExit:
            # Consumer called ``gen.close()`` on the outer RPC generator —
            # yielding is forbidden after GeneratorExit (Python raises
            # ``RuntimeError: generator ignored GeneratorExit`` and prints
            # an "Exception ignored in:" traceback to stderr at GC time).
            # The buffered frame is lost; there is no wire to flush to.
            if buffered is not None:
                _set_stage("dropped_buffered_on_exception")
            raise
        except BaseException:
            if buffered is not None:
                try:
                    yield buffered
                    _set_stage("flushed_first_on_exception")
                except BaseException:
                    _set_stage("dropped_buffered_on_exception")
            raise
