"""Pure gRPC forward servicer for predict_v2.proto.

Supports multiple downstream addresses with round-robin selection across
an optional per-addr HTTP/2 channel pool.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import List, Optional

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc

_FORWARD_ENV_KEY = "DASH_SC_GRPC_FORWARD_ADDR"

# Per-addr HTTP/2 connection pool size. The forwarder multiplexes every
# outbound RPC over ``grpc.insecure_channel(addr)``, which is one HTTP/2
# connection. When that one connection fails (keepalive timeout, LBS RST,
# backend GC stall) *every* concurrent stream on it aborts simultaneously —
# the 14:19 cascade (22 × 1ms ``Exception iterating requests!`` inside 100ms)
# is the exact signature. Opening N independent channels per addr partitions
# the failure domain: one connection going bad only blasts ~1/N of in-flight
# RPCs. Round-robin picks a channel per RPC — HTTP/2 multiplexing still
# handles in-channel concurrency, the pool is about fault isolation, not
# bandwidth.
#
# Default 1 is backward-compatible: set
# ``DASH_SC_GRPC_FORWARD_CHANNELS_PER_ADDR=N`` in deployment to enable. Pool
# size is decoupled from ``DashScGrpcConfig.max_server_workers`` — that
# controls inbound concurrency, this controls outbound failure domains. A
# single server worker reuses the whole pool across the lifetime of its RPC.
_CHANNELS_PER_ADDR_ENV_KEY = "DASH_SC_GRPC_FORWARD_CHANNELS_PER_ADDR"
_DEFAULT_CHANNELS_PER_ADDR = 1

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


def _parse_channels_per_addr(env_value: str) -> int:
    """Parse ``DASH_SC_GRPC_FORWARD_CHANNELS_PER_ADDR`` with safe defaults.

    Non-integer / non-positive values fall back to ``_DEFAULT_CHANNELS_PER_ADDR``
    rather than raising — misconfiguration should not prevent server startup.
    """
    try:
        n = int(env_value.strip())
    except (TypeError, ValueError, AttributeError):
        return _DEFAULT_CHANNELS_PER_ADDR
    return n if n >= 1 else _DEFAULT_CHANNELS_PER_ADDR


class PureForwardServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """Pure transparent proxy with connection pool for multiple downstream addresses."""

    @staticmethod
    def has_forward_config() -> bool:
        """Check if forward addresses are configured via environment variable."""
        return bool(os.environ.get(_FORWARD_ENV_KEY, "").strip())

    def __init__(
        self,
        forward_addrs: Optional[List[str]] = None,
        channels_per_addr: Optional[int] = None,
    ):
        if forward_addrs is None:
            env_value = os.environ.get(_FORWARD_ENV_KEY, "")
            forward_addrs = _parse_forward_addrs(env_value)

        if not forward_addrs:
            raise ValueError(
                f"No forward addresses provided. Set {_FORWARD_ENV_KEY} env or pass forward_addrs."
            )

        if channels_per_addr is None:
            channels_per_addr = _parse_channels_per_addr(
                os.environ.get(_CHANNELS_PER_ADDR_ENV_KEY, "")
            )
        elif channels_per_addr < 1:
            channels_per_addr = _DEFAULT_CHANNELS_PER_ADDR

        self._forward_addrs = forward_addrs
        self._channels_per_addr = channels_per_addr
        self._channels: List[grpc.Channel] = []
        self._stubs: List[predict_v2_pb2_grpc.GRPCInferenceServiceStub] = []
        # Parallel list: for stub i, ``_stub_addr_idx[i]`` is the index into
        # ``_forward_addrs`` — lets ``_next_stub`` return the original addr
        # index without divmod arithmetic, which means swapping pool layout
        # (e.g. grouped vs interleaved) won't break caller assumptions.
        self._stub_addr_idx: List[int] = []

        for addr_i, addr in enumerate(forward_addrs):
            for _ in range(channels_per_addr):
                channel = grpc.insecure_channel(addr, options=_FORWARD_CHANNEL_OPTS)
                stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
                self._channels.append(channel)
                self._stubs.append(stub)
                self._stub_addr_idx.append(addr_i)

        # Round-robin cursor. A Lock makes concurrent ``_next_stub`` safe under
        # the gRPC thread pool. Contention is negligible (one increment per
        # RPC entry), but without it two workers racing ``__iadd__`` could
        # skew distribution under extreme load. ``itertools.cycle`` would be
        # simpler but is not thread-safe.
        self._rr_idx = 0
        self._rr_lock = threading.Lock()

        logging.info(
            "[DashScGrpc] PureForwardServicer initialized: %d addresses × %d channels/addr = %d total stubs: %s",
            len(forward_addrs),
            channels_per_addr,
            len(self._stubs),
            forward_addrs,
        )

    def _next_stub(
        self,
    ) -> tuple[Optional[predict_v2_pb2_grpc.GRPCInferenceServiceStub], int]:
        """Pick the next (stub, addr_index_in_forward_addrs) via round-robin.

        Returns ``(None, -1)`` when the pool is empty — this happens during
        the ``server.stop(grace)`` → ``servicer.close()`` shutdown window,
        when a stray RPC already dispatched by grpcio reaches the handler
        after ``close()`` has cleared ``_stubs``. Without this guard the
        empty-pool modulo raises ``ZeroDivisionError`` and the access log
        shows ``UNKNOWN_ZeroDivisionError``; the caller translates the
        ``None`` return into ``UNAVAILABLE`` instead — correct gRPC
        semantics and a bounded ``error_code`` bucket on Grafana.

        Otherwise returns the index into ``self._forward_addrs`` (not
        ``self._stubs``); the stub may correspond to any of the
        ``channels_per_addr`` connections pointed at that addr.
        """
        with self._rr_lock:
            if not self._stubs:
                return None, -1
            i = self._rr_idx
            self._rr_idx = (self._rr_idx + 1) % len(self._stubs)
        return self._stubs[i], self._stub_addr_idx[i]

    def close(self):
        for channel in self._channels:
            if channel is not None:
                channel.close()
        self._channels.clear()
        self._stubs.clear()

    def ModelStreamInfer(self, request_iterator, context):
        stub, idx = self._next_stub()
        if stub is None:
            # Shutdown race: ``close()`` cleared ``_stubs`` after grpcio had
            # already dispatched this RPC. ``context.abort`` sends the client
            # a proper UNAVAILABLE + message, raises ``grpc.RpcError`` here
            # so the handler unwinds cleanly, and the interceptor's
            # ``_classify_rpc_exception`` picks the code up as
            # ``UNAVAILABLE`` (bounded ``error_code`` bucket) instead of
            # ``UNKNOWN_ZeroDivisionError``.
            context.abort(grpc.StatusCode.UNAVAILABLE, "forwarder shutting down")
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

        # Propagate client-sent metadata to the downstream stub so that
        # correlation headers (``x-dashscope-request-id`` / ``x-request-id``
        # / ``traceparent`` / …) travel end-to-end. Without this the backend
        # frontend's access log has no way to link a ``req_count=0`` RPC to
        # the upstream dashscope-serving request that provoked it.
        try:
            md = context.invocation_metadata() or ()
        except Exception:
            md = ()

        upstream_iter = stub.ModelStreamInfer(request_iterator, metadata=md)

        # Cancel propagation mirrors ``dash_sc/service.py::_register_cancel_callback``:
        # grpc-python does NOT auto-cancel a streaming call when the consumer
        # stops iterating, and the server thread is blocked inside ``next(it)``
        # so it cannot observe the inbound cancel on its own. ``add_callback``
        # fires on a grpcio-internal thread the moment the inbound RPC
        # terminates (cancel / deadline / normal close) and cancels the
        # downstream stub call there, so rtp-llm stops burning GPU on a
        # client that is already gone. Without this the forward->rtp-llm RPC
        # keeps running long after the chat side has cut (observed: 567s
        # tail with 2 tokens).
        def _cancel_downstream() -> None:
            try:
                upstream_iter.cancel()
            except Exception:
                pass

        try:
            context.add_callback(_cancel_downstream)
        except Exception:
            pass

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
