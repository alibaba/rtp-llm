"""gRPC→gRPC reverse-proxy servicer for predict_v2.proto (grpc.aio).

Supports multiple backend addresses with round-robin selection across an
optional per-addr HTTP/2 channel pool. Runs on grpc.aio — every servicer
method is a coroutine or async generator so the whole proxy path stays on
a single asyncio event loop.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import List, Optional

import grpc

from rtp_llm.dash_sc.codec import build_parameter_error_response, parse_sampling_params
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc


def _is_stream_done(resp: predict_v2_pb2.ModelStreamInferResponse) -> bool:
    """True when the proxy should stop forwarding after this frame."""
    if resp.error_message:
        return True
    infer = resp.infer_response
    for i, out in enumerate(infer.outputs):
        if out.name == "finished" and i < len(infer.raw_output_contents):
            return infer.raw_output_contents[i] == b"\x01"
    return False


_FORWARD_ENV_KEY = "DASH_SC_GRPC_FORWARD_ADDR"

# Per-addr HTTP/2 connection pool size. The forwarder multiplexes every
# outbound RPC over ``grpc.aio.insecure_channel(addr)``, which is one HTTP/2
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
# ``DASH_SC_GRPC_FORWARD_CHANNELS_PER_ADDR=N`` in deployment to enable.
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


def _invalid_max_new_tokens_message(request) -> str | None:
    sampling = parse_sampling_params(request)
    if sampling.max_new_tokens > 0:
        return None
    param_name = "max_completion_tokens" if getattr(
        sampling, "max_new_tokens_from_completion_alias", False
    ) else "max_new_tokens"
    return (
        f"invalid {param_name}: {sampling.max_new_tokens}; "
        "must be greater than 0"
    )


class DashScProxyServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """Pure transparent proxy (grpc.aio) with a channel pool across downstream addrs."""

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
        self._channels: List[grpc.aio.Channel] = []
        self._stubs: List[predict_v2_pb2_grpc.GRPCInferenceServiceStub] = []
        # Parallel list: for stub i, ``_stub_addr_idx[i]`` is the index into
        # ``_forward_addrs`` — lets ``_next_stub`` return the original addr
        # index without divmod arithmetic, which means swapping pool layout
        # (e.g. grouped vs interleaved) won't break caller assumptions.
        self._stub_addr_idx: List[int] = []

        for addr_i, addr in enumerate(forward_addrs):
            for _ in range(channels_per_addr):
                channel = grpc.aio.insecure_channel(addr, options=_FORWARD_CHANNEL_OPTS)
                stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
                self._channels.append(channel)
                self._stubs.append(stub)
                self._stub_addr_idx.append(addr_i)

        # Round-robin cursor. Even under a single-loop aio server, ``_next_stub``
        # is plain sync code so a Lock is cheap insurance against future
        # multi-worker topologies that share the servicer across threads.
        self._rr_idx = 0
        self._rr_lock = threading.Lock()

        logging.info(
            "[DashScGrpc] DashScProxyServicer initialized: %d addresses × %d channels/addr = %d total stubs: %s",
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

    async def close(self) -> None:
        """Close all aio channels. Safe to call multiple times."""
        for channel in self._channels:
            if channel is None:
                continue
            try:
                await channel.close()
            except Exception as e:
                logging.warning("[DashScGrpc] forward channel close failed: %s", e)
        self._channels.clear()
        self._stubs.clear()

    async def ModelStreamInfer(self, request_iterator, context):
        # Grab the access-log aggregate (installed by the access-log
        # interceptor) so proxy-path diagnostics land inline on the access
        # log line — ``backend_addr`` / ``backend_resp_count`` /
        # ``buffered_stage`` are what lets an operator triage a
        # ``resp_count=0`` RPC without cross-grepping the proxy debug log.
        agg = getattr(context, "_dash_sc_access_agg", None)

        request_iter = request_iterator.__aiter__()
        try:
            first_request = await request_iter.__anext__()
        except StopAsyncIteration:
            return

        invalid_message = _invalid_max_new_tokens_message(first_request)
        if invalid_message is not None:
            logging.warning("[DashScGrpc] proxy parameter error: %s", invalid_message)
            try:
                aclose = getattr(request_iter, "aclose", None)
                if aclose is not None:
                    await aclose()
            except Exception:
                pass
            yield build_parameter_error_response(str(first_request.id), invalid_message)
            return

        stub, idx = self._next_stub()
        if stub is None:
            # Shutdown race: ``close()`` cleared ``_stubs`` after grpcio had
            # already dispatched this RPC. Translate it to UNAVAILABLE.
            await context.abort(grpc.StatusCode.UNAVAILABLE, "forwarder shutting down")
            return
        addr = self._forward_addrs[idx]
        if agg is not None:
            try:
                agg.backend_addr = addr
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

        async def validated_request_iter():
            yield first_request
            async for req in request_iter:
                yield req

        upstream_iter = stub.ModelStreamInfer(validated_request_iter(), metadata=md)

        # Cancel propagation under grpc.aio is implicit: when the inbound RPC
        # is cancelled by the client, the grpc.aio framework cancels the
        # handler coroutine; ``asyncio.CancelledError`` then unwinds through
        # the ``async for`` over ``upstream_iter`` and cancels the downstream
        # aio call automatically. No ``context.add_callback`` plumbing needed.

        if agg is not None:

            async def counting_response_iter():
                async for resp in upstream_iter:
                    try:
                        agg.backend_resp_count += 1
                    except Exception:
                        pass
                    yield resp

            downstream_iter = counting_response_iter()
        else:
            downstream_iter = upstream_iter

        # Explicit aclose() wrapping: when the RPC generator is aclose()'d by
        # grpc.aio (client disconnect) or athrow()'d by a handler, the
        # exception lands at our ``yield resp`` below — but Python does *not*
        # automatically close the inner ``_buffered_iter`` in that case. The
        # inner would leak suspended at its own yield and its
        # ``except GeneratorExit`` (responsible for the
        # ``dropped_buffered_on_exception`` stage) would never fire. Wrapping
        # in try/finally + aclose() makes the close deterministic.
        #
        # The outer ``finally`` then tears down the downstream call itself.
        # Without that, when ``_buffered_iter`` returns after detecting the
        # finished frame the inner generators and the underlying grpc.aio.Call
        # linger until Python's async-gen finalizer hook fires — which only
        # happens after grpc.aio GCs this handler coroutine. The gap between
        # ``finished=True`` and the actual downstream cancel is what lets the
        # backend (and access logs) record a clean stream end as a late
        # client-cancel race.
        buffered = self._buffered_iter(downstream_iter, agg)
        try:
            try:
                async for resp in buffered:
                    try:
                        yield resp
                    except BaseException:
                        if access_record is not None:
                            try:
                                if access_record.buffered_stage == "waiting_second":
                                    access_record.buffered_stage = (
                                        "dropped_buffered_on_exception"
                                    )
                            except Exception:
                                pass
                        raise
            finally:
                try:
                    await buffered.aclose()
                except Exception:
                    pass
        finally:
            # Safety net. ``_close_downstream`` is also exposed as a public-ish
            # static method so any future code path that wants to tear down
            # the downstream stream early (e.g. on a fatal upstream signal)
            # can call it explicitly — both aclose and cancel are idempotent,
            # so re-invocation by this ``finally`` is a no-op.
            await self._close_downstream(downstream_iter, upstream_iter)

    @staticmethod
    async def _close_downstream(downstream_iter, upstream_iter) -> None:
        """Deterministically tear down the downstream stub call.

        Call this when you want the proxy -> backend stream closed *now*
        (e.g. you've observed an end-of-stream marker on a wire format the
        proxy doesn't understand, or you're aborting forwarding for any
        other reason). The default ``ModelStreamInfer`` flow already calls
        this from its ``finally`` clause; this method is exposed so callers
        that don't want to rely on the implicit finally — or that want the
        close to happen *before* further awaits in their own code path —
        can invoke it directly.

        Closes in two layers:

        - If a wrapper async generator was placed between ``upstream_iter``
          and the buffered iterator (currently ``counting_response_iter``),
          ``aclose()`` is awaited on it first. This injects
          ``GeneratorExit`` into the wrapper synchronously, letting any
          stage labels / finally hooks run.
        - ``cancel()`` is then invoked on ``upstream_iter``. For a real
          grpc.aio call this triggers RST_STREAM toward the backend
          immediately; without it the call lingers until Python's async-gen
          finalizer hook fires after grpc.aio releases the handler
          coroutine, which is the GC-latency tail that lets backends log a
          clean stream end as a late client-cancel race.

        Both operations are idempotent — safe to call multiple times and
        safe to call on an already-completed stream.
        """
        # Close any wrapping async generator first (e.g. counting wrapper).
        if downstream_iter is not upstream_iter:
            try:
                aclose = getattr(downstream_iter, "aclose", None)
                if aclose is not None:
                    await aclose()
            except Exception:
                pass
        # Cancel the underlying grpc.aio.Call. Sync, idempotent.
        try:
            cancel = getattr(upstream_iter, "cancel", None)
            if cancel is not None:
                cancel()
        except Exception:
            pass
        # Tests / non-grpc fakes expose ``aclose`` on the upstream iterator
        # (real grpc.aio calls don't); cover that path too.
        try:
            aclose = getattr(upstream_iter, "aclose", None)
            if aclose is not None:
                await aclose()
        except Exception:
            pass

    @staticmethod
    async def _buffered_iter(downstream_iter, diag_agg=None):
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

        it = downstream_iter.__aiter__()
        buffered = None
        _set_stage("waiting_first")
        try:
            try:
                buffered = await it.__anext__()
            except StopAsyncIteration:
                return
            _set_stage("waiting_second")
            try:
                second = await it.__anext__()
            except StopAsyncIteration:
                yield buffered
                buffered = None
                _set_stage("flushed_first")
                return
            yield buffered
            buffered = None
            yield second
            _set_stage("flushed_both")
            if _is_stream_done(second):
                return
            async for remaining in it:
                yield remaining
                if _is_stream_done(remaining):
                    return
        except GeneratorExit:
            # Consumer called ``aclose()`` on the outer RPC generator —
            # yielding is forbidden after GeneratorExit. The buffered frame
            # is lost; there is no wire to flush to.
            if buffered is not None:
                _set_stage("dropped_buffered_on_exception")
            raise
        except BaseException:
            if buffered is not None:
                _set_stage("dropped_buffered_on_exception")
                try:
                    yield buffered
                    _set_stage("flushed_first_on_exception")
                except BaseException:
                    _set_stage("dropped_buffered_on_exception")
            raise
