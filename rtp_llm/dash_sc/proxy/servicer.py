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
from typing import List, Optional

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.dash_sc.proxy.access_record import ForwardAccessRecord
from rtp_llm.dash_sc.proxy.channel_pool import ForwardChannelPool


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

# Age-based channel recycling. The forward target is an L4 PVL VIP; a long-lived
# HTTP/2 connection sticks to one backend machine and never rebalances, so
# machines added behind the VIP later get no traffic. Periodically retiring an
# aged channel forces a reconnect through the VIP, which hands us the current
# machine set. Default 0 = OFF (pool built once, never recycled — unchanged
# behavior). Set ``DASH_SC_GRPC_FORWARD_CHANNEL_MAX_AGE_MS=N`` to enable.
_CHANNEL_MAX_AGE_MS_ENV_KEY = "DASH_SC_GRPC_FORWARD_CHANNEL_MAX_AGE_MS"
_CHANNEL_RECYCLE_INTERVAL_MS_ENV_KEY = (
    "DASH_SC_GRPC_FORWARD_CHANNEL_RECYCLE_INTERVAL_MS"
)
_CHANNEL_MAX_RECYCLE_PER_TICK_ENV_KEY = (
    "DASH_SC_GRPC_FORWARD_CHANNEL_MAX_RECYCLE_PER_TICK"
)

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


def _parse_int_env(env_value: str, default: int) -> int:
    """Parse a non-negative int env value, falling back to ``default``.

    Misconfiguration must not prevent startup, so any non-integer / negative
    value silently uses ``default``.
    """
    try:
        n = int(env_value.strip())
    except (TypeError, ValueError, AttributeError):
        return default
    return n if n >= 0 else default


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

        max_age_ms = _parse_int_env(os.environ.get(_CHANNEL_MAX_AGE_MS_ENV_KEY, ""), 0)
        recycle_interval_ms = _parse_int_env(
            os.environ.get(_CHANNEL_RECYCLE_INTERVAL_MS_ENV_KEY, ""), 0
        )
        max_recycle_per_tick = _parse_int_env(
            os.environ.get(_CHANNEL_MAX_RECYCLE_PER_TICK_ENV_KEY, ""), 1
        )

        # Channel/stub lifecycle, round-robin and in-flight accounting live in
        # the pool. grpc.aio channels are loop-affine, so the pool builds them
        # lazily on the running loop (via ``ensure_started``), never in __init__.
        self._pool = ForwardChannelPool(
            forward_addrs,
            channels_per_addr,
            stub_factory=predict_v2_pb2_grpc.GRPCInferenceServiceStub,
            channel_factory=lambda addr: grpc.aio.insecure_channel(
                addr, options=_FORWARD_CHANNEL_OPTS
            ),
            max_age_ms=max_age_ms,
            recycle_interval_ms=recycle_interval_ms or None,
            max_recycle_per_tick=max_recycle_per_tick,
        )

        logging.info(
            "[DashScGrpc] DashScProxyServicer configured: %d addresses × %d channels/addr "
            "(max_age_ms=%d): %s",
            len(forward_addrs),
            channels_per_addr,
            max_age_ms,
            forward_addrs,
        )

    async def open(self) -> None:
        """Build the outbound channel pool on the current running event loop."""
        self._pool.ensure_started()

    async def close(self) -> None:
        """Drain and close the channel pool. Safe to call multiple times.

        Called after ``server.stop(grace)`` has let in-flight RPCs finish, so
        the pool's force-close of any residue cannot interrupt a live RPC.
        """
        await self._pool.close()

    async def ModelStreamInfer(self, request_iterator, context):
        self._pool.ensure_started()
        pc = self._pool.acquire()
        if pc is None:
            # Shutdown race: ``close()`` cleared the pool after grpcio had
            # already dispatched this RPC. ``context.abort`` raises
            # ``grpc.aio.AbortError`` / ``grpc.RpcError`` here; the interceptor's
            # ``_classify_rpc_exception`` records it as ``UNAVAILABLE`` (a bounded
            # ``error_code`` bucket) instead of ``UNKNOWN_ZeroDivisionError``. No
            # channel was acquired, so there is nothing to release.
            await context.abort(grpc.StatusCode.UNAVAILABLE, "forwarder shutting down")
            return
        # ``release`` is synchronous and brackets the entire forward path, so the
        # channel's in-flight count is decremented even on client cancel /
        # GeneratorExit (the outer generator is ``aclose``'d). That count is what
        # lets the pool close a retired channel only when it is truly idle.
        try:
            async for resp in self._forward(pc, request_iterator, context):
                yield resp
        finally:
            self._pool.release(pc)

    async def _forward(self, pc, request_iterator, context):
        stub = pc.stub
        addr = pc.addr
        idx = pc.addr_idx

        # Grab the forward access record (installed by the access-log
        # interceptor) so proxy-path diagnostics land inline on the access log.
        access_record = ForwardAccessRecord.from_context(context)
        if access_record is not None:
            try:
                access_record.mark_backend_call_start(addr, idx)
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

        try:
            upstream_iter = stub.ModelStreamInfer(request_iterator, metadata=md)
        except BaseException as e:
            if access_record is not None:
                try:
                    access_record.mark_backend_error(e)
                    access_record.mark_backend_done()
                except Exception:
                    pass
            raise

        # Cancel propagation under grpc.aio is implicit: when the inbound RPC
        # is cancelled by the client, the grpc.aio framework cancels the
        # handler coroutine; ``asyncio.CancelledError`` then unwinds through
        # the ``async for`` over ``upstream_iter`` and cancels the downstream
        # aio call automatically. No ``context.add_callback`` plumbing needed.

        if access_record is not None:

            async def counting_response_iter():
                try:
                    async for resp in upstream_iter:
                        try:
                            access_record.capture_backend_response_chunk(resp)
                        except Exception:
                            pass
                        yield resp
                except BaseException as e:
                    try:
                        access_record.mark_backend_error(e)
                    except Exception:
                        pass
                    raise
                finally:
                    try:
                        access_record.mark_backend_done()
                    except Exception:
                        pass

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
        buffered = self._buffered_iter(downstream_iter, access_record)
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
    async def _buffered_iter(downstream_iter, access_record=None):
        """Hold the first chunk until the second arrives, then yield both back-to-back.

        Smooths PD-disaggregation TPOT perception: in PD mode the gap between
        token 1 (emitted by prefill) and token 2 (emitted by decode, after KV cache
        handoff) is much larger than steady-state TPOT. Buffering delays token 1
        until token 2 is ready, so clients see a uniform inter-token cadence.

        ``access_record`` is the forward access record (or ``None`` in tests);
        the stage labels it receives distinguish the three failure shapes at a
        glance on the access log line:

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
            if access_record is not None:
                try:
                    access_record.buffered_stage = s
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
