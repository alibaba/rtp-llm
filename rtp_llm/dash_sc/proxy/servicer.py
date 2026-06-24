"""gRPC→gRPC reverse-proxy servicer for predict_v2.proto (grpc.aio).

Supports multiple backend addresses with service discovery and a per-address
HTTP/2 channel cache. Runs on grpc.aio - every servicer method is a coroutine
or async generator so the whole proxy path stays on a single asyncio event loop.
"""

from __future__ import annotations

import logging
from typing import Optional

import grpc

from rtp_llm.dash_sc.access_log import emit_access_log, emit_query_log
from rtp_llm.dash_sc.access_record import GrpcAccessRecord, to_optional_int
from rtp_llm.dash_sc.codec import (
    DASH_ERROR_BAD_REQUEST,
    DASH_ERROR_CAPACITY,
    build_dash_error_response,
    parse_max_new_tokens_for_proxy,
)
from rtp_llm.dash_sc.grpc_metrics import (
    report_arrival,
    report_chunk,
    report_forwarder_rpc_done,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.dash_sc.proxy.service_route import create_service_discovery_from_env
from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool

_FORWARD_CHANNEL_OPTS: list[tuple[str, int]] = [
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", 0),
    ("grpc.http2.max_pings_without_data", 0),
]
_CHANNEL_CLEANUP_INTERVAL_S = 60


def _is_stream_done(resp: predict_v2_pb2.ModelStreamInferResponse) -> bool:
    """True when the proxy should stop forwarding after this frame."""
    if resp.error_message:
        return True
    infer = resp.infer_response
    for i, out in enumerate(infer.outputs):
        if out.name == "finished" and i < len(infer.raw_output_contents):
            return infer.raw_output_contents[i] == b"\x01"
    return False


def _invalid_max_new_tokens_message(request) -> str | None:
    max_new_tokens, from_completion_alias = parse_max_new_tokens_for_proxy(request)
    if max_new_tokens > 0:
        return None
    param_name = "max_completion_tokens" if from_completion_alias else "max_new_tokens"
    return f"invalid {param_name}: {max_new_tokens}; " "must be greater than 0"


async def _close_request_iterator_quietly(request_iter) -> None:
    try:
        await request_iter.aclose()
    except AttributeError:
        return
    except Exception:
        pass


async def _abort_with_downstream_grpc_error(context, exc: grpc.aio.AioRpcError) -> None:
    code = exc.code() or grpc.StatusCode.UNKNOWN
    details = exc.details() or code.name
    await context.abort(code, details)


class DashScProxyServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """Pure transparent proxy (grpc.aio) across discovered downstream addrs."""

    def __init__(
        self,
        *,
        rank_id: Optional[int] = None,
        server_id: str = "",
    ):
        self._channel_pool = GrpcHostChannelPool(
            options=_FORWARD_CHANNEL_OPTS,
            cleanup_interval=_CHANNEL_CLEANUP_INTERVAL_S,
        )
        self._discovery = create_service_discovery_from_env()
        # Access-log identity, injected at construction (``DashScApp`` owns the
        # rank/server identity). The two ids are the only state the log + metric
        # projections need; ``server_id`` arrives as the snowflake string,
        # coerced to ``Optional[int]`` once here. The kmonitor tag dict is
        # memoized per (rank, server) in ``grpc_metrics``, so the per-chunk hot
        # path never re-stringifies them. Tool-call repetition detection is
        # intentionally NOT wired in: it lives only on the frontend inference
        # path; the transparent proxy keeps a forward summary without loop
        # detection.
        self._rank_id = rank_id
        self._server_id = to_optional_int(server_id)
        logging.info("[DashScGrpc] DashScProxyServicer configured")

    async def open(self) -> None:
        """Keep the proxy lifecycle hook explicit; channels are opened lazily."""
        return

    async def close(self) -> None:
        """Drain and close channel-cache resources. Safe to call multiple times.

        Called after ``server.stop(grace)`` has let in-flight RPCs finish, so
        the cache's force-close of any residue cannot interrupt a live RPC.
        """
        await self._channel_pool.close()

    def _record_and_report_chunk(self, record: GrpcAccessRecord, resp) -> None:
        """Capture the frame and fan out per-chunk metrics (records, no log)."""
        is_first, now = record.record_response_chunk(resp)
        if _is_stream_done(resp):
            record.mark_terminal(now=now)
        report_chunk(
            record,
            rank_id=self._rank_id,
            server_id=self._server_id,
            is_first=is_first,
            now=now,
        )

    async def ModelStreamInfer(self, request_iterator, context):
        # Self-managed access-log lifecycle (the shared interceptor is gone).
        # Create/arrival/query go first — before any inbound frame — so a
        # frame-less RPC (peer closed before sending) still reports arrival and
        # produces an access line via the ``finally`` below.
        record = GrpcAccessRecord.create(
            context,
            "ModelStreamInfer",
            "bidi_stream",
            raw_mode=True,
        )
        emit_query_log(record, rank_id=self._rank_id, server_id=self._server_id)
        report_arrival(rank_id=self._rank_id, server_id=self._server_id)
        exc: Optional[BaseException] = None
        try:
            request_iter = request_iterator.__aiter__()
            try:
                first_request = await request_iter.__anext__()
            except StopAsyncIteration:
                record.mark_request_done("eof")
                return
            record.req_count = 1
            record.record_request_frame(first_request)

            invalid_message = _invalid_max_new_tokens_message(first_request)
            if invalid_message is not None:
                record.mark_request_done("eof")
                logging.warning(
                    "[DashScGrpc] proxy parameter error: %s", invalid_message
                )
                await _close_request_iterator_quietly(request_iter)
                error_spec = DASH_ERROR_BAD_REQUEST
                resp = build_dash_error_response(
                    str(first_request.id),
                    first_request.model_name,
                    error_spec=error_spec,
                    status_message=invalid_message,
                )
                self._record_and_report_chunk(record, resp)
                yield resp
                return

            async def validated_request_iter():
                status = "eof"
                try:
                    yield first_request
                    async for req in request_iter:
                        record.req_count += 1
                        yield req
                except BaseException:
                    status = "error"
                    raise
                finally:
                    record.mark_request_done(status)

            route_addr = self._discovery.resolve()
            if route_addr is None:
                record.mark_request_done("eof")
                msg = "forward backend unavailable"
                logging.warning("[DashScGrpc] proxy discovery unavailable: %s", msg)
                await _close_request_iterator_quietly(request_iter)
                error_spec = DASH_ERROR_CAPACITY
                resp = build_dash_error_response(
                    str(first_request.id),
                    first_request.model_name,
                    error_spec=error_spec,
                    status_message=msg,
                )
                self._record_and_report_chunk(record, resp)
                yield resp
                return

            grpc_target = route_addr.grpc_target
            try:
                channel = await self._channel_pool.get(grpc_target)
            except RuntimeError as e:
                record.mark_request_done("eof")
                msg = "forward backend unavailable"
                logging.warning(
                    "[DashScGrpc] proxy channel unavailable: backend=%s error=%s",
                    grpc_target,
                    e,
                )
                await _close_request_iterator_quietly(request_iter)
                error_spec = DASH_ERROR_CAPACITY
                resp = build_dash_error_response(
                    str(first_request.id),
                    first_request.model_name,
                    error_spec=error_spec,
                    status_message=msg,
                )
                self._record_and_report_chunk(record, resp)
                yield resp
                return

            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
            async for resp in self._forward(
                stub, grpc_target, validated_request_iter(), context, record
            ):
                self._record_and_report_chunk(record, resp)
                yield resp
        except BaseException as e:
            exc = e
            raise
        finally:
            end_ts = record.resolve_status(context, exc)
            # Log first, metrics second — a kmonitor hiccup must never delay or
            # drop the access record (user-mandated ordering).
            emit_access_log(
                record,
                rank_id=self._rank_id,
                server_id=self._server_id,
                end_ts=end_ts,
            )
            report_forwarder_rpc_done(
                record,
                rank_id=self._rank_id,
                server_id=self._server_id,
                status=record.status,
            )

    async def _forward(
        self,
        stub,
        addr: str,
        request_iterator,
        context,
        access_record: GrpcAccessRecord,
    ):
        access_record.mark_backend_call_start(addr)

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
        except grpc.aio.AioRpcError as e:
            access_record.mark_backend_error(e)
            access_record.mark_backend_done()
            await _abort_with_downstream_grpc_error(context, e)
            return
        except BaseException as e:
            access_record.mark_backend_error(e)
            access_record.mark_backend_done()
            raise

        # Cancel propagation under grpc.aio is implicit: when the inbound RPC
        # is cancelled by the client, the grpc.aio framework cancels the
        # handler coroutine; ``asyncio.CancelledError`` then unwinds through
        # the ``async for`` over ``upstream_iter`` and cancels the downstream
        # aio call automatically. No ``context.add_callback`` plumbing needed.

        async def counting_response_iter():
            try:
                async for resp in upstream_iter:
                    access_record.capture_backend_response_chunk(resp)
                    yield resp
            except BaseException as e:
                access_record.mark_backend_error(e)
                raise
            finally:
                access_record.mark_backend_done()

        downstream_iter = counting_response_iter()

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
                        if access_record.buffered_stage == "waiting_second":
                            access_record.buffered_stage = (
                                "dropped_buffered_on_exception"
                            )
                        raise
            finally:
                try:
                    await buffered.aclose()
                except Exception:
                    pass
        except grpc.aio.AioRpcError as e:
            await _abort_with_downstream_grpc_error(context, e)
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
