"""gRPC to gRPC reverse-proxy servicer for predict_v2.proto (grpc.aio)."""

from __future__ import annotations

import asyncio
from typing import Optional

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.dash_sc.proxy.access_record import ForwardAccessRecord
from rtp_llm.dash_sc.proxy.channel_pool import (
    CHANNELS_PER_ADDR_ENV_KEY,
    DEFAULT_CHANNELS_PER_ADDR,
    FORWARD_ADDR_ENV_KEY,
    ForwardChannelPoolConfig,
    LoopBoundForwardChannelPool,
    parse_channels_per_addr,
    parse_forward_addrs,
)
from rtp_llm.dash_sc.proxy.context import get_forward_access_record


def _is_stream_done(resp: predict_v2_pb2.ModelStreamInferResponse) -> bool:
    """True when the proxy should stop forwarding after this frame."""
    if resp.error_message:
        return True
    infer = resp.infer_response
    for i, out in enumerate(infer.outputs):
        if out.name == "finished" and i < len(infer.raw_output_contents):
            return infer.raw_output_contents[i] == b"\x01"
    return False

# Backward-compatible names used by existing tests and entrypoints.
_FORWARD_ENV_KEY = FORWARD_ADDR_ENV_KEY
_CHANNELS_PER_ADDR_ENV_KEY = CHANNELS_PER_ADDR_ENV_KEY
_DEFAULT_CHANNELS_PER_ADDR = DEFAULT_CHANNELS_PER_ADDR
_parse_forward_addrs = parse_forward_addrs
_parse_channels_per_addr = parse_channels_per_addr


class DashScProxyServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """Transparent proxy whose transport lifecycle is owned by a channel pool."""

    def __init__(
        self,
        forward_addrs: Optional[list[str]] = None,
        channels_per_addr: Optional[int] = None,
        channel_pool: Optional[LoopBoundForwardChannelPool] = None,
    ):
        if channel_pool is None:
            config = ForwardChannelPoolConfig.from_env(
                forward_addrs=forward_addrs,
                channels_per_addr=channels_per_addr,
            )
            channel_pool = LoopBoundForwardChannelPool(config)
        self._channel_pool = channel_pool

    async def open(self) -> None:
        await self._channel_pool.open()

    async def close(self) -> None:
        await self._channel_pool.close()

    async def ModelStreamInfer(self, request_iterator, context):
        endpoint = self._channel_pool.pick()
        if endpoint is None:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "forwarder shutting down")
            return

        record = get_forward_access_record(context)
        if record is not None:
            record.mark_backend_call_start(endpoint.addr, endpoint.addr_index)

        try:
            metadata = context.invocation_metadata() or ()
        except Exception:
            metadata = ()

        try:
            upstream_iter = endpoint.stub.ModelStreamInfer(
                request_iterator, metadata=metadata
            )
        except BaseException as e:
            _mark_backend_error(record, e)
            raise

        downstream_iter = self._observe_backend_responses(upstream_iter, record)
        buffered = self._buffer_first_chunk(downstream_iter, record)
        try:
            try:
                async for resp in buffered:
                    yield resp
            finally:
                await buffered.aclose()
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
          and the buffered iterator (currently ``_observe_backend_responses``),
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
    async def _observe_backend_responses(
        upstream_iter, record: Optional[ForwardAccessRecord]
    ):
        try:
            async for resp in upstream_iter:
                if record is not None:
                    record.capture_backend_response_chunk(resp)
                yield resp
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except BaseException as e:
            _mark_backend_error(record, e)
            raise
        finally:
            if record is not None:
                record.mark_backend_done()

    @staticmethod
    async def _buffer_first_chunk(downstream_iter, record: Optional[ForwardAccessRecord]):
        """Hold token 1 until token 2 so PD handoff does not look like bad TPOT.

        The buffer state is reported through ``ForwardAccessRecord`` so access
        logs can distinguish backend silence from token-1 buffering and client
        disconnects.
        """

        def set_stage(stage: str) -> None:
            if record is not None:
                record.mark_buffer_stage(stage)

        def flush_stage(stage: str) -> None:
            if record is not None:
                record.mark_buffer_flushed(stage)

        it = downstream_iter.__aiter__()
        buffered = None
        set_stage("waiting_first")
        try:
            try:
                buffered = await it.__anext__()
            except StopAsyncIteration:
                return
            if _is_stream_done(buffered):
                yield buffered
                buffered = None
                flush_stage("flushed_first")
                return

            set_stage("waiting_second")
            if record is not None:
                record.mark_buffer_wait_start()
            try:
                second = await it.__anext__()
            except StopAsyncIteration:
                yield buffered
                buffered = None
                flush_stage("flushed_first")
                return

            yield buffered
            buffered = None
            yield second
            flush_stage("flushed_both")
            if _is_stream_done(second):
                return
            async for remaining in it:
                yield remaining
                if _is_stream_done(remaining):
                    return
        except GeneratorExit:
            if buffered is not None:
                set_stage("dropped_buffered_on_exception")
            raise
        except BaseException:
            if buffered is not None:
                try:
                    yield buffered
                    flush_stage("flushed_first_on_exception")
                except BaseException:
                    set_stage("dropped_buffered_on_exception")
            raise


def _mark_backend_error(record: Optional[ForwardAccessRecord], exc: BaseException) -> None:
    if record is not None:
        record.mark_backend_error(exc)
        record.mark_backend_done()
