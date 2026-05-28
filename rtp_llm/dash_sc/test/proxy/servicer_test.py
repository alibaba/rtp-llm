"""Unit tests for ``rtp_llm.dash_sc.proxy.servicer`` (grpc.aio)."""

from __future__ import annotations

import asyncio
import struct
import time
import unittest
from unittest.mock import MagicMock, patch

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.proxy.access_record import ForwardAccessRecord
from rtp_llm.dash_sc.proxy.channel_pool import (
    ForwardChannelPoolConfig,
    ForwardEndpoint,
    LoopBoundForwardChannelPool,
)
from rtp_llm.dash_sc.proxy.context import attach_forward_access_record
from rtp_llm.dash_sc.proxy.servicer import (
    DashScProxyServicer,
    _parse_channels_per_addr,
    _parse_forward_addrs,
)


def _make_request(
    model_name: str = "test_model", id: str = "test_id"
) -> predict_v2_pb2.ModelInferRequest:
    req = predict_v2_pb2.ModelInferRequest()
    req.model_name = model_name
    req.id = id
    inp = req.inputs.add()
    inp.name = "input_ids"
    inp.datatype = "INT32"
    inp.shape.append(2)
    req.raw_input_contents.append(struct.pack("<2i", 1, 2))
    return req


def _make_response(tag: str = "") -> predict_v2_pb2.ModelStreamInferResponse:
    resp = predict_v2_pb2.ModelStreamInferResponse()
    resp.infer_response.id = tag
    infer = resp.infer_response
    out = infer.outputs.add()
    out.name = "generated_ids"
    out.datatype = "INT32"
    out.shape.append(1)
    infer.raw_output_contents.append(struct.pack("<i", 42))
    return resp


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        return item


async def _drain(aiter):
    return [x async for x in aiter]


async def _request_gen(*reqs):
    for req in reqs:
        yield req


class _FakePool:
    def __init__(self, endpoints: list[ForwardEndpoint]):
        self._endpoints = endpoints
        self._i = 0
        self.opened = False
        self.closed = False

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True
        self._endpoints = []

    def pick(self):
        if not self._endpoints:
            return None
        endpoint = self._endpoints[self._i]
        self._i = (self._i + 1) % len(self._endpoints)
        return endpoint


def _servicer_with_stub(addrs: list[str] | None = None):
    addrs = addrs or ["127.0.0.1:1"]
    stubs = [MagicMock() for _ in addrs]
    endpoints = [
        ForwardEndpoint(stub=stub, addr=addr, addr_index=i)
        for i, (addr, stub) in enumerate(zip(addrs, stubs))
    ]
    pool = _FakePool(endpoints)
    return DashScProxyServicer(channel_pool=pool), pool, stubs


def _ctx_with_record() -> tuple[MagicMock, ForwardAccessRecord]:
    record = ForwardAccessRecord(
        method="/x.Svc/M",
        stream_type="bidi_stream",
        peer="",
        start_ts=time.time(),
    )
    ctx = MagicMock()
    attach_forward_access_record(ctx, record)
    return ctx, record


class ParseForwardAddrsTest(unittest.TestCase):
    def test_single_address(self) -> None:
        self.assertEqual(_parse_forward_addrs("10.0.0.1:8096"), ["10.0.0.1:8096"])

    def test_comma_separated(self) -> None:
        result = _parse_forward_addrs("10.0.0.1:8096,10.0.0.2:8096")
        self.assertEqual(result, ["10.0.0.1:8096", "10.0.0.2:8096"])

    def test_json_array(self) -> None:
        result = _parse_forward_addrs('["10.0.0.1:8096", "10.0.0.2:8096"]')
        self.assertEqual(result, ["10.0.0.1:8096", "10.0.0.2:8096"])

    def test_empty_values(self) -> None:
        self.assertEqual(_parse_forward_addrs(""), [])
        self.assertEqual(_parse_forward_addrs("   "), [])

    def test_parse_channels_per_addr_fallback(self) -> None:
        self.assertEqual(_parse_channels_per_addr(""), 1)
        self.assertEqual(_parse_channels_per_addr("abc"), 1)
        self.assertEqual(_parse_channels_per_addr("0"), 1)
        self.assertEqual(_parse_channels_per_addr("-4"), 1)
        self.assertEqual(_parse_channels_per_addr("4"), 4)


class ProxyServicerTest(unittest.IsolatedAsyncioTestCase):
    async def test_open_and_close_delegate_to_pool(self) -> None:
        servicer, pool, _ = _servicer_with_stub()
        await servicer.open()
        await servicer.close()
        self.assertTrue(pool.opened)
        self.assertTrue(pool.closed)

    async def test_iterator_passed_not_list(self) -> None:
        servicer, _, stubs = _servicer_with_stub()
        stubs[0].ModelStreamInfer.return_value = _AsyncIter(
            [_make_response(), _make_response()]
        )

        responses = await _drain(
            servicer.ModelStreamInfer(
                _request_gen(_make_request("req1"), _make_request("req2")),
                MagicMock(),
            )
        )

        call_arg = stubs[0].ModelStreamInfer.call_args[0][0]
        self.assertTrue(hasattr(call_arg, "__aiter__"))
        self.assertEqual(len(responses), 2)

    async def test_metadata_forwarded_to_downstream_stub(self) -> None:
        servicer, _, stubs = _servicer_with_stub()
        md = [("x-dashscope-request-id", "corr-abc-42")]
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = md
        stubs[0].ModelStreamInfer.return_value = _AsyncIter([_make_response()])

        await _drain(servicer.ModelStreamInfer(_request_gen(_make_request()), ctx))

        kwargs = stubs[0].ModelStreamInfer.call_args.kwargs
        self.assertEqual(list(kwargs["metadata"]), md)

    async def test_abort_when_pool_empty(self) -> None:
        servicer = DashScProxyServicer(channel_pool=_FakePool([]))
        ctx = MagicMock()

        async def abort(*_args, **_kwargs):
            raise grpc.RpcError("UNAVAILABLE")

        ctx.abort.side_effect = abort
        with self.assertRaises(grpc.RpcError):
            await _drain(servicer.ModelStreamInfer(_request_gen(_make_request()), ctx))
        self.assertEqual(ctx.abort.call_args[0][0], grpc.StatusCode.UNAVAILABLE)

    async def test_backend_diagnostics_recorded(self) -> None:
        servicer, _, stubs = _servicer_with_stub(
            ["10.0.0.1:8096", "10.0.0.2:8096"]
        )
        stubs[0].ModelStreamInfer.return_value = _AsyncIter([_make_response("a")])
        ctx, record = _ctx_with_record()

        await _drain(servicer.ModelStreamInfer(_request_gen(_make_request()), ctx))

        self.assertEqual(record.backend_addr, "10.0.0.1:8096")
        self.assertEqual(record.backend_addr_index, 0)
        self.assertEqual(record.backend_resp_count, 1)
        self.assertEqual(record.buffered_stage, "flushed_first")


class BufferFirstTokenTest(unittest.IsolatedAsyncioTestCase):
    def _make_resp(self, tag: str) -> predict_v2_pb2.ModelStreamInferResponse:
        return _make_response(tag)

    async def test_buffer_happy_path_holds_first_until_second(self) -> None:
        servicer, _, stubs = _servicer_with_stub()
        yielded_marker: list[str] = []

        async def downstream_gen():
            yielded_marker.append("yielded_a")
            yield self._make_resp("a")
            yielded_marker.append("yielded_b")
            yield self._make_resp("b")
            yielded_marker.append("yielded_c")
            yield self._make_resp("c")

        stubs[0].ModelStreamInfer.return_value = downstream_gen()
        out = await _drain(
            servicer.ModelStreamInfer(_request_gen(_make_request()), MagicMock())
        )

        self.assertEqual([r.infer_response.id for r in out], ["a", "b", "c"])
        self.assertEqual(yielded_marker, ["yielded_a", "yielded_b", "yielded_c"])

    async def test_buffer_single_chunk_flushes_on_stream_end(self) -> None:
        servicer, _, stubs = _servicer_with_stub()
        stubs[0].ModelStreamInfer.return_value = _AsyncIter([self._make_resp("only")])

        out = await _drain(
            servicer.ModelStreamInfer(_request_gen(_make_request()), MagicMock())
        )
        self.assertEqual([r.infer_response.id for r in out], ["only"])

    async def test_buffer_zero_chunk_returns_cleanly(self) -> None:
        servicer, _, stubs = _servicer_with_stub()
        stubs[0].ModelStreamInfer.return_value = _AsyncIter([])

        out = await _drain(
            servicer.ModelStreamInfer(_request_gen(_make_request()), MagicMock())
        )
        self.assertEqual(out, [])

    async def test_buffer_error_after_first_chunk_flushes_then_raises(self) -> None:
        servicer, _, stubs = _servicer_with_stub()

        class FakeRpcError(grpc.RpcError):
            pass

        async def downstream_gen():
            yield self._make_resp("a")
            raise FakeRpcError("downstream failed")

        stubs[0].ModelStreamInfer.return_value = downstream_gen()

        collected = []
        with self.assertRaises(FakeRpcError):
            async for r in servicer.ModelStreamInfer(
                _request_gen(_make_request()), MagicMock()
            ):
                collected.append(r)
        self.assertEqual([r.infer_response.id for r in collected], ["a"])

    async def test_buffer_error_before_any_chunk_raises_cleanly(self) -> None:
        servicer, _, stubs = _servicer_with_stub()

        class FakeRpcError(grpc.RpcError):
            pass

        async def downstream_gen():
            raise FakeRpcError("downstream failed")
            yield  # pragma: no cover

        stubs[0].ModelStreamInfer.return_value = downstream_gen()

        with self.assertRaises(FakeRpcError):
            await _drain(
                servicer.ModelStreamInfer(_request_gen(_make_request()), MagicMock())
            )


class ChannelPoolTest(unittest.IsolatedAsyncioTestCase):
    async def test_pool_opens_on_current_loop_and_round_robins(self) -> None:
        created = []

        class FakeChannel:
            def unary_unary(self, *_args, **_kwargs):
                return MagicMock()

            def unary_stream(self, *_args, **_kwargs):
                return MagicMock()

            def stream_unary(self, *_args, **_kwargs):
                return MagicMock()

            def stream_stream(self, *_args, **_kwargs):
                return MagicMock()

            async def close(self):
                return None

        def make_channel(addr, **_kwargs):
            created.append((addr, asyncio.get_running_loop()))
            return FakeChannel()

        with patch("grpc.aio.insecure_channel", side_effect=make_channel):
            pool = LoopBoundForwardChannelPool(
                ForwardChannelPoolConfig(("10.0.0.1:8096", "10.0.0.2:8096"), 2)
            )
            await pool.open()

        loop = asyncio.get_running_loop()
        self.assertEqual([addr for addr, _ in created], ["10.0.0.1:8096"] * 2 + ["10.0.0.2:8096"] * 2)
        self.assertTrue(all(owner is loop for _, owner in created))
        self.assertEqual([pool.pick().addr_index for _ in range(5)], [0, 0, 1, 1, 0])
        await pool.close()
        self.assertIsNone(pool.pick())

    def test_channels_open_on_request_loop_after_sync_construction(self) -> None:
        created_loops = []

        class FakeChannel:
            def unary_unary(self, *_args, **_kwargs):
                return MagicMock()

            def unary_stream(self, *_args, **_kwargs):
                return MagicMock()

            def stream_unary(self, *_args, **_kwargs):
                return MagicMock()

            def stream_stream(self, *_args, **_kwargs):
                return MagicMock()

            async def close(self):
                return None

        def make_channel(*_args, **_kwargs):
            created_loops.append(asyncio.get_running_loop())
            return FakeChannel()

        with patch("grpc.aio.insecure_channel", side_effect=make_channel):
            pool = LoopBoundForwardChannelPool(
                ForwardChannelPoolConfig(("127.0.0.1:1",), 1)
            )

            async def invoke():
                running_loop = asyncio.get_running_loop()
                await pool.open()
                self.assertEqual(created_loops, [running_loop])
                await pool.close()

            asyncio.run(invoke())


class StreamCloseTimingTest(unittest.IsolatedAsyncioTestCase):
    """Measures the gap between *receiving the finished frame downstream* and
    *the entire stream being torn down* across the four code paths the proxy
    can take.

    For each scenario the test records:

    - ``t_finish``: the moment the downstream emits the ``finished=True`` frame
      (or the moment we explicitly invoke close on the manual-API test).
    - ``t_outer``: the moment the consumer of ``ModelStreamInfer`` sees
      ``StopAsyncIteration`` (i.e. when grpc.aio gets the cue to send
      END_STREAM + OK trailers to the gateway).
    - ``t_close``: the moment the downstream's ``except GeneratorExit`` (or
      ``cancel``-observation handler) fires — i.e. when the actual call to
      backend is torn down.

    The reported delta is ``t_close - t_finish``. The implementation aims for
    this to be sub-millisecond and, critically, ``t_close <= t_outer`` (close
    completes *before* upstream trailers fire, so the backend never sees a
    half-closed gap that registers as a client cancel race).
    """

    async def asyncSetUp(self) -> None:
        self.servicer, self.pool, stubs = _servicer_with_stub()
        self.mock_stub = stubs[0]

    async def asyncTearDown(self) -> None:
        await self.servicer.close()

    @staticmethod
    def _make_token_resp(tag: str) -> predict_v2_pb2.ModelStreamInferResponse:
        resp = _make_response()
        resp.infer_response.id = tag
        return resp

    @staticmethod
    def _make_finished_resp() -> predict_v2_pb2.ModelStreamInferResponse:
        """Frame matching ``_is_stream_done``: output named ``finished`` with
        body ``b"\\x01"``."""
        resp = predict_v2_pb2.ModelStreamInferResponse()
        infer = resp.infer_response
        out = infer.outputs.add()
        out.name = "finished"
        out.datatype = "BOOL"
        out.shape.append(1)
        infer.raw_output_contents.append(b"\x01")
        return resp

    def _assert_prompt_close(
        self,
        scenario: str,
        finish_ts: float,
        outer_ts: float,
        close_ts: float | None,
        expected_frames: int,
        collected_frames: int,
        leaked: bool,
    ) -> None:
        """Common assertion + diagnostic print for every scenario.

        Keeps the per-test methods focused on the *shape* of the scenario;
        the timing contract lives here so every path is held to the same bar.
        """
        outer_delta_ms = (outer_ts - finish_ts) * 1000
        close_delta_ms = (close_ts - finish_ts) * 1000 if close_ts is not None else None
        order_delta_ms = (close_ts - outer_ts) * 1000 if close_ts is not None else None

        print(
            f"\n[StreamCloseTimingTest:{scenario}]\n"
            f"  finished -> outer StopAsyncIteration : {outer_delta_ms:7.3f} ms\n"
            f"  finished -> downstream close         : "
            f"{('%.3f ms' % close_delta_ms) if close_delta_ms is not None else 'NOT OBSERVED (relies on GC)'}\n"
            f"  close vs outer return                : "
            f"{('%+.3f ms' % order_delta_ms) if order_delta_ms is not None else 'N/A'}",
            flush=True,
        )

        self.assertEqual(
            collected_frames,
            expected_frames,
            f"[{scenario}] proxy forwarded {collected_frames} frames; "
            f"expected exactly {expected_frames}",
        )
        self.assertFalse(
            leaked,
            f"[{scenario}] downstream was iterated past the finished frame",
        )
        # Outer return gates upstream trailers — must be sub-50ms.
        self.assertLess(
            outer_delta_ms,
            50.0,
            f"[{scenario}] outer return took {outer_delta_ms:.1f}ms after "
            "finished — gateway will see this as a late close",
        )
        # Downstream close must be deterministic, not GC-deferred.
        self.assertIsNotNone(
            close_ts,
            f"[{scenario}] downstream close was never observed — leak via GC",
        )
        # And must complete no later than the outer returns (5ms slack for
        # clock granularity) — otherwise the backend sees a half-closed gap.
        self.assertLessEqual(
            order_delta_ms,
            5.0,
            f"[{scenario}] downstream close fired {order_delta_ms:.1f}ms AFTER "
            "outer returned — backend will log a client-cancel race",
        )

    async def test_close_timing_finished_as_second_frame(self) -> None:
        """``_is_stream_done(second)`` inline check path (token, finished)."""
        loop = asyncio.get_running_loop()
        finish_ts: list[float] = []
        close_ts: list[float] = []
        leaked: list[bool] = []

        async def downstream_gen():
            try:
                yield self._make_token_resp("token1")
                finish_ts.append(loop.time())
                yield self._make_finished_resp()
                await asyncio.sleep(5.0)
                leaked.append(True)
                yield self._make_token_resp("leaked")
            except GeneratorExit:
                close_ts.append(loop.time())
                raise

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        collected = []
        async for resp in self.servicer.ModelStreamInfer(
            _request_gen(_make_request("req1")), MagicMock()
        ):
            collected.append(resp)
        outer_ts = loop.time()

        self._assert_prompt_close(
            "finished@frame2",
            finish_ts[0],
            outer_ts,
            close_ts[0] if close_ts else None,
            expected_frames=2,
            collected_frames=len(collected),
            leaked=bool(leaked),
        )

    async def test_close_timing_finished_deep_in_stream(self) -> None:
        """``async for remaining in it`` early-exit path (10 tokens, then finished).

        Uses bare ``_make_response()`` (no ``error_message`` tag) so the leading
        frames don't trip ``_is_stream_done``'s ``error_message`` short-circuit."""
        loop = asyncio.get_running_loop()
        finish_ts: list[float] = []
        close_ts: list[float] = []
        leaked: list[bool] = []

        async def downstream_gen():
            try:
                for _ in range(10):
                    yield _make_response()
                finish_ts.append(loop.time())
                yield self._make_finished_resp()
                await asyncio.sleep(5.0)
                leaked.append(True)
                yield _make_response()
            except GeneratorExit:
                close_ts.append(loop.time())
                raise

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        collected = []
        async for resp in self.servicer.ModelStreamInfer(
            _request_gen(_make_request("req1")), MagicMock()
        ):
            collected.append(resp)
        outer_ts = loop.time()

        self._assert_prompt_close(
            "finished@frame11",
            finish_ts[0],
            outer_ts,
            close_ts[0] if close_ts else None,
            expected_frames=11,  # 10 tokens + finished
            collected_frames=len(collected),
            leaked=bool(leaked),
        )

    async def test_close_timing_with_record_observer(self) -> None:
        """``ForwardAccessRecord`` path wraps the call for backend telemetry.
        Verifies the wrapper's ``aclose`` propagates the close to the inner
        ``upstream_iter`` synchronously."""
        loop = asyncio.get_running_loop()
        finish_ts: list[float] = []
        close_ts: list[float] = []
        leaked: list[bool] = []

        async def downstream_gen():
            try:
                yield self._make_token_resp("token1")
                finish_ts.append(loop.time())
                yield self._make_finished_resp()
                await asyncio.sleep(5.0)
                leaked.append(True)
                yield self._make_token_resp("leaked")
            except GeneratorExit:
                close_ts.append(loop.time())
                raise

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        ctx = MagicMock()
        record = ForwardAccessRecord(
            method="/x.Svc/M", stream_type="bidi_stream", peer="", start_ts=time.time()
        )
        attach_forward_access_record(ctx, record)

        collected = []
        async for resp in self.servicer.ModelStreamInfer(
            _request_gen(_make_request("req1")), ctx
        ):
            collected.append(resp)
        outer_ts = loop.time()

        self._assert_prompt_close(
            "with_counting_wrapper",
            finish_ts[0],
            outer_ts,
            close_ts[0] if close_ts else None,
            expected_frames=2,
            collected_frames=len(collected),
            leaked=bool(leaked),
        )
        self.assertEqual(record.backend_resp_count, 2)

    async def test_close_timing_cancel_only_call(self) -> None:
        """Mock that mirrors ``grpc.aio.Call``'s surface: exposes ``cancel()``
        but not ``aclose()``. Verifies the production cancel path triggers
        teardown promptly (the previous tests rely on ``aclose`` because async
        generators expose it; a real grpc call doesn't)."""
        loop = asyncio.get_running_loop()
        finish_ts: list[float] = []
        close_ts: list[float] = []

        class CancelOnlyCall:
            """Async-iterable with ``cancel()`` but no ``aclose``.

            ``cancel()`` flips an event that the iterator awaits; observing it
            is how the test catches the close moment."""

            def __init__(self, frames):
                self._frames = list(frames)
                self._i = 0
                self._cancelled = asyncio.Event()

            def cancel(self) -> bool:
                if not self._cancelled.is_set():
                    self._cancelled.set()
                    close_ts.append(loop.time())
                    return True
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._cancelled.is_set():
                    raise StopAsyncIteration
                if self._i >= len(self._frames):
                    # Simulate the slow backend: sit here until cancelled.
                    await self._cancelled.wait()
                    raise StopAsyncIteration
                x = self._frames[self._i]
                self._i += 1
                return x

        def gen_call(*args, **kwargs):
            return CancelOnlyCall(
                [
                    self._make_token_resp("token1"),
                    self._make_finished_resp(),
                ]
            )

        self.mock_stub.ModelStreamInfer.side_effect = gen_call

        # Tag finish_ts the moment the 2nd frame leaves the (fake) backend.
        # We can't instrument inside CancelOnlyCall cleanly so wrap the iter.
        real_iter_cls = CancelOnlyCall

        async def _patched_anext(self):
            if self._cancelled.is_set():
                raise StopAsyncIteration
            if self._i >= len(self._frames):
                await self._cancelled.wait()
                raise StopAsyncIteration
            x = self._frames[self._i]
            self._i += 1
            if self._i == len(self._frames):
                finish_ts.append(loop.time())
            return x

        real_iter_cls.__anext__ = _patched_anext

        collected = []
        async for resp in self.servicer.ModelStreamInfer(
            _request_gen(_make_request("req1")), MagicMock()
        ):
            collected.append(resp)
        outer_ts = loop.time()

        self._assert_prompt_close(
            "cancel_only_call",
            finish_ts[0],
            outer_ts,
            close_ts[0] if close_ts else None,
            expected_frames=2,
            collected_frames=len(collected),
            leaked=False,
        )

    async def test_close_timing_manual_api_invocation(self) -> None:
        """Direct call of the public-ish ``_close_downstream`` helper. Measures
        ``t_close - t_invoke`` to confirm the manual path is also prompt and
        idempotent (calling it twice is a no-op)."""
        loop = asyncio.get_running_loop()
        close_ts: list[float] = []

        async def downstream_gen():
            try:
                while True:
                    yield self._make_token_resp("noop")
                    await asyncio.sleep(0.001)
            except GeneratorExit:
                close_ts.append(loop.time())
                raise

        gen = downstream_gen()
        # Pull one frame so the gen is suspended at its yield (mirrors the
        # state of a real grpc call mid-stream).
        first = await gen.__anext__()
        self.assertTrue(first is not None)

        invoke_ts = loop.time()
        await DashScProxyServicer._close_downstream(gen, gen)
        delta_ms = (close_ts[0] - invoke_ts) * 1000 if close_ts else None
        print(
            f"\n[StreamCloseTimingTest:manual_api]\n"
            f"  _close_downstream invoke -> downstream close: "
            f"{('%.3f ms' % delta_ms) if delta_ms is not None else 'NOT OBSERVED'}",
            flush=True,
        )
        self.assertIsNotNone(close_ts and close_ts[0])
        self.assertLess(delta_ms, 5.0, "manual close did not fire promptly")

        # Idempotency: second call on the already-closed gen must not raise.
        await DashScProxyServicer._close_downstream(gen, gen)
        self.assertEqual(
            len(close_ts), 1, "second close re-entered GeneratorExit handler"
        )


if __name__ == "__main__":
    unittest.main()
