"""Unit tests for ``rtp_llm.dash_sc.proxy.servicer`` (grpc.aio).

Tests verify the async request_iterator is passed correctly to the backend
stub, first-chunk buffering behavior, access-log diagnostic injection, metadata
propagation, and the per-addr channel pool.
"""

from __future__ import annotations

import struct
import unittest
from unittest.mock import MagicMock, patch

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.proxy.servicer import (
    DashScProxyServicer,
    _parse_channels_per_addr,
    _parse_forward_addrs,
)


def _make_request(
    model_name: str = "test_model", id: str = "test_id"
) -> predict_v2_pb2.ModelInferRequest:
    """Create a minimal ModelInferRequest for testing."""
    req = predict_v2_pb2.ModelInferRequest()
    req.model_name = model_name
    req.id = id
    inp = req.inputs.add()
    inp.name = "input_ids"
    inp.datatype = "INT32"
    inp.shape.append(2)
    req.raw_input_contents.append(struct.pack("<2i", 1, 2))
    return req


def _make_response() -> predict_v2_pb2.ModelStreamInferResponse:
    """Create a minimal ModelStreamInferResponse for testing."""
    resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = resp.infer_response
    out = infer.outputs.add()
    out.name = "output"
    out.datatype = "INT32"
    out.shape.append(1)
    infer.raw_output_contents.append(struct.pack("<i", 42))
    return resp


class _AsyncIter:
    """Minimal async iterator over a prebuilt list."""

    def __init__(self, items):
        self._items = list(items)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        x = self._items[self._i]
        self._i += 1
        return x


async def _drain(aiter):
    return [x async for x in aiter]


async def _request_gen(*reqs):
    for r in reqs:
        yield r


class ParseForwardAddrsTest(unittest.TestCase):
    def test_single_address(self) -> None:
        result = _parse_forward_addrs("10.0.0.1:8096")
        self.assertEqual(result, ["10.0.0.1:8096"])

    def test_comma_separated(self) -> None:
        result = _parse_forward_addrs("10.0.0.1:8096,10.0.0.2:8096")
        self.assertEqual(result, ["10.0.0.1:8096", "10.0.0.2:8096"])

    def test_json_array(self) -> None:
        result = _parse_forward_addrs('["10.0.0.1:8096", "10.0.0.2:8096"]')
        self.assertEqual(result, ["10.0.0.1:8096", "10.0.0.2:8096"])

    def test_empty_string(self) -> None:
        result = _parse_forward_addrs("")
        self.assertEqual(result, [])

    def test_whitespace_only(self) -> None:
        result = _parse_forward_addrs("   ")
        self.assertEqual(result, [])

    def test_comma_with_spaces(self) -> None:
        result = _parse_forward_addrs("10.0.0.1:8096 , 10.0.0.2:8096 ")
        self.assertEqual(result, ["10.0.0.1:8096", "10.0.0.2:8096"])


class IteratorBehaviorTest(unittest.IsolatedAsyncioTestCase):
    """Verify the async request iterator is passed through to the downstream stub."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "grpc.aio.insecure_channel", return_value=MagicMock()
        )
        self.channel_patcher.start()

        self.servicer = DashScProxyServicer(["127.0.0.1:1"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub]

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        self.channel_patcher.stop()

    async def test_iterator_passed_not_list(self) -> None:
        """The stub receives an async iterator, not a list."""
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [_make_response(), _make_response()]
        )

        responses = await _drain(
            self.servicer.ModelStreamInfer(
                _request_gen(_make_request("req1"), _make_request("req2")),
                MagicMock(),
            )
        )

        call_arg = self.mock_stub.ModelStreamInfer.call_args[0][0]
        self.assertTrue(
            hasattr(call_arg, "__aiter__"),
            "Must be async iterable",
        )
        self.assertEqual(len(responses), 2)


class BufferFirstTokenTest(unittest.IsolatedAsyncioTestCase):
    """First-chunk buffering preserves order, flushes on single-chunk end,
    re-raises cleanly on error."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "grpc.aio.insecure_channel", return_value=MagicMock()
        )
        self.channel_patcher.start()

        self.servicer = DashScProxyServicer(["127.0.0.1:1"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub]

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        self.channel_patcher.stop()

    def _make_resp(self, tag: str) -> predict_v2_pb2.ModelStreamInferResponse:
        resp = _make_response()
        resp.error_message = tag
        return resp

    async def test_buffer_happy_path_holds_first_until_second(self) -> None:
        yielded_marker: list[str] = []

        async def downstream_gen():
            yielded_marker.append("yielded_a")
            yield self._make_resp("a")
            yielded_marker.append("yielded_b")
            yield self._make_resp("b")
            yielded_marker.append("yielded_c")
            yield self._make_resp("c")

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        out = await _drain(
            self.servicer.ModelStreamInfer(
                _request_gen(_make_request("req1")), MagicMock()
            )
        )

        self.assertEqual([r.error_message for r in out], ["a", "b", "c"])
        self.assertEqual(yielded_marker, ["yielded_a", "yielded_b", "yielded_c"])

    async def test_buffer_single_chunk_flushes_on_stream_end(self) -> None:
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [self._make_resp("only")]
        )

        out = await _drain(
            self.servicer.ModelStreamInfer(
                _request_gen(_make_request("req1")), MagicMock()
            )
        )
        self.assertEqual([r.error_message for r in out], ["only"])

    async def test_buffer_zero_chunk_returns_cleanly(self) -> None:
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter([])

        out = await _drain(
            self.servicer.ModelStreamInfer(
                _request_gen(_make_request("req1")), MagicMock()
            )
        )
        self.assertEqual(out, [])

    async def test_buffer_error_after_first_chunk_flushes_then_raises(self) -> None:
        class FakeRpcError(grpc.RpcError):
            pass

        async def downstream_gen():
            yield self._make_resp("a")
            raise FakeRpcError("downstream failed")

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        collected = []
        with self.assertRaises(FakeRpcError):
            async for r in self.servicer.ModelStreamInfer(
                _request_gen(_make_request("req1")), MagicMock()
            ):
                collected.append(r)
        self.assertEqual([r.error_message for r in collected], ["a"])

    async def test_buffer_error_before_any_chunk_raises_cleanly(self) -> None:
        class FakeRpcError(grpc.RpcError):
            pass

        async def downstream_gen():
            raise FakeRpcError("downstream failed")
            yield  # pragma: no cover

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        collected = []
        with self.assertRaises(FakeRpcError):
            async for r in self.servicer.ModelStreamInfer(
                _request_gen(_make_request("req1")), MagicMock()
            ):
                collected.append(r)
        self.assertEqual(collected, [])


class AccessLogDiagInjectionTest(unittest.IsolatedAsyncioTestCase):
    """Forwarder writes ``backend_addr`` / ``backend_resp_count`` /
    ``buffered_stage`` onto the access-log aggregate attached at
    ``context._dash_sc_access_agg``."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "grpc.aio.insecure_channel", return_value=MagicMock()
        )
        self.channel_patcher.start()

        self.servicer = DashScProxyServicer(["10.0.0.1:8096", "10.0.0.2:8096"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub, self.mock_stub]

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        self.channel_patcher.stop()

    def _ctx_with_agg(self) -> tuple[MagicMock, object]:
        import time as _time

        from rtp_llm.dash_sc.access_log import _RpcAggregate

        agg = _RpcAggregate(
            method="/x.Svc/M",
            stream_type="bidi_stream",
            peer="",
            start_ts=_time.time(),
        )
        ctx = MagicMock()
        ctx._dash_sc_access_agg = agg
        return ctx, agg

    def _make_resp(self, tag: str) -> predict_v2_pb2.ModelStreamInferResponse:
        resp = _make_response()
        resp.error_message = tag
        return resp

    def _patch_addr(self, idx: int) -> None:
        self.servicer._next_stub = lambda: (self.mock_stub, idx)

    async def test_backend_addr_set_to_chosen_backend(self) -> None:
        self._patch_addr(1)
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [self._make_resp("a")]
        )
        ctx, agg = self._ctx_with_agg()

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        self.assertEqual(agg.backend_addr, "10.0.0.2:8096")

    async def test_backend_resp_count_tracks_upstream_frames(self) -> None:
        self._patch_addr(0)
        chunks = [self._make_resp("a"), self._make_resp("b"), self._make_resp("c")]
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(chunks)
        ctx, agg = self._ctx_with_agg()

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        self.assertEqual(agg.backend_resp_count, 3)

    async def test_stage_waiting_first_on_immediate_downstream_error(self) -> None:
        self._patch_addr(0)

        class FakeRpcError(grpc.RpcError):
            pass

        async def downstream_gen():
            raise FakeRpcError("backend silent")
            yield  # pragma: no cover

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()
        ctx, agg = self._ctx_with_agg()

        with self.assertRaises(FakeRpcError):
            await _drain(
                self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
            )
        self.assertEqual(agg.buffered_stage, "waiting_first")
        self.assertEqual(agg.backend_resp_count, 0)

    async def test_stage_flushed_first_on_single_chunk_clean_end(self) -> None:
        self._patch_addr(0)
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [self._make_resp("only")]
        )
        ctx, agg = self._ctx_with_agg()

        out = await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(agg.buffered_stage, "flushed_first")
        self.assertEqual(agg.backend_resp_count, 1)

    async def test_stage_flushed_both_on_happy_path(self) -> None:
        self._patch_addr(0)
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [self._make_resp("a"), self._make_resp("b"), self._make_resp("c")]
        )
        ctx, agg = self._ctx_with_agg()

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        self.assertEqual(agg.buffered_stage, "flushed_both")
        self.assertEqual(agg.backend_resp_count, 3)

    async def test_stage_flushed_first_on_exception_when_client_consumes(
        self,
    ) -> None:
        """Downstream errors after token 1 and the client fully drains -> flushed_first_on_exception."""
        self._patch_addr(0)

        class FakeRpcError(grpc.RpcError):
            pass

        async def downstream_gen():
            yield self._make_resp("a")
            raise FakeRpcError("downstream cut after token 1")

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()
        ctx, agg = self._ctx_with_agg()

        got: list[str] = []
        with self.assertRaises(FakeRpcError):
            async for r in self.servicer.ModelStreamInfer(
                _request_gen(_make_request("req1")), ctx
            ):
                got.append(r.error_message)
        self.assertEqual(got, ["a"])
        self.assertEqual(agg.buffered_stage, "flushed_first_on_exception")
        self.assertEqual(agg.backend_resp_count, 1)

    async def test_stage_dropped_buffered_when_client_went_away(self) -> None:
        """Downstream errors after token 1, client-side drops mid-yield ->
        dropped_buffered_on_exception."""
        self._patch_addr(0)

        class FakeRpcError(grpc.RpcError):
            pass

        async def downstream_gen():
            yield self._make_resp("a")
            raise FakeRpcError("downstream cut after token 1")

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()
        ctx, agg = self._ctx_with_agg()

        gen = self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        # Consume first frame.
        first = await gen.__anext__()
        self.assertEqual(first.error_message, "a")
        # Simulate client handler dying — gRPC framework closes the generator,
        # which inside our ``yield buffered`` reads as ``athrow()``.
        with self.assertRaises(BaseException):
            await gen.athrow(RuntimeError("client gone"))
        self.assertEqual(agg.buffered_stage, "dropped_buffered_on_exception")


class MetadataPropagationTest(unittest.IsolatedAsyncioTestCase):
    """Client-sent gRPC metadata must reach the downstream stub verbatim."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "grpc.aio.insecure_channel", return_value=MagicMock()
        )
        self.channel_patcher.start()
        self.servicer = DashScProxyServicer(["127.0.0.1:1"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub]

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        self.channel_patcher.stop()

    async def test_client_metadata_forwarded_to_downstream_stub(self) -> None:
        md = [
            ("x-dashscope-request-id", "corr-abc-42"),
            ("x-request-id", "generic-7"),
        ]
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = md
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter([_make_response()])

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        kwargs = self.mock_stub.ModelStreamInfer.call_args.kwargs
        self.assertIn("metadata", kwargs)
        forwarded = list(kwargs["metadata"])
        self.assertEqual(forwarded, md)

    async def test_invocation_metadata_raising_does_not_break_forwarder(
        self,
    ) -> None:
        """Context whose ``invocation_metadata()`` raises falls back to empty metadata."""
        ctx = MagicMock()
        ctx.invocation_metadata.side_effect = RuntimeError("context teardown race")
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter([_make_response()])

        out = await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        self.assertEqual(len(out), 1)
        kwargs = self.mock_stub.ModelStreamInfer.call_args.kwargs
        self.assertEqual(tuple(kwargs["metadata"]), ())


class ChannelPoolTest(unittest.IsolatedAsyncioTestCase):
    """Per-addr channel pool: N channels/addr, round-robin over all stubs."""

    async def test_default_one_channel_per_addr(self) -> None:
        with patch("grpc.aio.insecure_channel", return_value=MagicMock()) as mock_ch:
            servicer = DashScProxyServicer(["10.0.0.1:8096", "10.0.0.2:8096"])
        self.assertEqual(len(servicer._channels), 2)
        self.assertEqual(len(servicer._stubs), 2)
        self.assertEqual(mock_ch.call_count, 2)
        await servicer.close()

    async def test_pool_round_robin_over_addrs(self) -> None:
        with patch("grpc.aio.insecure_channel", return_value=MagicMock()):
            servicer = DashScProxyServicer(
                ["10.0.0.1:8096", "10.0.0.2:8096"], channels_per_addr=3
            )
        self.assertEqual(len(servicer._stubs), 6)
        self.assertEqual(servicer._stub_addr_idx, [0, 0, 0, 1, 1, 1])
        seq = [servicer._next_stub()[1] for _ in range(7)]
        self.assertEqual(seq, [0, 0, 0, 1, 1, 1, 0])
        await servicer.close()

    def test_parse_channels_per_addr_fallback(self) -> None:
        self.assertEqual(_parse_channels_per_addr(""), 1)
        self.assertEqual(_parse_channels_per_addr("abc"), 1)
        self.assertEqual(_parse_channels_per_addr("0"), 1)
        self.assertEqual(_parse_channels_per_addr("-4"), 1)
        self.assertEqual(_parse_channels_per_addr("4"), 4)

    async def test_next_stub_returns_none_after_close(self) -> None:
        """Shutdown race: after ``close()`` the pool is empty and
        ``_next_stub`` returns ``(None, -1)``; ``ModelStreamInfer`` translates
        that into a proper UNAVAILABLE abort.
        """
        with patch("grpc.aio.insecure_channel", return_value=MagicMock()):
            servicer = DashScProxyServicer(["10.0.0.1:8096"])
        await servicer.close()
        self.assertEqual(servicer._next_stub(), (None, -1))

        ctx = MagicMock()

        async def _abort(*_args, **_kwargs):
            raise grpc.RpcError("UNAVAILABLE")

        ctx.abort.side_effect = _abort

        with self.assertRaises(grpc.RpcError):
            await _drain(
                servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
            )
        ctx.abort.assert_called_once()
        self.assertEqual(ctx.abort.call_args[0][0], grpc.StatusCode.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
