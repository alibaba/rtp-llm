"""Unit tests for ``rtp_llm.dash_sc.forward_service``.

Tests verify that request_iterator is passed correctly to downstream stub
(not converted to list, which was a bug in the initial implementation).
"""

from __future__ import annotations

import struct
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import grpc

from rtp_llm.dash_sc.forward_service import (
    PureForwardServicer,
    _parse_channels_per_addr,
    _parse_forward_addrs,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2


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


class ParseForwardAddrsTest(TestCase):
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


class IteratorBehaviorTest(TestCase):
    """Core test: verify iterator is passed to stub (not converted to list).

    This tests the bug fix where [req] was passed instead of an iterator,
    causing TypeError: 'list' object is not an iterator.
    """

    def setUp(self) -> None:
        # Mock grpc.insecure_channel to avoid real connections
        self.channel_patcher = patch("grpc.insecure_channel", return_value=MagicMock())
        self.channel_patcher.start()

        self.servicer = PureForwardServicer(["127.0.0.1:1"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub]

    def tearDown(self) -> None:
        self.servicer.close()
        self.channel_patcher.stop()

    def test_iterator_passed_not_list(self) -> None:
        """BUG FIX: iterator must be passed to the downstream stub, not a list."""

        def request_gen():
            yield _make_request("req1")
            yield _make_request("req2")

        mock_resp = _make_response()
        self.mock_stub.ModelStreamInfer.return_value = iter([mock_resp, mock_resp])

        responses = list(self.servicer.ModelStreamInfer(request_gen(), MagicMock()))

        # KEY ASSERTION: stub received iterator (has __next__), not list
        call_arg = self.mock_stub.ModelStreamInfer.call_args[0][0]
        self.assertTrue(hasattr(call_arg, "__iter__"), "Must be iterable")
        self.assertTrue(
            hasattr(call_arg, "__next__"),
            "Must be iterator (has __next__), not list",
        )
        self.assertEqual(len(responses), 2)


class BufferFirstTokenTest(TestCase):
    """Verify first-token buffer behavior for PD-disaggregation TPOT smoothing.

    The first chunk from downstream is held until the second arrives (or the
    stream ends / errors); buffering is always on.
    """

    def setUp(self) -> None:
        self.channel_patcher = patch("grpc.insecure_channel", return_value=MagicMock())
        self.channel_patcher.start()

        self.servicer = PureForwardServicer(["127.0.0.1:1"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub]

    def tearDown(self) -> None:
        self.servicer.close()
        self.channel_patcher.stop()

    def _make_resp(self, tag: str) -> predict_v2_pb2.ModelStreamInferResponse:
        resp = _make_response()
        resp.error_message = tag
        return resp

    def test_buffer_happy_path_holds_first_until_second(self) -> None:
        """First chunk is not yielded until second arrives; final order preserved."""
        yielded_marker: list[str] = []

        def downstream_gen():
            yielded_marker.append("yielded_a")
            yield self._make_resp("a")
            yielded_marker.append("yielded_b")
            yield self._make_resp("b")
            yielded_marker.append("yielded_c")
            yield self._make_resp("c")

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        def request_gen():
            yield _make_request("req1")

        out_iter = self.servicer.ModelStreamInfer(request_gen(), MagicMock())
        out = list(out_iter)

        self.assertEqual([r.error_message for r in out], ["a", "b", "c"])
        # Before emitting "a", buffer pulled past "b"; confirms hold-then-flush.
        self.assertEqual(yielded_marker, ["yielded_a", "yielded_b", "yielded_c"])

    def test_buffer_single_chunk_flushes_on_stream_end(self) -> None:
        """Downstream ends after 1 chunk (e.g., max_new_tokens=1): buffered chunk must flush."""
        self.mock_stub.ModelStreamInfer.return_value = iter([self._make_resp("only")])

        def request_gen():
            yield _make_request("req1")

        out = list(self.servicer.ModelStreamInfer(request_gen(), MagicMock()))
        self.assertEqual([r.error_message for r in out], ["only"])

    def test_buffer_zero_chunk_returns_cleanly(self) -> None:
        """Downstream yields nothing: proxy returns an empty stream without error."""
        self.mock_stub.ModelStreamInfer.return_value = iter([])

        def request_gen():
            yield _make_request("req1")

        out = list(self.servicer.ModelStreamInfer(request_gen(), MagicMock()))
        self.assertEqual(out, [])

    def test_buffer_error_after_first_chunk_flushes_then_raises(self) -> None:
        """Downstream errors after yielding 1 chunk: client still gets that chunk, then error."""

        class FakeRpcError(grpc.RpcError):
            pass

        def downstream_gen():
            yield self._make_resp("a")
            raise FakeRpcError("downstream failed")

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        def request_gen():
            yield _make_request("req1")

        collected = []
        with self.assertRaises(FakeRpcError):
            for r in self.servicer.ModelStreamInfer(request_gen(), MagicMock()):
                collected.append(r)
        self.assertEqual([r.error_message for r in collected], ["a"])

    def test_buffer_error_before_any_chunk_raises_cleanly(self) -> None:
        """Downstream errors on the very first next(): nothing yielded, error propagates."""

        class FakeRpcError(grpc.RpcError):
            pass

        def downstream_gen():
            raise FakeRpcError("downstream failed")
            yield  # pragma: no cover

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()

        def request_gen():
            yield _make_request("req1")

        collected = []
        with self.assertRaises(FakeRpcError):
            for r in self.servicer.ModelStreamInfer(request_gen(), MagicMock()):
                collected.append(r)
        self.assertEqual(collected, [])


class AccessLogDiagInjectionTest(TestCase):
    """Forwarder writes ``downstream_addr`` / ``downstream_resp_count`` /
    ``buffered_stage`` onto the access-log aggregate attached at
    ``context._dash_sc_access_agg``.

    The real aggregate object lives in :mod:`rtp_llm.dash_sc.access_log`;
    we construct one directly and attach it to a fake context, bypassing the
    interceptor. That way the forwarder's write-back path is exercised in
    isolation and every assertion is against a real dataclass field (no mock
    slippage where ``agg.downstream_resp_count += 1`` would silently no-op).
    """

    def setUp(self) -> None:
        self.channel_patcher = patch("grpc.insecure_channel", return_value=MagicMock())
        self.channel_patcher.start()

        # Two-addr config -> randomness in _next_stub; we patch it to deterministic.
        self.servicer = PureForwardServicer(["10.0.0.1:8096", "10.0.0.2:8096"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub, self.mock_stub]

    def tearDown(self) -> None:
        self.servicer.close()
        self.channel_patcher.stop()

    def _ctx_with_agg(self) -> tuple[MagicMock, object]:
        """Return (context, agg) where agg is a real ``_RpcAggregate``."""
        import time as _time

        from rtp_llm.dash_sc.access_log import _RpcAggregate

        agg = _RpcAggregate(
            method="/x.Svc/M",
            stream_type="bidi_stream",
            peer="",
            start_ts=_time.time(),
        )
        ctx = MagicMock()
        ctx._dash_sc_access_agg = agg  # explicit attribute — overrides MagicMock auto
        return ctx, agg

    def _make_resp(self, tag: str) -> predict_v2_pb2.ModelStreamInferResponse:
        resp = _make_response()
        resp.error_message = tag
        return resp

    def _patch_addr(self, idx: int) -> None:
        """Pin ``_next_stub`` to a deterministic addr index."""
        self.servicer._next_stub = lambda: (self.mock_stub, idx)

    def test_downstream_addr_set_to_chosen_backend(self) -> None:
        self._patch_addr(1)
        self.mock_stub.ModelStreamInfer.return_value = iter([self._make_resp("a")])
        ctx, agg = self._ctx_with_agg()

        def request_gen():
            yield _make_request("req1")

        list(self.servicer.ModelStreamInfer(request_gen(), ctx))
        self.assertEqual(agg.downstream_addr, "10.0.0.2:8096")

    def test_downstream_resp_count_tracks_upstream_frames(self) -> None:
        self._patch_addr(0)
        chunks = [self._make_resp("a"), self._make_resp("b"), self._make_resp("c")]
        self.mock_stub.ModelStreamInfer.return_value = iter(chunks)
        ctx, agg = self._ctx_with_agg()

        def request_gen():
            yield _make_request("req1")

        list(self.servicer.ModelStreamInfer(request_gen(), ctx))
        self.assertEqual(agg.downstream_resp_count, 3)

    def test_stage_waiting_first_on_immediate_downstream_error(self) -> None:
        """Downstream errors before any frame -> stuck at waiting_first.

        This is the "backend never produced anything" signature; buffering
        is innocent here, the issue is downstream or the LBS below it.
        """
        self._patch_addr(0)

        class FakeRpcError(grpc.RpcError):
            pass

        def downstream_gen():
            raise FakeRpcError("backend silent")
            yield  # pragma: no cover

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()
        ctx, agg = self._ctx_with_agg()

        def request_gen():
            yield _make_request("req1")

        with self.assertRaises(FakeRpcError):
            list(self.servicer.ModelStreamInfer(request_gen(), ctx))
        self.assertEqual(agg.buffered_stage, "waiting_first")
        self.assertEqual(agg.downstream_resp_count, 0)

    def test_stage_flushed_first_on_single_chunk_clean_end(self) -> None:
        """Downstream ends with exactly one chunk -> flushed_first (normal path)."""
        self._patch_addr(0)
        self.mock_stub.ModelStreamInfer.return_value = iter([self._make_resp("only")])
        ctx, agg = self._ctx_with_agg()

        def request_gen():
            yield _make_request("req1")

        out = list(self.servicer.ModelStreamInfer(request_gen(), ctx))
        self.assertEqual(len(out), 1)
        self.assertEqual(agg.buffered_stage, "flushed_first")
        self.assertEqual(agg.downstream_resp_count, 1)

    def test_stage_flushed_both_on_happy_path(self) -> None:
        """Multi-chunk happy path -> flushed_both after second frame yields."""
        self._patch_addr(0)
        self.mock_stub.ModelStreamInfer.return_value = iter(
            [self._make_resp("a"), self._make_resp("b"), self._make_resp("c")]
        )
        ctx, agg = self._ctx_with_agg()

        def request_gen():
            yield _make_request("req1")

        list(self.servicer.ModelStreamInfer(request_gen(), ctx))
        self.assertEqual(agg.buffered_stage, "flushed_both")
        self.assertEqual(agg.downstream_resp_count, 3)

    def test_stage_flushed_first_on_exception_when_client_consumes(self) -> None:
        """Downstream errors after token 1 and the client fully drains the stream
        -> ``flushed_first_on_exception``: the buffered frame did reach the wire.

        This is the "LBS cut downstream but client still alive" case; the
        forwarder successfully flushed token 1 to the client before re-raising.
        """
        self._patch_addr(0)

        class FakeRpcError(grpc.RpcError):
            pass

        def downstream_gen():
            yield self._make_resp("a")
            raise FakeRpcError("downstream cut after token 1")

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()
        ctx, agg = self._ctx_with_agg()

        def request_gen():
            yield _make_request("req1")

        got: list[str] = []
        with self.assertRaises(FakeRpcError):
            for r in self.servicer.ModelStreamInfer(request_gen(), ctx):
                got.append(r.error_message)
        self.assertEqual(got, ["a"])  # client got token 1
        self.assertEqual(agg.buffered_stage, "flushed_first_on_exception")
        self.assertEqual(agg.downstream_resp_count, 1)

    def test_stage_dropped_buffered_when_client_went_away(self) -> None:
        """Downstream errors after token 1, client-side yield raises (simulated
        via generator.throw) -> ``dropped_buffered_on_exception``.

        This is the pathological HoL case we saw in production: backend
        produced token 1, forwarder buffered it waiting for token 2, client
        disconnected, the buffered frame died with the generator.
        """
        self._patch_addr(0)

        class FakeRpcError(grpc.RpcError):
            pass

        def downstream_gen():
            yield self._make_resp("a")
            raise FakeRpcError("downstream cut after token 1")

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()
        ctx, agg = self._ctx_with_agg()

        def request_gen():
            yield _make_request("req1")

        gen = self.servicer.ModelStreamInfer(request_gen(), ctx)
        # Consumer receives token 1 via the except-path yield; generator is
        # now suspended right after ``yield buffered`` inside the except block.
        first = next(gen)
        self.assertEqual(first.error_message, "a")
        # Simulate the client handler dying — gRPC framework closes the
        # generator, which inside our ``yield buffered`` reads as throw().
        with self.assertRaises(BaseException):
            gen.throw(RuntimeError("client gone"))
        self.assertEqual(agg.buffered_stage, "dropped_buffered_on_exception")


class MetadataPropagationTest(TestCase):
    """Client-sent gRPC metadata must reach the downstream stub verbatim.

    Without this, correlation headers (``x-dashscope-request-id`` etc.) die
    at the forwarder and the backend frontend has no way to tie a
    ``req_count=0`` access-log entry to the originating dashscope-serving
    request.
    """

    def setUp(self) -> None:
        self.channel_patcher = patch("grpc.insecure_channel", return_value=MagicMock())
        self.channel_patcher.start()
        self.servicer = PureForwardServicer(["127.0.0.1:1"])
        self.mock_stub = MagicMock()
        self.servicer._stubs = [self.mock_stub]

    def tearDown(self) -> None:
        self.servicer.close()
        self.channel_patcher.stop()

    def test_client_metadata_forwarded_to_downstream_stub(self) -> None:
        md = [
            ("x-dashscope-request-id", "corr-abc-42"),
            ("x-request-id", "generic-7"),
        ]
        ctx = MagicMock()
        ctx.invocation_metadata.return_value = md
        self.mock_stub.ModelStreamInfer.return_value = iter([_make_response()])

        def request_gen():
            yield _make_request("req1")

        list(self.servicer.ModelStreamInfer(request_gen(), ctx))
        kwargs = self.mock_stub.ModelStreamInfer.call_args.kwargs
        self.assertIn("metadata", kwargs)
        forwarded = list(kwargs["metadata"])
        self.assertEqual(forwarded, md)

    def test_invocation_metadata_raising_does_not_break_forwarder(self) -> None:
        """A context whose ``invocation_metadata()`` raises must fall back to
        empty metadata, not propagate the error and drop the RPC."""
        ctx = MagicMock()
        ctx.invocation_metadata.side_effect = RuntimeError("context teardown race")
        self.mock_stub.ModelStreamInfer.return_value = iter([_make_response()])

        def request_gen():
            yield _make_request("req1")

        out = list(self.servicer.ModelStreamInfer(request_gen(), ctx))
        self.assertEqual(len(out), 1)
        kwargs = self.mock_stub.ModelStreamInfer.call_args.kwargs
        self.assertEqual(tuple(kwargs["metadata"]), ())


class ChannelPoolTest(TestCase):
    """Verify per-addr channel pool: N channels/addr, round-robin over all stubs.

    Default N=1 preserves prior behavior; N>1 multiplies channels per addr and
    ``_next_stub`` cycles through all of them while still returning the correct
    index into ``_forward_addrs`` so ``ModelStreamInfer`` can name the addr.
    """

    def test_default_one_channel_per_addr(self) -> None:
        with patch("grpc.insecure_channel", return_value=MagicMock()) as mock_ch:
            servicer = PureForwardServicer(["10.0.0.1:8096", "10.0.0.2:8096"])
        self.assertEqual(len(servicer._channels), 2)
        self.assertEqual(len(servicer._stubs), 2)
        self.assertEqual(mock_ch.call_count, 2)
        servicer.close()

    def test_pool_round_robin_over_addrs(self) -> None:
        with patch("grpc.insecure_channel", return_value=MagicMock()):
            servicer = PureForwardServicer(
                ["10.0.0.1:8096", "10.0.0.2:8096"], channels_per_addr=3
            )
        # 2 addrs × 3 channels/addr = 6 stubs, addrs-index sequence [0,0,0,1,1,1].
        self.assertEqual(len(servicer._stubs), 6)
        self.assertEqual(servicer._stub_addr_idx, [0, 0, 0, 1, 1, 1])
        # Round-robin cycles through all 6 stubs, returning the owning addr idx.
        seq = [servicer._next_stub()[1] for _ in range(7)]
        self.assertEqual(seq, [0, 0, 0, 1, 1, 1, 0])
        servicer.close()

    def test_parse_channels_per_addr_fallback(self) -> None:
        # Bad input -> default, never raises.
        self.assertEqual(_parse_channels_per_addr(""), 1)
        self.assertEqual(_parse_channels_per_addr("abc"), 1)
        self.assertEqual(_parse_channels_per_addr("0"), 1)
        self.assertEqual(_parse_channels_per_addr("-4"), 1)
        self.assertEqual(_parse_channels_per_addr("4"), 4)


if __name__ == "__main__":
    main()
