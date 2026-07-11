"""Unit tests for ``rtp_llm.dash_sc.proxy.servicer`` (grpc.aio).

Tests verify the async request_iterator is passed correctly to the backend
stub, first-chunk buffering behavior, access-log diagnostic injection, metadata
propagation, and the per-addr channel cache.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import unittest
from unittest.mock import MagicMock, patch

import grpc

from rtp_llm.dash_sc.access_record import GrpcAccessRecord
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.proxy.service_route import BackendAddr, VipServerServiceDiscovery
from rtp_llm.dash_sc.proxy.service_route_config import (
    LEGACY_FORWARD_ENV_KEY,
    SERVICE_ROUTE_ENV_KEY,
    load_service_route_config_from_env,
    parse_service_route_config,
)
from rtp_llm.dash_sc.proxy.servicer import DashScProxyServicer
from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool


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


def _make_finished_response() -> predict_v2_pb2.ModelStreamInferResponse:
    resp = predict_v2_pb2.ModelStreamInferResponse()
    infer = resp.infer_response
    out = infer.outputs.add()
    out.name = "finished"
    out.datatype = "BOOL"
    out.shape.append(1)
    infer.raw_output_contents.append(b"\x01")
    return resp


def _dash_error_payload(resp) -> tuple[int, dict]:
    infer = resp.infer_response
    return (
        infer.parameters["error_no"].int64_param,
        json.loads(infer.parameters["error_msg"].string_param),
    )


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


class _FakeChannel:
    def __init__(self, addr: str = ""):
        self.addr = addr
        self.closed = False

    def get_state(self, try_to_connect: bool = False):
        return grpc.ChannelConnectivity.READY

    async def close(self):
        self.closed = True


async def _drain(aiter):
    return [x async for x in aiter]


async def _request_gen(*reqs):
    for r in reqs:
        yield r


def _install_mock_stub(servicer, mock_stub) -> None:
    """Make downstream stub construction return ``mock_stub`` in a test."""
    patcher = patch(
        "rtp_llm.dash_sc.proto.predict_v2_pb2_grpc.GRPCInferenceServiceStub",
        return_value=mock_stub,
    )
    patcher.start()
    servicer._test_stub_patcher = patcher


def _stop_mock_stub(servicer) -> None:
    patcher = getattr(servicer, "_test_stub_patcher", None)
    if patcher is not None:
        patcher.stop()
        servicer._test_stub_patcher = None


def _servicer_pool(servicer):
    return servicer._channel_pool


def _make_servicer(
    forward_addrs: list[str],
) -> DashScProxyServicer:
    address = ";".join(forward_addrs)
    saved_route = os.environ.get(SERVICE_ROUTE_ENV_KEY)
    try:
        os.environ[SERVICE_ROUTE_ENV_KEY] = json.dumps(
            {"type": "ip_port_list", "address": address}
        )
        return DashScProxyServicer()
    finally:
        if saved_route is None:
            os.environ.pop(SERVICE_ROUTE_ENV_KEY, None)
        else:
            os.environ[SERVICE_ROUTE_ENV_KEY] = saved_route


class ServiceRouteConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_route = os.environ.pop(SERVICE_ROUTE_ENV_KEY, None)
        self._saved_legacy = os.environ.pop(LEGACY_FORWARD_ENV_KEY, None)

    def tearDown(self) -> None:
        if self._saved_route is None:
            os.environ.pop(SERVICE_ROUTE_ENV_KEY, None)
        else:
            os.environ[SERVICE_ROUTE_ENV_KEY] = self._saved_route
        if self._saved_legacy is None:
            os.environ.pop(LEGACY_FORWARD_ENV_KEY, None)
        else:
            os.environ[LEGACY_FORWARD_ENV_KEY] = self._saved_legacy

    def test_parse_supported_route_types(self) -> None:
        cfg = parse_service_route_config(
            '{"type": "ip_port_list", "address": ' '"10.0.0.1:8096;10.0.0.2:8096"}'
        )
        self.assertEqual(
            (cfg.type, cfg.address),
            ("ip_port_list", "10.0.0.1:8096;10.0.0.2:8096"),
        )

        cfg = parse_service_route_config(
            '{"type": "vipserver", "address": "com.example.svc"}'
        )
        self.assertEqual((cfg.type, cfg.address), ("vipserver", "com.example.svc"))

        with self.assertRaisesRegex(RuntimeError, "unsupported 'type': 'ip_port'"):
            parse_service_route_config(
                '{"type": "ip_port", "address": "10.0.0.1:8096"}'
            )

        with self.assertRaisesRegex(RuntimeError, "unsupported 'type': 'host_port'"):
            parse_service_route_config(
                '{"type": "host_port", "address": "frontend.example:8096"}'
            )

        with self.assertRaisesRegex(RuntimeError, "'type' and 'address'"):
            parse_service_route_config(
                '{"type": "ip_port_list", "address": '
                '["10.0.0.1:8096", "10.0.0.2:8096"]}'
            )

    def test_legacy_forward_addr_converts_to_ip_port_list(self) -> None:
        os.environ[LEGACY_FORWARD_ENV_KEY] = "10.0.0.1:8096, 10.0.0.2:8096"

        cfg = load_service_route_config_from_env()

        self.assertEqual(cfg.type, "ip_port_list")
        self.assertEqual(cfg.address, "10.0.0.1:8096;10.0.0.2:8096")

    def test_legacy_forward_addr_json_array_converts_to_ip_port_list(self) -> None:
        os.environ[LEGACY_FORWARD_ENV_KEY] = (
            '["10.0.0.1:8096", "10.0.0.2:8096", "10.0.0.1:8096"]'
        )

        cfg = load_service_route_config_from_env()

        self.assertEqual(cfg.type, "ip_port_list")
        self.assertEqual(cfg.address, "10.0.0.1:8096;10.0.0.2:8096")

    def test_service_route_wins_over_legacy_forward_addr(self) -> None:
        os.environ[SERVICE_ROUTE_ENV_KEY] = (
            '{"type": "vipserver", "address": "com.example.svc"}'
        )
        os.environ[LEGACY_FORWARD_ENV_KEY] = "10.0.0.1:8096"

        cfg = load_service_route_config_from_env()

        self.assertEqual(cfg.type, "vipserver")
        self.assertEqual(cfg.address, "com.example.svc")

    def test_vipserver_discovery_resolves_one_host_from_resolver(self) -> None:
        fake_host = MagicMock(ip="10.0.0.1", port=8096)
        discovery = VipServerServiceDiscovery(
            "com.example.svc",
            resolver=lambda _domain: fake_host,
        )

        addr = discovery.resolve()
        self.assertIsNotNone(addr)
        assert addr is not None
        self.assertEqual(addr.ip, "10.0.0.1")
        self.assertEqual(addr.http_port, 8096)
        self.assertEqual(addr.grpc_port, 8104)
        self.assertEqual(addr.grpc_target, "10.0.0.1:8104")

    def test_vipserver_discovery_logs_resolver_error(self) -> None:
        def resolver(_domain: str):
            raise RuntimeError("vipserver failed")

        discovery = VipServerServiceDiscovery("com.example.svc", resolver=resolver)

        with self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(discovery.resolve())

        self.assertTrue(
            any("vipserver resolve failed" in msg for msg in logs.output),
            logs.output,
        )

    def test_servicer_reads_service_route_env(self) -> None:
        os.environ[SERVICE_ROUTE_ENV_KEY] = (
            '{"type": "ip_port_list", "address": ' '"10.0.0.1:8096;10.0.0.2:8096"}'
        )
        servicer = DashScProxyServicer()
        self.assertEqual(
            [addr.http_target for addr in servicer._discovery._addrs],
            ["10.0.0.1:8096", "10.0.0.2:8096"],
        )
        self.assertEqual(
            [addr.grpc_target for addr in servicer._discovery._addrs],
            ["10.0.0.1:8104", "10.0.0.2:8104"],
        )


class IteratorBehaviorTest(unittest.IsolatedAsyncioTestCase):
    """Verify the async request iterator is passed through to the downstream stub."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        )
        self.channel_patcher.start()

        self.servicer = _make_servicer(["127.0.0.1:1"])
        await self.servicer.open()
        self.mock_stub = MagicMock()
        _install_mock_stub(self.servicer, self.mock_stub)

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        _stop_mock_stub(self.servicer)
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


class ParameterValidationTest(unittest.IsolatedAsyncioTestCase):
    """Proxy rejects bad first-frame sampling params before forwarding."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        )
        self.channel_patcher.start()

        self.servicer = _make_servicer(["127.0.0.1:1"])
        self.mock_stub = MagicMock()
        _install_mock_stub(self.servicer, self.mock_stub)

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        _stop_mock_stub(self.servicer)
        self.channel_patcher.stop()

    def _assert_parameter_error(self, responses) -> None:
        self.assertEqual(len(responses), 1)
        self.assertFalse(responses[0].error_message)
        infer = responses[0].infer_response
        self.assertEqual(infer.parameters["error_no"].int64_param, 8)
        payload = json.loads(infer.parameters["error_msg"].string_param)
        self.assertEqual(payload["status_code"], 400)
        self.assertEqual(payload["status_name"], "InvalidParameter")

    async def test_max_completion_tokens_non_positive_forwarded(
        self,
    ) -> None:
        for value in (-1, 0):
            with self.subTest(value=value):
                self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
                    [_make_response()]
                )
                req = _make_request("req1")
                req.parameters["max_completion_tokens"].int64_param = value

                responses = await _drain(
                    self.servicer.ModelStreamInfer(_request_gen(req), MagicMock())
                )

                self.assertEqual(len(responses), 1)
                self.mock_stub.ModelStreamInfer.assert_called_once()
                self.mock_stub.reset_mock()

    async def test_max_new_tokens_non_positive_rejected_without_forward(
        self,
    ) -> None:
        for value in (-1, 0):
            with self.subTest(value=value):
                req = _make_request("req1")
                req.parameters["max_new_tokens"].int64_param = value

                responses = await _drain(
                    self.servicer.ModelStreamInfer(_request_gen(req), MagicMock())
                )

                self._assert_parameter_error(responses)
                self.mock_stub.ModelStreamInfer.assert_not_called()
                self.mock_stub.reset_mock()

    async def test_structural_tag_is_not_parsed_by_proxy_validation_hot_path(
        self,
    ) -> None:
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter([_make_response()])
        req = _make_request("req1")
        req.parameters["max_new_tokens"].int64_param = 16
        req.parameters["tool_call_structural_tag"].string_param = "not-json"

        responses = await _drain(
            self.servicer.ModelStreamInfer(_request_gen(req), MagicMock())
        )

        self.assertEqual(len(responses), 1)
        self.mock_stub.ModelStreamInfer.assert_called_once()


class BufferFirstTokenTest(unittest.IsolatedAsyncioTestCase):
    """First-chunk buffering preserves order, flushes on single-chunk end,
    re-raises cleanly on error."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        )
        self.channel_patcher.start()

        self.servicer = _make_servicer(["127.0.0.1:1"])
        await self.servicer.open()
        self.mock_stub = MagicMock()
        _install_mock_stub(self.servicer, self.mock_stub)

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        _stop_mock_stub(self.servicer)
        self.channel_patcher.stop()

    def _make_resp(self, tag: str) -> predict_v2_pb2.ModelStreamInferResponse:
        resp = _make_response()
        resp.infer_response.id = tag
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

        self.assertEqual([r.infer_response.id for r in out], ["a", "b", "c"])
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
        self.assertEqual([r.infer_response.id for r in out], ["only"])

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
        self.assertEqual([r.infer_response.id for r in collected], ["a"])

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
    ``buffered_stage`` onto the forward access record attached at the gRPC
    context."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        )
        self.channel_patcher.start()

        self.servicer = _make_servicer(["10.0.0.1:8096", "10.0.0.2:8096"])
        await self.servicer.open()
        self.mock_stub = MagicMock()
        _install_mock_stub(self.servicer, self.mock_stub)

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        _stop_mock_stub(self.servicer)
        self.channel_patcher.stop()

    def _ctx(self) -> MagicMock:
        # The servicer now owns record creation: ``ModelStreamInfer`` calls
        # ``GrpcAccessRecord.create(context, ...)`` at its top and attaches the
        # record to the context. Tests inspect that servicer-built record via
        # ``_record_of(ctx)`` after the call (no pre-attached record to orphan).
        return MagicMock()

    @staticmethod
    def _record_of(ctx) -> GrpcAccessRecord:
        record = GrpcAccessRecord.from_context(ctx)
        assert record is not None, "servicer did not attach an access record"
        return record

    def _make_resp(self, tag: str) -> predict_v2_pb2.ModelStreamInferResponse:
        resp = _make_response()
        resp.infer_response.id = tag
        return resp

    def _patch_addr(self, idx: int) -> None:
        # Force discovery to resolve the backend address used by this test.
        addrs = (
            BackendAddr.from_http_target("10.0.0.1:8096"),
            BackendAddr.from_http_target("10.0.0.2:8096"),
        )
        self.servicer._discovery._addrs = (addrs[idx],)

    async def test_backend_addr_set_to_chosen_backend(self) -> None:
        self._patch_addr(1)
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [self._make_resp("a")]
        )
        ctx = self._ctx()

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        self.assertEqual(self._record_of(ctx).backend_addr, "10.0.0.2:8104")

    async def test_backend_resp_count_tracks_upstream_frames(self) -> None:
        self._patch_addr(0)
        chunks = [self._make_resp("a"), self._make_resp("b"), self._make_resp("c")]
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(chunks)
        ctx = self._ctx()

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        self.assertEqual(self._record_of(ctx).backend_resp_count, 3)

    async def test_forward_summary_omits_raw_token_payload(self) -> None:
        # no-structured-payload contract: the forwarder logs a forward_summary
        # line and must never carry token ids or frontend-only statistics.
        self._patch_addr(0)
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [self._make_resp("a")]
        )
        ctx = self._ctx()

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        payload = self._record_of(ctx).build_record(None, None)
        self.assertEqual(payload["capture_mode"], "forward_summary")
        for field in (
            "latency_ttft_ms",
            "latency_tpot_ms",
            "first_token_ts_epoch_ms",
            "input_token_len",
            "backend_input_token_len",
            "output_token_len",
            "finish_reason",
            "finished",
            "terminal_seen",
            "prompt_cached_token_num",
            "token_frame_count",
            "empty_frame_count",
            "finished_only_frame_count",
            "multi_token_frame_count",
            "max_tokens_per_frame",
            "generate_config",
            "generate_config_role_addrs",
            "input_ids",
            "generated_ids",
            "repetition_monitor_impl",
            "repetition_monitor_available",
            "repetition_monitor_unavailable_reason",
            "tool_call_loop_impl",
            "tool_call_loop_error",
        ):
            self.assertNotIn(field, payload)

    async def test_finished_frame_marks_terminal_without_struct_stats(self) -> None:
        self._patch_addr(0)
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [_make_response(), _make_finished_response()]
        )
        ctx = self._ctx()

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        record = self._record_of(ctx)
        payload = record.build_record(None, None)
        self.assertTrue(record.terminal_seen)
        self.assertIsNotNone(payload["finished_ts_epoch_ms"])
        self.assertNotIn("terminal_seen", payload)

    async def test_stage_waiting_first_on_immediate_downstream_error(self) -> None:
        self._patch_addr(0)

        class FakeRpcError(grpc.RpcError):
            pass

        async def downstream_gen():
            raise FakeRpcError("backend silent")
            yield  # pragma: no cover

        self.mock_stub.ModelStreamInfer.return_value = downstream_gen()
        ctx = self._ctx()

        with self.assertRaises(FakeRpcError):
            await _drain(
                self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
            )
        record = self._record_of(ctx)
        self.assertEqual(record.buffered_stage, "waiting_first")
        self.assertEqual(record.backend_resp_count, 0)

    async def test_stage_flushed_first_on_single_chunk_clean_end(self) -> None:
        self._patch_addr(0)
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [self._make_resp("only")]
        )
        ctx = self._ctx()

        out = await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        record = self._record_of(ctx)
        self.assertEqual(len(out), 1)
        self.assertEqual(record.buffered_stage, "flushed_first")
        self.assertEqual(record.backend_resp_count, 1)

    async def test_stage_flushed_both_on_happy_path(self) -> None:
        self._patch_addr(0)
        self.mock_stub.ModelStreamInfer.return_value = _AsyncIter(
            [self._make_resp("a"), self._make_resp("b"), self._make_resp("c")]
        )
        ctx = self._ctx()

        await _drain(
            self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        record = self._record_of(ctx)
        self.assertEqual(record.buffered_stage, "flushed_both")
        self.assertEqual(record.backend_resp_count, 3)

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
        ctx = self._ctx()

        got: list[str] = []
        with self.assertRaises(FakeRpcError):
            async for r in self.servicer.ModelStreamInfer(
                _request_gen(_make_request("req1")), ctx
            ):
                got.append(r.infer_response.id)
        record = self._record_of(ctx)
        self.assertEqual(got, ["a"])
        self.assertEqual(record.buffered_stage, "flushed_first_on_exception")
        self.assertEqual(record.backend_resp_count, 1)

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
        ctx = self._ctx()

        gen = self.servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        # Consume first frame.
        first = await gen.__anext__()
        self.assertEqual(first.infer_response.id, "a")
        # Simulate client handler dying — gRPC framework closes the generator,
        # which inside our ``yield buffered`` reads as ``athrow()``.
        with self.assertRaises(BaseException):
            await gen.athrow(RuntimeError("client gone"))
        self.assertEqual(
            self._record_of(ctx).buffered_stage, "dropped_buffered_on_exception"
        )


class MetadataPropagationTest(unittest.IsolatedAsyncioTestCase):
    """Client-sent gRPC metadata must reach the downstream stub verbatim."""

    async def asyncSetUp(self) -> None:
        self.channel_patcher = patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        )
        self.channel_patcher.start()
        self.servicer = _make_servicer(["127.0.0.1:1"])
        await self.servicer.open()
        self.mock_stub = MagicMock()
        _install_mock_stub(self.servicer, self.mock_stub)

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        _stop_mock_stub(self.servicer)
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


class ChannelLoopAffinityTest(unittest.TestCase):
    """Outbound aio channels are opened on the request/server loop, not __init__."""

    def test_channels_open_on_request_loop_after_sync_construction(self) -> None:
        created_loops: list[asyncio.AbstractEventLoop] = []
        mock_stub = MagicMock()

        def make_channel(*_args, **_kwargs):
            created_loops.append(asyncio.get_running_loop())

            class _Channel:
                async def close(self):
                    return None

            return _Channel()

        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=make_channel,
        ) as mock_ch:
            with patch(
                "rtp_llm.dash_sc.proto.predict_v2_pb2_grpc.GRPCInferenceServiceStub",
                return_value=mock_stub,
            ):
                servicer = _make_servicer(["127.0.0.1:1"])
                self.assertEqual(mock_ch.call_count, 0)

                async def invoke():
                    running_loop = asyncio.get_running_loop()
                    mock_stub.ModelStreamInfer.return_value = _AsyncIter(
                        [_make_response()]
                    )
                    out = await _drain(
                        servicer.ModelStreamInfer(
                            _request_gen(_make_request("req1")), MagicMock()
                        )
                    )
                    self.assertEqual(len(out), 1)
                    self.assertEqual(created_loops, [running_loop])
                    await servicer.close()

                asyncio.run(invoke())


class ChannelPoolTest(unittest.IsolatedAsyncioTestCase):
    """Dash SC reuses the shared lazy per-address gRPC channel cache."""

    async def test_open_does_not_prewarm_configured_addrs(self) -> None:
        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        ) as mock_ch:
            servicer = _make_servicer(["10.0.0.1:8096", "10.0.0.2:8096"])
            self.assertEqual(_servicer_pool(servicer)._channels, {})
            self.assertEqual(mock_ch.call_count, 0)
            await servicer.open()
            self.assertEqual(_servicer_pool(servicer)._channels, {})
            self.assertEqual(mock_ch.call_count, 0)
            await servicer.close()

    async def test_servicer_uses_grpc_port_for_channel_target(self) -> None:
        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        ) as mock_ch:
            servicer = _make_servicer(["10.0.0.1:8096"])
            await servicer.open()
            mock_stub = MagicMock()
            _install_mock_stub(servicer, mock_stub)
            try:
                mock_stub.ModelStreamInfer.return_value = _AsyncIter([_make_response()])

                await _drain(
                    servicer.ModelStreamInfer(
                        _request_gen(_make_request("req1")), MagicMock()
                    )
                )

                self.assertEqual(mock_ch.call_args[0][0], "10.0.0.1:8104")
            finally:
                _stop_mock_stub(servicer)
                await servicer.close()

    async def test_shared_pool_reuses_cached_channel_by_addr(self) -> None:
        pool = GrpcHostChannelPool(cleanup_interval=60)
        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        ) as mock_ch:
            ch1 = await pool.get("10.0.0.1:8096")
            ch2 = await pool.get("10.0.0.1:8096")
            ch3 = await pool.get("10.0.0.2:8096")

        self.assertIs(ch1, ch2)
        self.assertIsNot(ch1, ch3)
        self.assertEqual(mock_ch.call_count, 2)
        await pool.close()

    async def test_vipserver_discovery_delegates_each_resolve_to_resolver(self) -> None:
        snapshots = [
            MagicMock(ip="10.0.0.1", port=8096),
            MagicMock(ip="10.0.0.2", port=8096),
        ]

        def resolver(_domain: str):
            return snapshots.pop(0)

        discovery = VipServerServiceDiscovery(
            "com.example.svc",
            resolver=resolver,
        )

        first = discovery.resolve()
        second = discovery.resolve()
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertEqual(first.http_target, "10.0.0.1:8096")
        self.assertEqual(first.grpc_target, "10.0.0.1:8104")
        self.assertEqual(second.http_target, "10.0.0.2:8096")
        self.assertEqual(second.grpc_target, "10.0.0.2:8104")

    async def test_closed_pool_returns_dash_503_error_frame(self) -> None:
        with patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        ):
            servicer = _make_servicer(["10.0.0.1:8096"])
            await servicer.open()
        await servicer.close()

        ctx = MagicMock()

        responses = await _drain(
            servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )
        ctx.abort.assert_not_called()
        self.assertEqual(len(responses), 1)
        error_no, payload = _dash_error_payload(responses[0])
        self.assertEqual(error_no, 5)
        self.assertEqual(payload["status_code"], 503)
        self.assertEqual(payload["status_name"], "ServiceUnavailable")
        self.assertEqual(payload["status_message"], "forward backend unavailable")

    async def test_discovery_none_returns_dash_503_error_frame(self) -> None:
        servicer = _make_servicer(["10.0.0.1:8096"])
        servicer._discovery.resolve = MagicMock(return_value=None)
        ctx = MagicMock()

        responses = await _drain(
            servicer.ModelStreamInfer(_request_gen(_make_request("req1")), ctx)
        )

        ctx.abort.assert_not_called()
        self.assertEqual(len(responses), 1)
        error_no, payload = _dash_error_payload(responses[0])
        self.assertEqual(error_no, 5)
        self.assertEqual(payload["status_code"], 503)
        await servicer.close()


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
        self.channel_patcher = patch(
            "rtp_llm.utils.grpc_host_channel_pool.aio.insecure_channel",
            side_effect=lambda addr, **_kwargs: _FakeChannel(addr),
        )
        self.channel_patcher.start()

        self.servicer = _make_servicer(["127.0.0.1:1"])
        await self.servicer.open()
        self.mock_stub = MagicMock()
        _install_mock_stub(self.servicer, self.mock_stub)

    async def asyncTearDown(self) -> None:
        await self.servicer.close()
        _stop_mock_stub(self.servicer)
        self.channel_patcher.stop()

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

    async def test_close_timing_with_counting_wrapper(self) -> None:
        """``agg != None`` path — ``counting_response_iter`` wraps the call.
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

        # ``ModelStreamInfer`` creates + attaches its own record at the top, so
        # the test only needs a bare context here.
        collected = []
        async for resp in self.servicer.ModelStreamInfer(
            _request_gen(_make_request("req1")), MagicMock()
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
