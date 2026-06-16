"""End-to-end smoke for ``DashScInferenceServicer`` over a real grpc.aio wire.

Mirrors ``request_smoke_test.py`` (fixture-driven, golden-compared) but bolts a
real ``grpc.aio.Server`` (via :class:`DashScGrpcServer`) in front of the
servicer and drives it with a real ``grpc.aio.insecure_channel`` client. So
the request/response now actually traverses HTTP/2 framing, proto
serialization, and the access-log interceptor — coverage that the in-memory
``await servicer.ModelStreamInfer(...)`` smokes can't give.

Same fixture file as ``request_smoke_test.py``: adding a case = JSON edit only.
"""

from __future__ import annotations

import socket
import unittest

import grpc

from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc
from rtp_llm.dash_sc.server import DashScGrpcServer
from rtp_llm.dash_sc.test.inference.request_smoke_test import (
    _FakeVisitor,
    _assert_generate_config,
    _assert_response,
    _build_fake_stream,
    _build_request,
    _build_servicer,
    _load_cases,
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class DashScGrpcInprocSmokeTest(unittest.IsolatedAsyncioTestCase):
    """One ``test_<case>`` per fixture entry; methods installed below."""


async def _run_case_over_grpc(test: DashScGrpcInprocSmokeTest, case: dict) -> None:
    case_name = case["name"]
    visitor = _FakeVisitor(_build_fake_stream(case))
    servicer = _build_servicer(case, visitor)

    server = DashScGrpcServer()
    port = _free_port()
    await server.start(port=port, servicer=servicer, server_id="0")
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)

            request = _build_request(case)
            metadata = case["request"].get("invocation_metadata") or ()
            kwargs = {"metadata": tuple((k, v) for k, v in metadata)} if metadata else {}

            async def _req_iter():
                yield request

            responses = []
            try:
                async for resp in stub.ModelStreamInfer(_req_iter(), **kwargs):
                    responses.append(resp)
            except grpc.aio.AioRpcError as e:
                test.fail(f"[{case_name}] unexpected grpc error: {e}")

        # ---- enqueue side: fixture goldens against what the codec produced
        expected_enqueue = case.get("expected_enqueue_called")
        if expected_enqueue is not None:
            test.assertEqual(
                visitor.enqueue_called,
                expected_enqueue,
                msg=f"[{case_name}] enqueue_called mismatch",
            )
        if "expected_generate_config" in case:
            test.assertIsNotNone(
                visitor.last_generate_input,
                msg=f"[{case_name}] expected an enqueue but visitor saw none",
            )
            _assert_generate_config(
                test,
                case_name,
                case["expected_generate_config"],
                visitor.last_generate_input.generate_config,
            )
        if "expected_headers" in case:
            test.assertIsNotNone(visitor.last_generate_input)
            test.assertEqual(
                visitor.last_generate_input.headers,
                case["expected_headers"],
                msg=f"[{case_name}] headers mismatch",
            )

        # ---- response side: golden-compare every frame the client received
        expected_responses = case.get("expected_responses", [])
        test.assertEqual(
            len(responses),
            len(expected_responses),
            msg=(
                f"[{case_name}] response count {len(responses)} "
                f"!= expected {len(expected_responses)}"
            ),
        )
        for idx, golden in enumerate(expected_responses):
            _assert_response(test, case_name, idx, golden, responses[idx])
    finally:
        # ``DashScGrpcServer.stop`` is the cross-thread variant; we awaited
        # ``start`` directly on the test loop so we shut the underlying aio
        # server down in-place. ``servicer.close`` is a no-op for inference.
        if server._server is not None:
            await server._server.stop(None)


def _install_case_methods() -> None:
    """Attach one ``test_<case_name>`` per fixture entry so each shows up
    individually in bazel output (mirrors ``request_smoke_test._install_…``)."""
    for case in _load_cases():

        async def _method(self, _case=case) -> None:
            await _run_case_over_grpc(self, _case)

        method_name = f"test_{case['name']}"
        _method.__name__ = method_name
        setattr(DashScGrpcInprocSmokeTest, method_name, _method)


_install_case_methods()


if __name__ == "__main__":
    unittest.main()
