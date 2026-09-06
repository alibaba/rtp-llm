"""Trace-attribute contract for the FlexLB schedule CLIENT span.

Exercises the real MasterClient._send_schedule_request rather than a subclass
override, because the attribute writes under test live in that method. The gRPC
stub and the span factory are the only replacements, so the assertions cover the
production code path.
"""

import asyncio
import unittest
from unittest import mock

import grpc

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.py_config_modules import MasterConfig
from rtp_llm.server.master_client import SUCCESS_CODE, MasterClient
from rtp_llm.telemetry import attributes as trace_attrs


class _RecordingSpan:
    """Stands in for the telemetry span so no provider or exporter is needed."""

    def __init__(self):
        self.attributes = {}
        self.finish_calls = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def finish(self, **kwargs):
        self.finish_calls.append(kwargs)


class _FakeResponse:
    def __init__(self, code):
        self.code = code


class _FakeStub:
    """Returns a canned response, or raises, from Schedule()."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.received_metadata = None
        self.cancel_reasons = []
        self.cancel_metadata = None

    async def Schedule(self, request_pb, timeout=None, metadata=None):
        self.received_metadata = metadata
        if self._error is not None:
            raise self._error
        return self._response

    async def Cancel(self, request_pb, timeout=None, metadata=None):
        # Present so the deadline and cancellation paths exercise their real
        # best-effort cancel instead of dying on a missing attribute.
        self.cancel_reasons.append(request_pb.reason)
        self.cancel_metadata = metadata


def _rpc_error(code):
    """An AioRpcError carrying a specific status code, built without a channel."""
    return grpc.aio.AioRpcError(code, grpc.aio.Metadata(), grpc.aio.Metadata())


class MasterClientScheduleSpanTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = MasterClient(MasterConfig())
        self.span = _RecordingSpan()

    async def _send(self, stub):
        """Drives the real method with the span factory and stub replaced."""
        with mock.patch.object(
            self.client, "_get_channel", return_value=object()
        ), mock.patch(
            "rtp_llm.server.master_client.FlexlbServiceStub", return_value=stub
        ), mock.patch(
            "rtp_llm.server.master_client.start_client_span",
            return_value=(
                self.span,
                [("traceparent", "00-" + "1" * 32 + "-" + "2" * 16 + "-01")],
            ),
        ):
            return await self.client._send_schedule_request(
                "127.0.0.1:7001", mock.MagicMock(priority=50), 1.0, 3540218608800727041
            )

    async def test_success_records_request_id_status_and_business_code(self):
        stub = _FakeStub(response=_FakeResponse(SUCCESS_CODE))

        response = await self._send(stub)

        self.assertEqual(SUCCESS_CODE, response.code)
        attributes = self.span.attributes
        # Double-write: the platform indexes the unprefixed string key for span
        # search, so the numeric twin alone leaves the span unsearchable.
        self.assertEqual("3540218608800727041", attributes[trace_attrs.REQUEST_ID])
        self.assertIsInstance(attributes[trace_attrs.REQUEST_ID], str)
        self.assertEqual(
            3540218608800727041, attributes[trace_attrs.RTP_LLM_REQUEST_ID]
        )
        self.assertIsInstance(attributes[trace_attrs.RTP_LLM_REQUEST_ID], int)
        self.assertEqual("OK", attributes[trace_attrs.RPC_RESPONSE_STATUS_CODE])
        self.assertEqual(SUCCESS_CODE, attributes[trace_attrs.RTP_LLM_SCHEDULE_CODE])
        # A successful schedule closes clean.
        self.assertEqual([{}], self.span.finish_calls)

        # rpc.system on this span breaks the platform's Total tokens aggregation.
        self.assertNotIn("rpc.system", self.span.attributes)

    async def test_rejecting_business_code_does_not_close_as_success(self):
        stub = _FakeStub(response=_FakeResponse(8402))

        # A rejecting code is returned, not raised: the caller decides to raise on
        # it. The span must still not report success.
        response = await self._send(stub)

        self.assertEqual(8402, response.code)
        attributes = self.span.attributes
        # The transport did succeed, and that is recorded, but the span must not
        # report success while carrying a rejecting business code.
        self.assertEqual("OK", attributes[trace_attrs.RPC_RESPONSE_STATUS_CODE])
        self.assertEqual(8402, attributes[trace_attrs.RTP_LLM_SCHEDULE_CODE])
        self.assertEqual(
            [{"error_type": "FlexlbBusinessRejected"}], self.span.finish_calls
        )

    async def test_rpc_error_records_the_canonical_status_code(self):
        # DEADLINE_EXCEEDED is the one transport failure this method raises on.
        stub = _FakeStub(error=_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED))

        with self.assertRaises(FtRuntimeException):
            await self._send(stub)

        self.assertEqual(
            "DEADLINE_EXCEEDED",
            self.span.attributes[trace_attrs.RPC_RESPONSE_STATUS_CODE],
        )
        self.assertEqual(1, len(self.span.finish_calls))
        self.assertEqual("RpcError", self.span.finish_calls[0]["error_type"])
        self.assertTrue(stub.cancel_metadata)
        self.assertEqual(stub.received_metadata, stub.cancel_metadata)

    async def test_cancellation_is_distinguished_from_an_rpc_error(self):
        stub = _FakeStub(error=asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await self._send(stub)

        self.assertEqual(1, len(self.span.finish_calls))
        self.assertEqual("Cancelled", self.span.finish_calls[0]["error_type"])
        self.assertTrue(stub.cancel_metadata)
        self.assertEqual(stub.received_metadata, stub.cancel_metadata)
        # A cancellation has no gRPC status of its own to report.
        self.assertNotIn(trace_attrs.RPC_RESPONSE_STATUS_CODE, self.span.attributes)

    async def test_request_id_and_status_survive_a_swallowed_transport_failure(self):
        # Every transport failure other than a deadline is swallowed into a None
        # return. The span is the only remaining record of it, so it must carry
        # both the request id and the real status code.
        stub = _FakeStub(error=_rpc_error(grpc.StatusCode.UNAVAILABLE))

        self.assertIsNone(await self._send(stub))

        self.assertEqual(
            "3540218608800727041", self.span.attributes[trace_attrs.REQUEST_ID]
        )
        self.assertEqual(
            "UNAVAILABLE", self.span.attributes[trace_attrs.RPC_RESPONSE_STATUS_CODE]
        )
        self.assertEqual("RpcError", self.span.finish_calls[0]["error_type"])

    async def test_disabled_telemetry_leaves_the_call_untouched(self):
        # start_client_span returns (None, None) when tracing is off, which is the
        # default deployment; the schedule must still work.
        stub = _FakeStub(response=_FakeResponse(SUCCESS_CODE))
        with mock.patch.object(
            self.client, "_get_channel", return_value=object()
        ), mock.patch(
            "rtp_llm.server.master_client.FlexlbServiceStub", return_value=stub
        ), mock.patch(
            "rtp_llm.server.master_client.start_client_span", return_value=(None, None)
        ):
            response = await self.client._send_schedule_request(
                "127.0.0.1:7001", mock.MagicMock(priority=50), 1.0, 42
            )

        self.assertEqual(SUCCESS_CODE, response.code)
        self.assertEqual({}, self.span.attributes)
        self.assertIsNone(stub.received_metadata)

    async def test_disabled_telemetry_still_cancels_without_metadata(self):
        stub = _FakeStub(error=asyncio.CancelledError())
        with mock.patch.object(
            self.client, "_get_channel", return_value=object()
        ), mock.patch(
            "rtp_llm.server.master_client.FlexlbServiceStub", return_value=stub
        ), mock.patch(
            "rtp_llm.server.master_client.start_client_span", return_value=(None, [])
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self.client._send_schedule_request(
                    "127.0.0.1:7001", mock.MagicMock(priority=50), 1.0, 42
                )
        self.assertEqual(1, len(stub.cancel_reasons))
        self.assertIsNone(stub.cancel_metadata)


if __name__ == "__main__":
    unittest.main()
