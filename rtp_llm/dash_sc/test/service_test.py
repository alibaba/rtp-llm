"""Unit tests for ``rtp_llm.dash_sc.service`` (grpc.aio).

Covers:
- ``iter_real_model_stream_infer``: success, empty-stream fallback, exception propagation.
- ``DashScGrpcInferenceServicer.ModelStreamInfer``: fake mode, real mode,
  missing input_ids, request_id snowflake scheme alignment with HTTP
  ``generate_request_id``.
"""

from __future__ import annotations

import struct
import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.dash_sc.codec import OtherParams, SamplingParams
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.service import (
    DashScGrpcInferenceServicer,
    iter_real_model_stream_infer,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


def _add_input_tensor(
    req: predict_v2_pb2.ModelInferRequest,
    name: str,
    datatype: str,
    shape: list[int],
    raw: bytes,
) -> None:
    inp = req.inputs.add()
    inp.name = name
    inp.datatype = datatype
    inp.shape[:] = shape
    req.raw_input_contents.append(raw)


def _unpack_int32_le(raw: bytes) -> list[int]:
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


class _FakeAsyncStream:
    """Simple async iterator over a fixed chunk list, with optional error injection."""

    def __init__(self, chunks, raise_after: int | None = None):
        self._chunks = list(chunks)
        self._raise_after = raise_after
        self._emitted = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._raise_after is not None and self._emitted >= self._raise_after:
            raise RuntimeError("backend down")
        if self._emitted >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._emitted]
        self._emitted += 1
        return item


class _FakeVisitor:
    """Async ``enqueue`` that returns a prebuilt ``_FakeAsyncStream``."""

    def __init__(self, stream: _FakeAsyncStream):
        self._stream = stream
        self.enqueue_called = 0

    async def enqueue(self, _generate_input):
        self.enqueue_called += 1
        return self._stream


async def _drain(aiter):
    return [x async for x in aiter]


class IterRealModelStreamInferTest(unittest.IsolatedAsyncioTestCase):
    def _minimal_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "trace-real"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 1, 2))
        return req

    async def test_yields_one_chunk_from_mock_enqueue(self) -> None:
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([3, 4], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        infer = chunks[0].infer_response
        self.assertEqual(infer.id, "trace-real")
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [3, 4])

    async def test_empty_list_yields_error_response(self) -> None:
        req = self._minimal_request()
        visitor = _FakeVisitor(_FakeAsyncStream([]))

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("empty outputs_list", chunks[0].error_message)

    async def test_enqueue_exception_yields_error_message(self) -> None:
        req = self._minimal_request()

        class _BoomVisitor:
            async def enqueue(self, _gi):
                raise RuntimeError("backend down")

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                _BoomVisitor(),
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("backend down", chunks[0].error_message)

    async def test_stream_exception_yields_error_message(self) -> None:
        req = self._minimal_request()
        visitor = _FakeVisitor(_FakeAsyncStream([], raise_after=0))

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=1,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("backend down", chunks[0].error_message)


class IterRealModelStreamInferEchoTest(unittest.IsolatedAsyncioTestCase):
    """Echo-prefill integration for ``iter_real_model_stream_infer``."""

    def _req(self, req_id: str = "echo-trace") -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = req_id
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 99, 100))
        return req

    async def _run(self, *, input_ids, echo_prefix_ids, upstream_ids):
        chunks_proto = []
        for ids in upstream_ids:
            out = GenerateOutput(
                output_ids=torch.tensor(ids, dtype=torch.int32) if ids else None,
                finished=False,
                aux_info=AuxInfo(input_len=len(input_ids), reuse_len=0),
            )
            chunks_proto.append(GenerateOutputs(generate_outputs=[out]))
        visitor = _FakeVisitor(_FakeAsyncStream(chunks_proto))
        return await _drain(
            iter_real_model_stream_infer(
                self._req(),
                input_ids,
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=1,
                echo_prefix_ids=echo_prefix_ids,
            )
        )

    def _gen_ids(self, chunk) -> list[int]:
        infer = chunk.infer_response
        for i, out in enumerate(infer.outputs):
            if out.name == "generated_ids":
                raw = infer.raw_output_contents[i]
                shape = list(out.shape)
                declared_len = shape[-1] if shape else 0
                if declared_len <= 0:
                    return []
                return _unpack_int32_le(raw)
        return []

    async def test_echoes_prefix_when_input_tail_matches(self) -> None:
        chunks = await self._run(
            input_ids=[1, 2, 99, 100],
            echo_prefix_ids=[99, 100],
            upstream_ids=[[3, 4], [5, 6]],
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(self._gen_ids(chunks[0]), [99, 100, 3, 4])
        self.assertEqual(self._gen_ids(chunks[1]), [5, 6])

    async def test_no_echo_when_tail_mismatch(self) -> None:
        chunks = await self._run(
            input_ids=[1, 2, 3],
            echo_prefix_ids=[99, 100],
            upstream_ids=[[3, 4]],
        )
        self.assertEqual(self._gen_ids(chunks[0]), [3, 4])

    async def test_no_echo_when_prefix_empty(self) -> None:
        chunks = await self._run(
            input_ids=[1, 2, 99, 100],
            echo_prefix_ids=[],
            upstream_ids=[[3, 4]],
        )
        self.assertEqual(self._gen_ids(chunks[0]), [3, 4])

    async def test_echo_skips_empty_chunks_and_applies_to_first_non_empty(self) -> None:
        chunks = await self._run(
            input_ids=[99, 100],
            echo_prefix_ids=[99, 100],
            upstream_ids=[[], [3, 4], [5]],
        )
        self.assertEqual(self._gen_ids(chunks[0]), [])
        self.assertEqual(self._gen_ids(chunks[1]), [99, 100, 3, 4])
        self.assertEqual(self._gen_ids(chunks[2]), [5])


async def _areq_iter(requests):
    for r in requests:
        yield r


class DashScGrpcInferenceServicerTest(unittest.IsolatedAsyncioTestCase):
    def _valid_infer_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "srv-1"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 42))
        return req

    async def test_fake_mode_returns_incremented_ids(self) -> None:
        servicer = DashScGrpcInferenceServicer(backend_visitor=None)
        req = self._valid_infer_request()
        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )
        self.assertEqual(len(responses), 1)
        infer = responses[0].infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [142])

    async def test_missing_input_ids_error(self) -> None:
        servicer = DashScGrpcInferenceServicer(backend_visitor=None)
        bad = predict_v2_pb2.ModelInferRequest()
        bad.id = "x"
        bad.model_name = "m"
        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([bad]), MagicMock())
        )
        self.assertEqual(len(responses), 1)
        self.assertIn("input_ids", responses[0].error_message)

    async def test_real_mode_uses_enqueue(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )

        servicer = DashScGrpcInferenceServicer(backend_visitor=visitor)
        responses = await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([self._valid_infer_request()]), MagicMock()
            )
        )
        self.assertEqual(len(responses), 1)
        self.assertEqual(visitor.enqueue_called, 1)
        infer = responses[0].infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [9])

    async def test_real_mode_request_id_matches_generate_request_id(self) -> None:
        """Backend ``GenerateInput.request_id`` follows the same snowflake scheme as HTTP path."""
        from rtp_llm.frontend import request_id_generator as rig

        captured: list[int] = []

        class _CaptureVisitor:
            async def enqueue(self, gi):
                captured.append(gi.request_id)
                return _FakeAsyncStream([])

        servicer = DashScGrpcInferenceServicer(
            backend_visitor=_CaptureVisitor(),
            ip="10.0.0.1",
            port=12345,
            server_id="srv-xyz",
        )
        with patch.object(rig.time, "time", return_value=1_700_000_000.0):
            await _drain(
                servicer.ModelStreamInfer(
                    _areq_iter([self._valid_infer_request()]), MagicMock()
                )
            )
            expected = rig.generate_request_id("10.0.0.1", 12345, "srv-xyz", 1)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], expected)


if __name__ == "__main__":
    unittest.main()
