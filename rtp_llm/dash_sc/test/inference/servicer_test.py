"""Unit tests for ``rtp_llm.dash_sc.inference.servicer`` (grpc.aio).

Covers:
- ``iter_real_model_stream_infer``: success, empty-stream fallback, exception propagation.
- ``DashScInferenceServicer.ModelStreamInfer``: fake mode, real mode,
  missing input_ids, request_id snowflake scheme alignment with HTTP
  ``generate_request_id``.
"""

from __future__ import annotations

import struct
import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.dash_sc.codec import OtherParams, SamplingParams
from rtp_llm.dash_sc.inference.servicer import (
    DashScInferenceServicer,
    iter_real_model_stream_infer,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
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
        self.last_generate_input = None
        self.generate_inputs = []

    async def enqueue(self, _generate_input):
        self.enqueue_called += 1
        self.last_generate_input = _generate_input
        self.generate_inputs.append(_generate_input)
        return self._stream


class _MultiStreamVisitor:
    """Async ``enqueue`` that returns one stream per call."""

    def __init__(self, streams):
        self._streams = list(streams)
        self.enqueue_called = 0
        self.last_generate_input = None
        self.generate_inputs = []

    async def enqueue(self, generate_input):
        self.enqueue_called += 1
        self.last_generate_input = generate_input
        self.generate_inputs.append(generate_input)
        return self._streams[self.enqueue_called - 1]


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 200000

    def __init__(self, mapping: dict[str, list[int]]):
        self._mapping = mapping
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append((text, add_special_tokens))
        return list(self._mapping[text])


class _GenerateEnvCfg:
    think_mode = 1
    think_end_token_id = -1
    think_start_tag = "<think>\n"
    think_end_tag = "</think>\n\n"


async def _drain(aiter):
    return [x async for x in aiter]


def _gen_ids(chunk) -> list[int]:
    infer = chunk.infer_response
    for i, out in enumerate(infer.outputs):
        if out.name == "generated_ids":
            shape = list(out.shape)
            declared_len = shape[-1] if shape else 0
            if declared_len <= 0:
                return []
            return _unpack_int32_le(infer.raw_output_contents[i])
    return []


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

    async def test_deepseek_v4_multi_think_uses_first_close_only(self) -> None:
        req = self._minimal_request()
        chunks_proto = []
        for ids, finished in (
            ([10, 128822, 11], False),
            ([12, 128822, 13], True),
        ):
            out = GenerateOutput(
                output_ids=torch.tensor(ids, dtype=torch.int32),
                finished=finished,
                aux_info=AuxInfo(input_len=2, reuse_len=0),
            )
            chunks_proto.append(GenerateOutputs(generate_outputs=[out]))
        visitor = _FakeVisitor(_FakeAsyncStream(chunks_proto))
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=1,
                tokenizer=tok,
                generate_env_config=_GenerateEnvCfg(),
                model_type="qwen2",
            )
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(_gen_ids(chunks[0]), [10, 128822, 11])
        self.assertEqual(_gen_ids(chunks[1]), [12, 128822, 13])
        self.assertEqual(
            chunks[0].infer_response.parameters["generate_think_token_num"].int64_param,
            2,
        )
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            2,
        )

    async def test_deepseek_v4_token1_forces_empty_think_phase2_prompt(self) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 1, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20, 128822, 21], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=_GenerateEnvCfg(),
                model_type="deepseek_v4",
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertEqual(visitor.generate_inputs[0].request_id, 100)
        self.assertEqual(visitor.generate_inputs[1].request_id, 200)
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10, 11])
        self.assertEqual(_gen_ids(chunks[1]), [128822, 271])
        self.assertEqual(_gen_ids(chunks[2]), [20, 128822, 21])
        self.assertEqual(chunks[2].infer_response.id, "trace-real-2")
        phase2_input_ids = visitor.generate_inputs[1].token_ids.cpu().int().tolist()
        self.assertEqual(phase2_input_ids, [7, 8, 128821, 271, 128822, 271])
        self.assertFalse(visitor.generate_inputs[1].generate_config.in_think_mode)
        self.assertNotIn(10, phase2_input_ids)
        self.assertNotIn(11, phase2_input_ids)
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            3,
        )

    async def test_deepseek_v4_token1_before_close_wins_within_chunk(self) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1, 128822, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=_GenerateEnvCfg(),
                model_type="deepseek_v4",
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10])
        self.assertEqual(_gen_ids(chunks[1]), [128822, 271])
        self.assertEqual(_gen_ids(chunks[2]), [20])
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            2,
        )


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


class IterRealModelStreamInferStopWordsTest(unittest.IsolatedAsyncioTestCase):
    """``extra_stop_word_ids`` injection (renderer + env extras the dash-sc path
    misses because pre-tokenized input bypasses the OpenAI endpoint)."""

    def _req(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "stop-trace"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 42))
        return req

    async def _captured_stop_words(self, *, extra_stop_word_ids):
        captured: list = []

        class _CaptureVisitor:
            async def enqueue(self, gi):
                captured.append(gi)
                return _FakeAsyncStream([])

        await _drain(
            iter_real_model_stream_infer(
                self._req(),
                [42],
                SamplingParams(),
                OtherParams(),
                _CaptureVisitor(),
                rtp_llm_request_id=1,
                extra_stop_word_ids=extra_stop_word_ids,
            )
        )
        self.assertEqual(len(captured), 1)
        return list(captured[0].generate_config.stop_words_list or [])

    async def test_extra_stop_word_ids_appended(self) -> None:
        sw = await self._captured_stop_words(extra_stop_word_ids=[[154827], [154829]])
        self.assertIn([154827], sw)
        self.assertIn([154829], sw)

    async def test_none_leaves_stop_words_unchanged(self) -> None:
        sw = await self._captured_stop_words(extra_stop_word_ids=None)
        self.assertNotIn([154827], sw)
        self.assertNotIn([154829], sw)

    async def test_dedup_against_request_stop_words(self) -> None:
        """When the request carries a stop_word that's also in extras, the
        merged list keeps a single entry. (Extras themselves are pre-deduped
        at startup by ``_derive_stop_word_ids_list``, so the hot path only
        dedups extras-vs-request, not extras-vs-extras.)"""
        captured: list = []

        class _CaptureVisitor:
            async def enqueue(self, gi):
                captured.append(gi)
                return _FakeAsyncStream([])

        await _drain(
            iter_real_model_stream_infer(
                self._req(),
                [42],
                SamplingParams(stop_words_list=((154827,),)),
                OtherParams(),
                _CaptureVisitor(),
                rtp_llm_request_id=1,
                extra_stop_word_ids=[[154827], [154829]],
            )
        )
        sw = list(captured[0].generate_config.stop_words_list or [])
        self.assertEqual(sw.count([154827]), 1)
        self.assertIn([154829], sw)


async def _areq_iter(requests):
    for r in requests:
        yield r


class DashScInferenceServicerTest(unittest.IsolatedAsyncioTestCase):
    def _valid_infer_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "srv-1"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 42))
        return req

    async def test_fake_mode_returns_incremented_ids(self) -> None:
        servicer = DashScInferenceServicer(backend_visitor=None)
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
        servicer = DashScInferenceServicer(backend_visitor=None)
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

        servicer = DashScInferenceServicer(backend_visitor=visitor)
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

        servicer = DashScInferenceServicer(
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

    async def test_real_mode_passes_invocation_metadata_to_generate_input(self) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        context = MagicMock()
        context.invocation_metadata.return_value = (
            ("User_ID", "u2"),
            ("x-dashscope-apikeyid", "ak2"),
            ("authorization", "secret"),
        )

        await _drain(
            servicer.ModelStreamInfer(
                _areq_iter([self._valid_infer_request()]), context
            )
        )

        self.assertIsNotNone(visitor.last_generate_input)
        self.assertEqual(
            visitor.last_generate_input.headers,
            {"user_id": "u2", "x-dashscope-apikeyid": "ak2"},
        )


if __name__ == "__main__":
    unittest.main()
