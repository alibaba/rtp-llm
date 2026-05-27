"""Unit tests for ``rtp_llm.dash_sc.inference.servicer`` (grpc.aio).

Covers:
- ``iter_real_model_stream_infer``: success, empty-stream fallback, exception propagation.
- ``DashScInferenceServicer.ModelStreamInfer``: fake mode, real mode,
  missing input_ids, request_id snowflake scheme alignment with HTTP
  ``generate_request_id``.
"""

from __future__ import annotations

import json
import struct
import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.generate_config import RoleAddr
from rtp_llm.dash_sc.codec import OtherParams, SamplingParams
from rtp_llm.dash_sc.inference.servicer import (
    DashScInferenceServicer,
    build_think_runtime,
    iter_real_model_stream_infer,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.ops import RoleType
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
        self.aclose_called = False

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

    async def aclose(self):
        self.aclose_called = True


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


def _dsv4_tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer(
        {
            "<think>\n": [128821, 198],
            "</think>\n\n": [128822, 271],
            "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
            "</think>": [128822],
        }
    )


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


def _finish_reason(chunk) -> int | None:
    infer = chunk.infer_response
    for i, out in enumerate(infer.outputs):
        if out.name == "finish_reason":
            return int(struct.unpack("<q", infer.raw_output_contents[i])[0])
    return None


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
        self.assertEqual(infer.parameters["prompt_token_num"].int64_param, 2)
        self.assertEqual(infer.parameters["prompt_cached_token_num"].int64_param, 0)

    async def test_reasoning_effort_override_reaches_generate_config(self) -> None:
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([3], dtype=torch.int32),
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
                OtherParams(reasoning_effort="xhigh"),
                visitor,
                rtp_llm_request_id=1,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            visitor.last_generate_input.generate_config.chat_template_kwargs,
            {"reasoning_effort": "xhigh"},
        )

    async def test_finished_at_max_new_tokens_reports_length_repro_p1(self) -> None:
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([7, 8, 9], dtype=torch.int32),
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
                SamplingParams(max_new_tokens=3),
                OtherParams(),
                visitor,
                rtp_llm_request_id=1,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(_gen_ids(chunks[0]), [7, 8, 9])
        self.assertEqual(_finish_reason(chunks[0]), 1)

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

    async def test_no_thinking_budget_zero_sets_sampler_mask_config_without_filtering(
        self,
    ) -> None:
        req = self._minimal_request()
        _add_input_tensor(
            req,
            "max_new_think_tokens",
            "INT32",
            [1],
            struct.pack("<i", 0),
        )
        out = GenerateOutput(
            output_ids=torch.tensor([10, 128822, 271], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )

        chunks = await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        gc = visitor.last_generate_input.generate_config
        self.assertFalse(gc.in_think_mode)
        self.assertEqual(gc.max_thinking_tokens, 0)
        self.assertEqual(gc.begin_think_token_ids, [128821, 198])
        self.assertEqual(gc.end_think_token_ids, [128822, 271])
        self.assertEqual(_gen_ids(chunks[0]), [10, 128822, 271])

    async def test_max_think_length_wins_final_config_over_max_new_think_tokens(
        self,
    ) -> None:
        req = self._minimal_request()
        _add_input_tensor(
            req,
            "max_new_think_tokens",
            "INT32",
            [1],
            struct.pack("<i", 0),
        )
        _add_input_tensor(
            req,
            "max_think_length",
            "INT32",
            [1],
            struct.pack("<i", -1),
        )
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        context = MagicMock()
        context.invocation_metadata.return_value = ()

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), context))

        self.assertEqual(visitor.enqueue_called, 1)
        gc = visitor.last_generate_input.generate_config
        self.assertTrue(gc.in_think_mode)
        self.assertEqual(gc.max_thinking_tokens, 2_147_483_647)
        self.assertEqual(gc.end_think_token_ids, [128822, 271])

    async def test_budget_zero_disables_thinking_even_if_add_thinking_params_fails(
        self,
    ) -> None:
        """Request-level zero budget must still produce a full think mask config."""
        req = self._minimal_request()
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        with patch(
            "rtp_llm.config.generate_config.GenerateConfig.add_thinking_params",
            side_effect=RuntimeError("boom"),
        ):
            await _drain(
                iter_real_model_stream_infer(
                    req,
                    [1, 2],
                    SamplingParams(max_new_think_tokens=0),
                    OtherParams(),
                    visitor,
                    rtp_llm_request_id=1,
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                )
            )

        gc = visitor.last_generate_input.generate_config
        self.assertFalse(gc.in_think_mode)
        self.assertEqual(gc.max_thinking_tokens, 0)
        self.assertEqual(gc.begin_think_token_ids, [128821, 198])
        self.assertEqual(gc.end_think_token_ids, [128822, 271])

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

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=1,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "qwen2"),
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
                    output_ids=torch.tensor([20, 21, 22], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )

        class _RoutingVisitor(_MultiStreamVisitor):
            async def enqueue(self, generate_input):
                if self.enqueue_called == 0:
                    generate_input.generate_config.role_addrs = [
                        RoleAddr(
                            role=RoleType.PREFILL,
                            ip="127.0.0.1",
                            http_port=8080,
                            grpc_port=8081,
                        )
                    ]
                return await super().enqueue(generate_input)

        visitor = _RoutingVisitor(
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

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(
                    response_format=json.dumps({"type": "json_object"}),
                ),
                OtherParams(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertEqual(visitor.generate_inputs[0].request_id, 100)
        self.assertEqual(visitor.generate_inputs[1].request_id, 200)
        self.assertTrue(visitor.generate_inputs[0].generate_config.in_think_mode)
        self.assertEqual(
            visitor.generate_inputs[0].generate_config.begin_think_token_ids,
            [128821, 198],
        )
        self.assertEqual(
            visitor.generate_inputs[0].generate_config.end_think_token_ids,
            [128822, 271],
        )
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10, 11])
        self.assertEqual(_gen_ids(chunks[1]), [128822, 271])
        self.assertEqual(_gen_ids(chunks[2]), [20, 21, 22])
        self.assertEqual(chunks[2].infer_response.id, "trace-real-2")
        # phase-2 trace_id mirrors phase-1 (no -2 suffix) so dashscope log search
        # finds both halves under a single trace.
        self.assertEqual(
            visitor.generate_inputs[1].generate_config.trace_id, "trace-real"
        )
        phase2_input_ids = visitor.generate_inputs[1].token_ids.cpu().int().tolist()
        self.assertEqual(phase2_input_ids, [7, 8, 128821, 271, 128822, 271])
        self.assertEqual(
            json.loads(visitor.generate_inputs[0].generate_config.response_format),
            {"type": "json_object"},
        )
        self.assertEqual(
            json.loads(visitor.generate_inputs[1].generate_config.response_format),
            {"type": "json_object"},
        )
        self.assertFalse(visitor.generate_inputs[1].generate_config.in_think_mode)
        self.assertEqual(
            len(visitor.generate_inputs[0].generate_config.role_addrs), 1
        )
        self.assertEqual(visitor.generate_inputs[1].generate_config.role_addrs, [])
        self.assertNotIn(10, phase2_input_ids)
        self.assertNotIn(11, phase2_input_ids)
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            3,
        )

    async def test_phase2_finished_at_max_new_tokens_reports_length(self) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20, 21], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [_FakeAsyncStream([phase1]), _FakeAsyncStream([phase2])]
        )
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(
                    max_new_tokens=2,
                    max_new_tokens_from_completion_alias=True,
                ),
                OtherParams(max_new_think_tokens=10),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        phase2_chunks = [c for c in chunks if c.infer_response.id.endswith("-2")]
        self.assertEqual(visitor.generate_inputs[0].generate_config.max_new_tokens, 2)
        self.assertEqual(visitor.generate_inputs[1].generate_config.max_new_tokens, 2)
        self.assertEqual(len(phase2_chunks), 1)
        self.assertEqual(_gen_ids(phase2_chunks[0]), [20, 21])
        self.assertEqual(_finish_reason(phase2_chunks[0]), 1)

    async def test_phase2_completion_alias_respects_max_tokens_total_cap(
        self,
    ) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor(
                        list(range(10, 20)) + [1], dtype=torch.int32
                    ),
                    finished=False,
                    aux_info=AuxInfo(input_len=2, reuse_len=0),
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
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(
                    max_new_tokens=100,
                    max_new_tokens_from_completion_alias=True,
                    max_total_tokens=105,
                ),
                OtherParams(max_new_think_tokens=10),
                visitor,
                rtp_llm_request_id=100,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.generate_inputs[0].generate_config.max_new_tokens, 100)
        self.assertEqual(visitor.generate_inputs[1].generate_config.max_new_tokens, 95)
        self.assertEqual(
            chunks[1].infer_response.parameters["generate_think_token_num"].int64_param,
            10,
        )

    async def test_phase2_completion_alias_budget_zero_does_not_enqueue_phase2(
        self,
    ) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 12, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=2, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor([_FakeAsyncStream([phase1])])
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(
                    max_new_tokens=3,
                    max_new_tokens_from_completion_alias=True,
                    max_total_tokens=3,
                ),
                OtherParams(max_new_think_tokens=10),
                visitor,
                rtp_llm_request_id=100,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertFalse(any(c.error_message for c in chunks))
        self.assertEqual(_gen_ids(chunks[-1]), [128822, 271])
        self.assertEqual(_finish_reason(chunks[-1]), 1)

    async def test_token1_phase2_closes_phase1_stream_before_phase2_enqueue(
        self,
    ) -> None:
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1], dtype=torch.int32),
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
        phase1_stream = _FakeAsyncStream([phase1])
        visitor = _MultiStreamVisitor([phase1_stream, _FakeAsyncStream([phase2])])
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()

        await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertTrue(phase1_stream.aclose_called)

    async def test_request_disable_thinking_prevents_token1_phase2(self) -> None:
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
        visitor = _FakeVisitor(_FakeAsyncStream([phase1]))
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(enable_thinking=False, max_new_think_tokens=0),
                visitor,
                rtp_llm_request_id=100,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(_gen_ids(chunks[0]), [10, 11, 1, 99])
        self.assertNotEqual(chunks[0].infer_response.id, "trace-real-2")
        cfg = visitor.generate_inputs[0].generate_config
        self.assertFalse(cfg.in_think_mode)
        self.assertEqual(cfg.max_thinking_tokens, 0)
        self.assertEqual(cfg.begin_think_token_ids, [128821, 198])
        self.assertEqual(cfg.end_think_token_ids, [128822, 271])
        self.assertNotIn("max_new_think_tokens", chunks[0].infer_response.parameters)
        self.assertNotIn(
            "generate_think_token_num", chunks[0].infer_response.parameters
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

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
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

    async def test_terminate_token_id_disabled_keeps_token_in_stream(self) -> None:
        """``terminate_token_id=None`` disables the in-stream "stop thinking"
        branch: token id 1 is emitted as a regular content token, with no
        truncation and no phase-2 enqueue. (No ``</think>`` in the stream so the
        regular close-driven phase-2 path also stays dormant.)"""
        req = self._minimal_request()
        out = GenerateOutput(
            output_ids=torch.tensor([10, 1, 11], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=2, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                visitor,
                rtp_llm_request_id=1,
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(
                    tok, env_cfg, "deepseek_v4", terminate_token_id=None
                ),
            )
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(_gen_ids(chunks[0]), [10, 1, 11])

    async def test_terminate_token_id_configurable_value(self) -> None:
        """A non-default ``terminate_token_id`` (here 42) drives the same
        truncation + phase-2 prompt rewrite that token id 1 does by default."""
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 42, 99], dtype=torch.int32),
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

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(
                    tok, env_cfg, "deepseek_v4", terminate_token_id=42
                ),
                phase2_request_id_factory=lambda: 200,
            )
        )

        self.assertEqual(visitor.enqueue_called, 2)
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10, 11])
        self.assertEqual(_gen_ids(chunks[1]), [128822, 271])
        self.assertEqual(_gen_ids(chunks[2]), [20])
        self.assertNotIn(42, visitor.generate_inputs[1].token_ids.cpu().int().tolist())

    async def test_natural_finish_without_close_does_not_trigger_phase2(self) -> None:
        """Phase-1 finishes naturally without ``</think>`` or terminate_token —
        the model dumped the answer entirely into reasoning. After the
        DashLLM-alignment change, phase-2 is NO LONGER fired in this case;
        only the terminate-token-id (DSV4 token 1) abort path enters phase-2.
        The whole phase-1 output is streamed as reasoning content.
        """
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=3, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor([_FakeAsyncStream([phase1])])
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        # Single phase-1 enqueue; no second request.
        self.assertEqual(visitor.enqueue_called, 1)
        # Whole phase-1 output streamed as reasoning, no phase-2 chunk follows.
        self.assertEqual(_gen_ids(chunks[0]), [128821, 10, 11, 12])
        for chunk in chunks:
            self.assertNotEqual(chunk.infer_response.id, "trace-real-2")

    async def test_dsv4_natural_close_does_not_trigger_phase2(self) -> None:
        """DSV4 phase-1 with a normal ``</think>`` close emits content in the
        same stream — phase-2 MUST NOT fire. Mirrors DashLLM ``_think.py``
        line 622-628: natural close only updates ``generate_think_token_num``.
        """
        req = self._minimal_request()
        # Stream: think tokens, ``</think>\n\n`` (128822, 271), then answer.
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor(
                        [10, 11, 128822, 271, 200, 201],
                        dtype=torch.int32,
                    ),
                    finished=True,
                    aux_info=AuxInfo(input_len=3, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor([_FakeAsyncStream([phase1])])
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        # Exactly one enqueue: phase-1 only, no phase-2 follow-up.
        self.assertEqual(visitor.enqueue_called, 1)
        for chunk in chunks:
            # Phase-2 chunks would carry the ``-2`` suffix on infer_response.id.
            self.assertNotEqual(chunk.infer_response.id, "trace-real-2")

    async def test_dsv4_token1_phase2_reports_metric_once(self) -> None:
        """Phase-2 entry MUST fan out exactly one increment of the DSV4 phase-2
        metric — guarded by ``phase2_triggered`` so the rate matches
        "requests with a think-abort", not "abort tokens seen"."""
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 11, 1, 99], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=3, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([20, 21], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=8, reuse_len=0),
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
        env_cfg = _GenerateEnvCfg()

        with patch("rtp_llm.dash_sc.inference.servicer.kmonitor.report") as mock_report:
            await _drain(
                iter_real_model_stream_infer(
                    req,
                    [7, 8, 128821],
                    SamplingParams(),
                    OtherParams(enable_thinking=True),
                    visitor,
                    rtp_llm_request_id=100,
                    echo_prefix_ids=[128821, 198],
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                    phase2_request_id_factory=lambda: 200,
                )
            )

        from rtp_llm.metrics import AccMetrics

        phase2_calls = [
            call_args
            for call_args in mock_report.call_args_list
            if call_args.args
            and call_args.args[0] is AccMetrics.DASH_SC_DSV4_PHASE2_QPS_METRIC
        ]
        self.assertEqual(len(phase2_calls), 1)
        _metric, value, tags = phase2_calls[0].args
        self.assertEqual(value, 1)
        self.assertEqual(tags["protocol"], "dash_sc_grpc")

    async def test_phase2_strips_leading_thinking_then_close(self) -> None:
        """Phase-2 model occasionally emits accidental thinking followed by
        ``</think>`` before the real answer. The leading reasoning + close
        sequence must be stripped so only post-close tokens reach the client."""
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
        phase2_a = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([55, 56], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2_b = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([128822, 271, 20, 21], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        visitor = _MultiStreamVisitor(
            [
                _FakeAsyncStream([phase1]),
                _FakeAsyncStream([phase2_a, phase2_b]),
            ]
        )
        tok = _FakeTokenizer(
            {
                "<think>\n": [128821, 198],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
                "</think>": [128822],
            }
        )

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        # Phase-1 emits two chunks (truncated content then synthesised eos).
        # Phase-2 sees [55, 56] (accidental thinking, buffered then dropped),
        # then [128822, 271, 20, 21] (close + tail content). Client only sees
        # [20, 21] from phase-2.
        phase2_chunks = [c for c in chunks if c.infer_response.id.endswith("-2")]
        self.assertEqual(len(phase2_chunks), 1)
        self.assertEqual(_gen_ids(phase2_chunks[0]), [20, 21])

    async def test_phase2_strips_trailing_eos_artifact(self) -> None:
        """Phase-2 ends with a structural ``</think>\\n\\n`` closing-tag
        artifact mirroring the empty-think prompt body. That trailing
        sequence must not leak into ``content``."""
        req = self._minimal_request()
        phase1 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([10, 1], dtype=torch.int32),
                    finished=False,
                    aux_info=AuxInfo(input_len=4, reuse_len=0),
                )
            ]
        )
        phase2 = GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor(
                        [30, 31, 32, 128822, 271], dtype=torch.int32
                    ),
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

        env_cfg = _GenerateEnvCfg()
        chunks = await _drain(
            iter_real_model_stream_infer(
                req,
                [7, 8, 128821],
                SamplingParams(),
                OtherParams(enable_thinking=True),
                visitor,
                rtp_llm_request_id=100,
                echo_prefix_ids=[128821, 198],
                tokenizer=tok,
                generate_env_config=env_cfg,
                think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                phase2_request_id_factory=lambda: 200,
            )
        )

        phase2_chunks = [c for c in chunks if c.infer_response.id.endswith("-2")]
        self.assertEqual(len(phase2_chunks), 1)
        # Trailing [128822, 271] is stripped; only the real answer ids survive.
        self.assertEqual(_gen_ids(phase2_chunks[0]), [30, 31, 32])


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


class _FakeGrpcContext:
    def __init__(self, metadata=()):
        self._metadata = tuple(metadata)
        self.initial_metadata = []

    def invocation_metadata(self):
        return self._metadata

    async def send_initial_metadata(self, metadata):
        self.initial_metadata.append(tuple(metadata))


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

    async def test_timeout_request_sets_dashscope_partial_response_metadata(
        self,
    ) -> None:
        servicer = DashScInferenceServicer(backend_visitor=None)
        req = self._valid_infer_request()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-dashscope-inner-timeout": 1}
        )
        context = _FakeGrpcContext()

        responses = await _drain(servicer.ModelStreamInfer(_areq_iter([req]), context))

        self.assertEqual(len(responses), 1)
        self.assertIn(
            (("x-dashscope-partialresponse", "true"),),
            context.initial_metadata,
        )

    async def test_max_new_tokens_negative_rejected_before_enqueue_repro_p3(
        self,
    ) -> None:
        """max_new_tokens=-1 must return 400 via __messages__ (DashLLM protocol),
        not 500 via error_message."""
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        _add_input_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", -1))

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 0)
        self.assertEqual(len(responses), 1)
        self.assertFalse(responses[0].error_message)
        payload = json.loads(
            responses[0].infer_response.parameters["__messages__"].string_param
        )
        self.assertEqual(payload["header"]["status_code"], 400)
        self.assertEqual(payload["header"]["status_name"], "InvalidParameter")
        self.assertIn("max_new_tokens", payload["header"]["status_message"])

    async def test_openai_compat_max_new_tokens_negative_uses_default(
        self,
    ) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        req = self._valid_infer_request()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-envoy-original-path": "/compatible-mode/v1/chat/completions"}
        )
        _add_input_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", -1))

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(
            visitor.last_generate_input.generate_config.max_new_tokens,
            32000,
        )

    async def test_max_completion_tokens_non_positive_uses_default_repro(
        self,
    ) -> None:
        """Compat alias values <= 0 should not reach backend as max_new_tokens."""
        for value in (-1, 0):
            with self.subTest(value=value):
                out = GenerateOutput(
                    output_ids=torch.tensor([9], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=1, reuse_len=0),
                )
                visitor = _FakeVisitor(
                    _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
                )
                servicer = DashScInferenceServicer(backend_visitor=visitor)
                req = self._valid_infer_request()
                req.parameters["max_completion_tokens"].int64_param = value

                responses = await _drain(
                    servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
                )

                self.assertEqual(visitor.enqueue_called, 1)
                self.assertEqual(len(responses), 1)
                self.assertEqual(
                    visitor.last_generate_input.generate_config.max_new_tokens,
                    32000,
                )

    async def test_max_completion_tokens_non_positive_blocks_legacy_aliases(
        self,
    ) -> None:
        for value in (-1, 0):
            with self.subTest(value=value):
                out = GenerateOutput(
                    output_ids=torch.tensor([9], dtype=torch.int32),
                    finished=True,
                    aux_info=AuxInfo(input_len=1, reuse_len=0),
                )
                visitor = _FakeVisitor(
                    _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
                )
                servicer = DashScInferenceServicer(backend_visitor=visitor)
                req = self._valid_infer_request()
                _add_input_tensor(
                    req,
                    "max_completion_tokens",
                    "INT32",
                    [1],
                    struct.pack("<i", value),
                )
                _add_input_tensor(
                    req,
                    "max_new_tokens",
                    "INT32",
                    [1],
                    struct.pack("<i", -1),
                )

                responses = await _drain(
                    servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
                )

                self.assertEqual(visitor.enqueue_called, 1)
                self.assertEqual(len(responses), 1)
                self.assertEqual(
                    visitor.last_generate_input.generate_config.max_new_tokens,
                    32000,
                )

    async def test_dash_generation_without_enable_thinking_disables_env_thinking(
        self,
    ) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        visitor = _FakeVisitor(
            _FakeAsyncStream([GenerateOutputs(generate_outputs=[out])])
        )
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["max_new_tokens"].int64_param = 3
        req.parameters["result_format"].string_param = "message"
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "x-dashscope-inner-timeout": 1800,
                "user_id": "u1",
            }
        )

        responses = await _drain(
            servicer.ModelStreamInfer(_areq_iter([req]), MagicMock())
        )

        self.assertEqual(visitor.enqueue_called, 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(_gen_ids(responses[0]), [9])
        generate_config = visitor.last_generate_input.generate_config
        self.assertEqual(generate_config.max_new_tokens, 3)
        self.assertFalse(generate_config.in_think_mode)
        self.assertEqual(generate_config.max_thinking_tokens, 0)

    async def test_dash_generation_enable_thinking_true_without_budget_keeps_thinking(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["enable_thinking"].bool_param = True

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        generate_config = visitor.last_generate_input.generate_config
        self.assertTrue(generate_config.in_think_mode)
        self.assertEqual(generate_config.max_thinking_tokens, 32000)

    async def test_dash_generation_json_object_with_enable_thinking_keeps_both_constraints(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["enable_thinking"].bool_param = True
        req.parameters["response_format"].string_param = json.dumps(
            {"type": "json_object"}
        )

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        generate_config = visitor.last_generate_input.generate_config
        self.assertTrue(generate_config.in_think_mode)
        self.assertEqual(generate_config.end_think_token_ids, [128822, 271])
        self.assertEqual(
            json.loads(generate_config.response_format), {"type": "json_object"}
        )

    async def test_dash_generation_budget_aliases_without_enable_thinking_keep_thinking(
        self,
    ) -> None:
        for param_name in ("thinking_budget", "max_new_think_tokens"):
            with self.subTest(param_name=param_name):
                visitor = _FakeVisitor(_FakeAsyncStream([]))
                tok = _dsv4_tokenizer()
                env_cfg = _GenerateEnvCfg()
                servicer = DashScInferenceServicer(
                    backend_visitor=visitor,
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
                )
                req = self._valid_infer_request()
                req.parameters[param_name].int64_param = 10

                await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

                self.assertEqual(visitor.enqueue_called, 1)
                generate_config = visitor.last_generate_input.generate_config
                self.assertTrue(generate_config.in_think_mode)
                self.assertEqual(generate_config.max_thinking_tokens, 10)

    async def test_max_completion_tokens_thinking_budget_keeps_backend_limit_repro(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        tok = _dsv4_tokenizer()
        env_cfg = _GenerateEnvCfg()
        servicer = DashScInferenceServicer(
            backend_visitor=visitor,
            tokenizer=tok,
            generate_env_config=env_cfg,
            think_runtime=build_think_runtime(tok, env_cfg, "deepseek_v4"),
        )
        req = self._valid_infer_request()
        req.parameters["max_new_tokens"].int64_param = 200
        req.parameters["max_completion_tokens"].int64_param = 100
        req.parameters["enable_thinking"].bool_param = True
        req.parameters["thinking_budget"].int64_param = 10

        await _drain(servicer.ModelStreamInfer(_areq_iter([req]), MagicMock()))

        self.assertEqual(visitor.enqueue_called, 1)
        generate_config = visitor.last_generate_input.generate_config
        self.assertEqual(generate_config.max_new_tokens, 100)
        self.assertTrue(generate_config.in_think_mode)
        self.assertEqual(generate_config.max_thinking_tokens, 10)

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

    async def test_real_mode_uses_ds_header_attributes_for_backend_controls(
        self,
    ) -> None:
        visitor = _FakeVisitor(_FakeAsyncStream([]))
        servicer = DashScInferenceServicer(backend_visitor=visitor)
        context = MagicMock()
        context.invocation_metadata.return_value = ()
        request = self._valid_infer_request()
        request.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "x-dashscope-inner-timeout": 1800,
                "x-ds-request-priority": "10",
                "user_id": "u1",
                "x-dashscope-apikeyid": "ak1",
            }
        )
        request.parameters["enable_thinking"].bool_param = False
        request.parameters["thinking_budget"].int64_param = 100

        await _drain(servicer.ModelStreamInfer(_areq_iter([request]), context))

        self.assertIsNotNone(visitor.last_generate_input)
        generate_config = visitor.last_generate_input.generate_config
        self.assertFalse(generate_config.in_think_mode)
        self.assertEqual(generate_config.max_thinking_tokens, 0)
        self.assertEqual(generate_config.end_think_token_ids, [])
        self.assertEqual(generate_config.timeout_ms, 1_795_000)
        self.assertEqual(generate_config.ttft_timeout_ms, 1_795_000)
        self.assertEqual(generate_config.traffic_reject_priority, 10)
        self.assertEqual(
            visitor.last_generate_input.headers,
            {"user_id": "u1", "x-dashscope-apikeyid": "ak1"},
        )


if __name__ == "__main__":
    unittest.main()
