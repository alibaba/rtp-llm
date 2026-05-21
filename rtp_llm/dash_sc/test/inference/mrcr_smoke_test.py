"""MRCR regression smoke for DashSc DeepSeek v4 thinking protocol.

Fixtures cover three captured failure modes — labeled by ``failure_mode`` on
each fixture entry:

* ``trailing_think_in_content`` (run 14054750 qid 1/2/4, run 14060932 qid 118
  / 573) — answer text reaches ``content`` correctly but a structural
  ``</think>\\n\\n`` artifact at the tail leaks in.
* ``leading_thinking_in_content`` (run 14060932 qid 203) — phase-2 emits a
  reasoning preface ahead of ``</think>`` and the dashscope-side parser
  treats the preface as ``content`` (``extracted_prefix`` ends up totally
  unrelated to ``gt_prefix``).
* ``empty_content_answer_in_reasoning`` (run 14060932 qid 40 / 643) — model
  EOSes inside reasoning without ever emitting ``</think>``; ``content``
  comes back empty even though ``reasoning_content`` already contains the
  full answer.

Dashscope-serving has already split out ``reasoning_content`` for these; the
regression is in how DashSc transports the phase boundary. This smoke stays
cheap by validating the GenerateConfig sent to the backend for those request
shapes, instead of rerunning full MRCR.
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

import torch

from rtp_llm.dash_sc.codec import OtherParams, SamplingParams
from rtp_llm.dash_sc.inference.servicer import (
    build_think_runtime,
    iter_real_model_stream_infer,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "mrcr_deepseek_v4_think_leak_cases.json"
)


def _load_cases() -> list[dict]:
    with _FIXTURE.open() as f:
        return json.load(f)


def _add_input_ids(req: predict_v2_pb2.ModelInferRequest, input_ids: list[int]) -> None:
    inp = req.inputs.add()
    inp.name = "input_ids"
    inp.datatype = "INT32"
    inp.shape[:] = [len(input_ids)]
    req.raw_input_contents.append(struct.pack("<%di" % len(input_ids), *input_ids))


async def _drain(aiter):
    return [x async for x in aiter]


class _Dsv4Tokenizer:
    eos_token_id = 2
    vocab_size = 200000

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(
            {
                "<think>\n": [128821, 201],
                "</think>\n\n": [128822, 271],
                "<think>\n\n</think>\n\n": [128821, 271, 128822, 271],
            }[text]
        )


class _GenerateEnvCfg:
    think_mode = 1
    think_end_token_id = -1
    think_start_tag = "<think>\n"
    think_end_tag = "</think>\n\n"


class _FakeAsyncStream:
    def __init__(self, chunks=None):
        self._chunks = list(chunks or [])
        self._emitted = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._emitted >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._emitted]
        self._emitted += 1
        return item


class _CaptureVisitor:
    def __init__(self):
        self.generate_inputs = []

    async def enqueue(self, generate_input):
        self.generate_inputs.append(generate_input)
        return _FakeAsyncStream()


class _MultiStreamVisitor:
    def __init__(self, streams):
        self._streams = list(streams)
        self.generate_inputs = []

    async def enqueue(self, generate_input):
        self.generate_inputs.append(generate_input)
        return self._streams[len(self.generate_inputs) - 1]


def _go(ids: list[int], *, finished: bool, input_len: int) -> GenerateOutputs:
    return GenerateOutputs(
        generate_outputs=[
            GenerateOutput(
                output_ids=torch.tensor(ids, dtype=torch.int32),
                finished=finished,
                aux_info=AuxInfo(input_len=input_len, reuse_len=0),
            )
        ]
    )


def _gen_ids(chunk) -> list[int]:
    infer = chunk.infer_response
    for i, out in enumerate(infer.outputs):
        if out.name == "generated_ids":
            shape = list(out.shape)
            if not shape or shape[-1] <= 0:
                return []
            raw = infer.raw_output_contents[i]
            return list(struct.unpack("<%di" % (len(raw) // 4), raw))
    return []


def _think_token_num(chunk) -> int | None:
    param = chunk.infer_response.parameters.get("generate_think_token_num")
    return param.int64_param if param is not None else None


_TRAILING = "trailing_think_in_content"
_LEADING = "leading_thinking_in_content"
_EMPTY = "empty_content_answer_in_reasoning"


class DeepSeekV4MrcrSmokeTest(unittest.IsolatedAsyncioTestCase):
    def test_fixture_covers_stable_mrcr_think_leak_cases(self) -> None:
        cases = _load_cases()
        # 14054750 qid 1/2/4 + 14060932 qid 40/118/203/573/643 (5 new cases
        # spanning the three captured failure_modes documented at the top of
        # this module).
        self.assertEqual(
            [case["question_id"] for case in cases],
            [1, 2, 4, 40, 643, 118, 573, 203],
        )
        # Every fixture entry must classify its failure mode and use the
        # production DSV4 assistant turn ending in the single ``<think>``
        # token; without that suffix the request would not exercise the
        # phase-1 boundary the protocol assumes.
        for case in cases:
            self.assertIn(case["failure_mode"], {_TRAILING, _LEADING, _EMPTY})
            self.assertEqual(case["input_ids_tail"][-1], 128821)
            self.assertGreaterEqual(case["context_tokens"], 16000)

        trailing = [c for c in cases if c["failure_mode"] == _TRAILING]
        leading = [c for c in cases if c["failure_mode"] == _LEADING]
        empty = [c for c in cases if c["failure_mode"] == _EMPTY]
        # Coverage gates so future fixture trims do not silently drop a mode.
        self.assertGreaterEqual(len(trailing), 4)
        self.assertGreaterEqual(len(leading), 1)
        self.assertGreaterEqual(len(empty), 2)

        # ``trailing_think_in_content`` entries from 14054750 carry the legacy
        # ``rtp_content_around_think`` snippet showing the ``</think>``
        # leakage; entries from 14060932 carry ``rtp_content_prefix`` /
        # ``rtp_content_suffix`` instead. Either form must show the answer
        # arriving in content (the bug is in the trailing tag, not in the
        # answer body).
        for case in trailing:
            around = case.get("rtp_content_around_think") or ""
            prefix = case.get("rtp_content_prefix") or ""
            self.assertTrue(
                "</think>" in around or prefix,
                f"qid={case['question_id']} missing trailing-think evidence",
            )

        # ``leading_thinking_in_content`` is the worst failure: the
        # ``content`` channel starts with a reasoning preface (e.g. "We can't
        # complete the request because..."), so ``content_prefix`` MUST NOT
        # equal the random-prefix marker the answer is supposed to start with.
        for case in leading:
            self.assertNotEqual(
                case.get("rtp_content_prefix", "")[:10],
                case.get("random_prefix", "x"),
                f"qid={case['question_id']} leading mode but content starts with random_prefix",
            )

        # ``empty_content_answer_in_reasoning`` is the dashscope-visible
        # symptom of phase-2 not firing: ``content`` is empty even though
        # ``reasoning_content`` already starts with the random prefix.
        for case in empty:
            self.assertEqual(case.get("rtp_content_prefix", ""), "")
            reasoning = case.get("rtp_reasoning_prefix", "")
            self.assertTrue(
                reasoning.startswith(case.get("random_prefix", "")),
                f"qid={case['question_id']} reasoning_prefix does not start with random_prefix",
            )

    async def test_mrcr_requests_use_dashllm_style_dsv4_phase1(self) -> None:
        cases = _load_cases()
        visitor = _CaptureVisitor()
        tok = _Dsv4Tokenizer()
        env_cfg = _GenerateEnvCfg()
        think_runtime = build_think_runtime(tok, env_cfg, "deepseek_v4")

        for i, case in enumerate(cases, start=1):
            req = predict_v2_pb2.ModelInferRequest()
            req.id = case["request_id"]
            req.model_name = "deepseek-v4-flash"
            _add_input_ids(req, case["input_ids_tail"])

            await _drain(
                iter_real_model_stream_infer(
                    req,
                    case["input_ids_tail"],
                    SamplingParams(max_new_tokens=384000, top_p=1.0, temperature=1.0),
                    OtherParams(enable_thinking=True),
                    visitor,
                    rtp_llm_request_id=i,
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=think_runtime,
                )
            )

        self.assertEqual(len(visitor.generate_inputs), len(cases))
        for generate_input in visitor.generate_inputs:
            generate_config = generate_input.generate_config
            self.assertTrue(generate_config.in_think_mode)
            self.assertEqual(generate_config.end_think_token_ids, [128822, 271])
            self.assertNotIn([128822, 271], generate_config.stop_words_list)

    async def test_deepseek_v4_flash_phase2_stream_fixes_fixture_cases(self) -> None:
        cases = [c for c in _load_cases() if "smoke_phase1_output_ids" in c]
        self.assertEqual([c["question_id"] for c in cases], [40])

        tok = _Dsv4Tokenizer()
        env_cfg = _GenerateEnvCfg()
        think_runtime = build_think_runtime(tok, env_cfg, "deepseek_v4")
        echo_prefix_ids = tok.encode("<think>\n", add_special_tokens=False)

        for case in cases:
            req = predict_v2_pb2.ModelInferRequest()
            req.id = case["request_id"]
            req.model_name = "deepseek-v4-flash"
            input_ids = case["input_ids_tail"]
            _add_input_ids(req, input_ids)
            visitor = _MultiStreamVisitor(
                [
                    _FakeAsyncStream(
                        [
                            _go(
                                case["smoke_phase1_output_ids"],
                                finished=False,
                                input_len=len(input_ids),
                            )
                        ]
                    ),
                    _FakeAsyncStream(
                        [
                            _go(
                                case["smoke_phase2_output_ids"],
                                finished=True,
                                input_len=len(input_ids),
                            )
                        ]
                    ),
                ]
            )

            chunks = await _drain(
                iter_real_model_stream_infer(
                    req,
                    input_ids,
                    SamplingParams(max_new_tokens=384000, top_p=1.0, temperature=1.0),
                    OtherParams(enable_thinking=True),
                    visitor,
                    rtp_llm_request_id=1,
                    echo_prefix_ids=echo_prefix_ids,
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=think_runtime,
                    phase2_request_id_factory=lambda: 2,
                )
            )

            self.assertEqual(len(visitor.generate_inputs), 2)
            self.assertEqual(
                visitor.generate_inputs[1].token_ids.cpu().int().tolist(),
                input_ids[:-1] + [128821, 271, 128822, 271],
            )
            self.assertEqual(
                _think_token_num(chunks[1]),
                case["expected_generate_think_token_num"],
            )
            phase2_chunks = [c for c in chunks if c.infer_response.id.endswith("-2")]
            self.assertEqual(len(phase2_chunks), 1)
            self.assertEqual(_gen_ids(phase2_chunks[0]), case["expected_phase2_ids"])


if __name__ == "__main__":
    unittest.main()
