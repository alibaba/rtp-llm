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

from rtp_llm.dash_sc.codec import OtherParams, SamplingParams
from rtp_llm.dash_sc.inference.servicer import iter_real_model_stream_infer
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.think import THINK_MODE_AUTO, DashScThinkConfig

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "mrcr_deepseek_v4_think_leak_cases.json"
)

_DEEPSEEK_V4_THINK_CONFIG = DashScThinkConfig(
    enabled=True,
    mode=THINK_MODE_AUTO,
    bos_tokens=(128821, 201),  # encode("<think>\n")
    end_think_token_ids=(128822, 271),  # encode("</think>\n\n")
    eos_tokens=(201, 128822, 271),  # encode("\n</think>\n\n")
    empty_tokens=(128821, 271, 128822, 271),  # encode("<think>\n\n</think>\n\n")
    is_deepseek_v4=True,
    dsv4_abort_token_id=1,
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


class _EmptyAsyncStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _CaptureVisitor:
    def __init__(self):
        self.generate_inputs = []

    async def enqueue(self, generate_input):
        self.generate_inputs.append(generate_input)
        return _EmptyAsyncStream()


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
                    think_config=_DEEPSEEK_V4_THINK_CONFIG,
                )
            )

        self.assertEqual(len(visitor.generate_inputs), len(cases))
        for generate_input in visitor.generate_inputs:
            generate_config = generate_input.generate_config
            self.assertTrue(generate_config.in_think_mode)
            self.assertEqual(generate_config.end_think_token_ids, [201, 128822, 271])
            self.assertEqual(generate_config.abort_think_token_ids, [1])
            self.assertIn([1], generate_config.stop_words_list)
            self.assertNotIn([128822, 271], generate_config.stop_words_list)
            self.assertNotIn([201, 128822, 271], generate_config.stop_words_list)


if __name__ == "__main__":
    unittest.main()
