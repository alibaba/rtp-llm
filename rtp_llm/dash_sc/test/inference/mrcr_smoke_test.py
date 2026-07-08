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
_THINK_BEGIN_IDS = [128821, 201]
_THINK_END_IDS = [128822, 271]
_THINK_BEGIN_TOKEN_ID = _THINK_BEGIN_IDS[0]
_THINK_END_TOKEN_ID = _THINK_END_IDS[0]


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
                "</think>": [128822],
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


def _go(
    ids: list[int],
    *,
    finished: bool,
    input_len: int,
    logits: torch.Tensor | None = None,
    all_probs: torch.Tensor | None = None,
) -> GenerateOutputs:
    return GenerateOutputs(
        generate_outputs=[
            GenerateOutput(
                output_ids=torch.tensor(ids, dtype=torch.int32),
                finished=finished,
                aux_info=AuxInfo(input_len=input_len, reuse_len=0),
                logits=logits,
                all_probs=all_probs,
            )
        ]
    )


def _go_with_boundary_scores(
    ids: list[int],
    *,
    finished: bool,
    input_len: int,
) -> GenerateOutputs:
    vocab_size = _Dsv4Tokenizer.vocab_size
    logits = torch.full((len(ids), vocab_size), -20.0, dtype=torch.float32)
    all_probs = torch.zeros((len(ids), vocab_size), dtype=torch.float32)
    for step, token_id in enumerate(ids):
        logits[step, token_id] = 12.0 + step
        all_probs[step, token_id] = 0.9
        logits[step, _THINK_BEGIN_TOKEN_ID] = float("-inf")
        logits[step, _THINK_END_TOKEN_ID] = float("-inf")
        all_probs[step, _THINK_BEGIN_TOKEN_ID] = 0.0
        all_probs[step, _THINK_END_TOKEN_ID] = 0.0
    return _go(
        ids,
        finished=finished,
        input_len=input_len,
        logits=logits,
        all_probs=all_probs,
    )


def _token_score(tensor: torch.Tensor | None, step: int, token_id: int) -> float | None:
    if tensor is None:
        return None
    t = tensor.detach().cpu()
    if t.dim() == 3:
        return float(t[0, step, token_id].item())
    if t.dim() == 2:
        return float(t[step, token_id].item())
    if t.dim() == 1:
        return float(t[token_id].item())
    return None


def _fmt_score(value: float | None) -> str:
    return "NA" if value is None else f"{value:.8g}"


def _log_boundary_scores(case_name: str, go: GenerateOutputs) -> None:
    out = go.generate_outputs[0]
    ids = out.output_ids.cpu().int().tolist() if out.output_ids is not None else []
    for step, token_id in enumerate(ids):
        begin_logit = _token_score(out.logits, step, _THINK_BEGIN_TOKEN_ID)
        end_logit = _token_score(out.logits, step, _THINK_END_TOKEN_ID)
        begin_prob = _token_score(out.all_probs, step, _THINK_BEGIN_TOKEN_ID)
        end_prob = _token_score(out.all_probs, step, _THINK_END_TOKEN_ID)
        print(
            "[DashScDsv4SmokeScores] "
            f"case={case_name} step={step} sampled_token={token_id} "
            f"<think>_logit={_fmt_score(begin_logit)} "
            f"<think>_prob={_fmt_score(begin_prob)} "
            f"</think>_logit={_fmt_score(end_logit)} "
            f"</think>_prob={_fmt_score(end_prob)}"
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

    async def test_deepseek_v4_flash_thinking_controls_log_boundary_scores(
        self,
    ) -> None:
        tok = _Dsv4Tokenizer()
        env_cfg = _GenerateEnvCfg()
        think_runtime = build_think_runtime(tok, env_cfg, "deepseek_v4")
        cases = [
            {
                "name": "enable_thinking_false",
                "input_ids": [1001, 1002] + tok.encode("<think>\n"),
                "sampling": SamplingParams(max_new_tokens=16),
                "other": OtherParams(enable_thinking=False),
                "expected_in_think_mode": False,
                "expected_max_thinking_tokens": 0,
            },
            {
                "name": "max_new_think_tokens_zero",
                "input_ids": [2001, 2002] + tok.encode("<think>\n"),
                "sampling": SamplingParams(
                    max_new_tokens=16,
                    max_new_think_tokens=0,
                ),
                "other": OtherParams(),
                "expected_in_think_mode": False,
                "expected_max_thinking_tokens": 0,
            },
            {
                "name": "input_ends_with_end_think",
                "input_ids": [3001, 3002] + tok.encode("</think>"),
                "sampling": SamplingParams(max_new_tokens=16),
                "other": OtherParams(enable_thinking=False),
                "expected_in_think_mode": False,
                "expected_max_thinking_tokens": 0,
            },
            {
                "name": "enable_thinking_true",
                "input_ids": [4001, 4002] + tok.encode("<think>\n"),
                "sampling": SamplingParams(
                    max_new_tokens=16,
                    max_new_think_tokens=8,
                ),
                "other": OtherParams(enable_thinking=True),
                "expected_in_think_mode": True,
                "expected_max_thinking_tokens": 8,
            },
        ]

        for i, case in enumerate(cases):
            req = predict_v2_pb2.ModelInferRequest()
            req.id = f"trace-{case['name']}"
            req.model_name = "deepseek-v4-flash"
            input_ids = list(case["input_ids"])
            output_ids = [5100 + i * 10, 5101 + i * 10]
            _add_input_ids(req, input_ids)
            go = _go_with_boundary_scores(
                output_ids,
                finished=True,
                input_len=len(input_ids),
            )
            _log_boundary_scores(str(case["name"]), go)
            visitor = _MultiStreamVisitor([_FakeAsyncStream([go])])

            chunks = await _drain(
                iter_real_model_stream_infer(
                    req,
                    input_ids,
                    case["sampling"],
                    case["other"],
                    visitor,
                    rtp_llm_request_id=i + 1,
                    tokenizer=tok,
                    generate_env_config=env_cfg,
                    think_runtime=think_runtime,
                )
            )

            self.assertEqual(len(visitor.generate_inputs), 1)
            generate_config = visitor.generate_inputs[0].generate_config
            self.assertEqual(
                generate_config.in_think_mode,
                case["expected_in_think_mode"],
                case["name"],
            )
            self.assertEqual(
                generate_config.max_thinking_tokens,
                case["expected_max_thinking_tokens"],
                case["name"],
            )
            self.assertEqual(generate_config.begin_think_token_ids, _THINK_BEGIN_IDS)
            self.assertEqual(generate_config.end_think_token_ids, _THINK_END_IDS)
            self.assertEqual(_gen_ids(chunks[0]), output_ids)

            out = go.generate_outputs[0]
            for step in range(len(output_ids)):
                self.assertEqual(
                    _token_score(out.all_probs, step, _THINK_BEGIN_TOKEN_ID),
                    0.0,
                )
                self.assertEqual(
                    _token_score(out.all_probs, step, _THINK_END_TOKEN_ID),
                    0.0,
                )

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
