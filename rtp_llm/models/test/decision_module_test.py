"""CPU-only unit tests for the unified decision module.

Covers: linear (autojev-style) and pointer (kev-style) head math against a
manual einsum reference, renderer request/response schema roundtrip, and the
factory dispatch on the checkpoint's "decision" config block.
"""

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import TaskType

from rtp_llm.models.downstream_modules import decision_module
from rtp_llm.models.downstream_modules.decision_module import (
    DecisionHandler,
    DecisionModule,
    DecisionRenderer,
    DecisionRequest,
    DecisionResponse,
    create_decision_module,
    question_labels,
    question_options,
)
from rtp_llm.models.downstream_modules.utils import (
    create_custom_module,
)

HIDDEN = 8
NUM_LABELS = 3
PROJ = 4
OPT_END_ID = 901
DECIDE_ID = 902


def _linear_block():
    return {
        "format_version": 1,
        "head_type": "linear",
        "prompt_format": "chat",
        "head_weight": "decision_head.weight",
        "num_labels": NUM_LABELS,
        "temperature": 2.0,
    }


def _pointer_block():
    return {
        "format_version": 1,
        "head_type": "pointer",
        "prompt_format": "kev_branch",
        "proj_dim": PROJ,
        "scale": 1.0 / math.sqrt(PROJ),
        "markers": {
            "state": "<s>",
            "question": "<q>",
            "option": "<o>",
            "option_end": "</o>",
            "decide": "<d>",
        },
        "marker_token_ids": {"option_end": OPT_END_ID, "decide": DECIDE_ID},
        "head_query_weight": "decision_head.q.weight",
        "head_query_bias": "decision_head.q.bias",
        "head_key_weight": "decision_head.k.weight",
        "head_key_bias": "decision_head.k.bias",
        "temperature": 1.5,
    }


def _config(ckpt_path, block):
    (Path(ckpt_path) / "config.json").write_text(json.dumps({"decision": block}))
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.hidden_size = HIDDEN
    config.data_type = "fp32"
    return config


class _FakeGenerator:
    """Captures prompt strings instead of tokenizing."""

    instances = []

    def __init__(self, tokenizer, config):
        self.prompts = None
        _FakeGenerator.instances.append(self)

    def generate(self, prompts):
        self.prompts = list(prompts)
        lengths = torch.tensor([10] * len(prompts), dtype=torch.int32)
        return SimpleNamespace(
            token_ids=torch.zeros(10 * len(prompts), dtype=torch.int32),
            token_type_ids=None,
            input_lengths=lengths,
        )


def _patch_generator():
    return patch.object(decision_module, "CommonInputGenerator", _FakeGenerator)


class DecisionHandlerTest(unittest.TestCase):
    def test_linear_head_matches_manual_einsum(self):
        with tempfile.TemporaryDirectory() as d:
            config = _config(d, _linear_block())
            torch.manual_seed(0)
            weight = torch.randn(NUM_LABELS, HIDDEN)
            handler = DecisionHandler(config, _linear_block())
            names = [w.name.split(".", 1)[1] for w in handler.custom_weight_info()]
            self.assertEqual(names, ["decision_head.weight"])
            handler.init({"decision_head.weight": weight})

            # two prompts of lengths 3 and 2 -> combo hidden states
            hidden = torch.randn(5, HIDDEN)
            input_ids = torch.arange(5, dtype=torch.int32)
            input_lengths = torch.tensor([3, 2], dtype=torch.int32)
            out = handler.forward(input_ids, hidden, input_lengths)
            self.assertEqual(len(out), 2)
            causal = config.attn_config.is_causal
            picked = hidden[[2, 4]] if causal else hidden[[0, 3]]
            ref = torch.einsum("lh,bh->bl", weight, picked) / 2.0
            for row, ref_row in zip(out, ref):
                torch.testing.assert_close(row, ref_row)
                probs = torch.softmax(row, dim=-1)
                self.assertAlmostEqual(probs.sum().item(), 1.0, places=6)
                self.assertEqual(
                    probs.argmax().item(), ref_row.argmax().item()
                )

    def test_pointer_head_reads_marker_positions(self):
        with tempfile.TemporaryDirectory() as d:
            config = _config(d, _pointer_block())
            torch.manual_seed(1)
            qw, qb = torch.randn(PROJ, HIDDEN), torch.randn(PROJ)
            kw, kb = torch.randn(PROJ, HIDDEN), torch.randn(PROJ)
            handler = DecisionHandler(config, _pointer_block())
            handler.init(
                {
                    "decision_head.q.weight": qw,
                    "decision_head.q.bias": qb,
                    "decision_head.k.weight": kw,
                    "decision_head.k.bias": kb,
                }
            )
            # branch 1: two options then decide; branch 2: three options then decide
            ids_b1 = [5, 6, OPT_END_ID, 7, OPT_END_ID, 8, DECIDE_ID]
            ids_b2 = [9, OPT_END_ID, 1, OPT_END_ID, 2, OPT_END_ID, 3, DECIDE_ID]
            input_ids = torch.tensor(ids_b1 + ids_b2, dtype=torch.int32)
            input_lengths = torch.tensor([len(ids_b1), len(ids_b2)], dtype=torch.int32)
            hidden = torch.randn(len(ids_b1) + len(ids_b2), HIDDEN)

            out = handler.forward(input_ids, hidden, input_lengths)
            self.assertEqual([o["scores"].numel() for o in out], [2, 3])

            def ref_scores(opt_abs, decide_abs):
                h_opts = hidden[opt_abs]
                h_dec = hidden[decide_abs]
                q = torch.einsum("ph,h->p", qw, h_dec) + qb
                k = torch.einsum("ph,nh->np", kw, h_opts) + kb
                return (k @ q) * (1.0 / math.sqrt(PROJ)) / 1.5

            refs = [
                ref_scores([2, 4], 6),
                ref_scores([8, 10, 12], 14),
            ]
            for item, ref in zip(out, refs):
                scores = item["scores"]
                torch.testing.assert_close(scores, ref)
                self.assertAlmostEqual(
                    torch.softmax(scores, dim=-1).sum().item(), 1.0, places=6
                )

    def test_pointer_head_missing_marker_raises(self):
        with tempfile.TemporaryDirectory() as d:
            config = _config(d, _pointer_block())
            handler = DecisionHandler(config, _pointer_block())
            handler.init(
                {
                    "decision_head.q.weight": torch.randn(PROJ, HIDDEN),
                    "decision_head.q.bias": torch.randn(PROJ),
                    "decision_head.k.weight": torch.randn(PROJ, HIDDEN),
                    "decision_head.k.bias": torch.randn(PROJ),
                }
            )
            with self.assertRaisesRegex(ValueError, "option-end"):
                handler.forward(
                    torch.tensor([1, DECIDE_ID], dtype=torch.int32),
                    torch.randn(2, HIDDEN),
                    torch.tensor([2], dtype=torch.int32),
                )
            with self.assertRaisesRegex(ValueError, "decide"):
                handler.forward(
                    torch.tensor([1, OPT_END_ID], dtype=torch.int32),
                    torch.randn(2, HIDDEN),
                    torch.tensor([2], dtype=torch.int32),
                )

    def test_linear_weight_shape_validation(self):
        with tempfile.TemporaryDirectory() as d:
            config = _config(d, _linear_block())
            handler = DecisionHandler(config, _linear_block())
            with self.assertRaises(ValueError):
                handler.init({"decision_head.weight": torch.randn(2, HIDDEN + 1)})


class DecisionRendererTest(unittest.TestCase):
    def test_request_validation(self):
        req = DecisionRequest(
            document="doc",
            questions=[
                {"id": "q1", "type": "choice", "prompt": "pick", "options": ["a", "b"]},
                {"id": "q2", "type": "noul", "prompt": "is it?"},
                {"id": "q3", "type": "score", "options": ["bad", "ok", "good"]},
            ],
        )
        self.assertEqual(question_options(req.questions[1]), ["no", "yes"])
        self.assertEqual(question_labels(req.questions[1], ["no", "yes"]), ["false", "true"])
        self.assertEqual(
            question_labels(req.questions[2], ["bad", "ok", "good"]), ["0", "1", "2"]
        )
        with self.assertRaises(ValueError):
            question_options(req.questions[0].model_copy(update={"options": ["only"]}))
        with self.assertRaises(ValueError):
            question_options(req.questions[0].model_copy(update={"type": "bogus"}))

    def test_kev_prompt_layout_and_response_roundtrip(self):
        with tempfile.TemporaryDirectory() as d, _patch_generator():
            config = _config(d, _pointer_block())
            renderer = DecisionRenderer(config, MagicMock(), _pointer_block())
            request = renderer.render_request(
                {
                    "document": "the doc <|fim_prefix|> text",
                    "questions": [
                        {
                            "id": "q1",
                            "type": "choice",
                            "prompt": "topic?",
                            "options": ["sports", "tech"],
                        }
                    ],
                }
            )
            inputs = renderer.create_input(request)
            gen = _FakeGenerator.instances[-1]
            self.assertEqual(len(gen.prompts), 1)
            self.assertEqual(
                gen.prompts[0],
                "<s>the doc < |fim_prefix| > text<q>topic?"
                "<o>sports</o><o>tech</o><d>",
            )
            outputs = SimpleNamespace(outputs=[{"scores": torch.tensor([0.0, 2.0])}])
            response = asyncio.run(renderer.render_response(request, inputs, outputs))
            parsed = DecisionResponse(**response)
            probs = parsed.results[0].probs
            self.assertEqual(parsed.results[0].question_id, "q1")
            self.assertEqual(set(probs), {"sports", "tech"})
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
            self.assertGreater(probs["tech"], probs["sports"])

    def test_chat_prompt_messages_and_linear_masking(self):
        with tempfile.TemporaryDirectory() as d, _patch_generator():
            config = _config(d, _linear_block())
            tokenizer = MagicMock()
            tokenizer.apply_chat_template.return_value = "rendered"
            renderer = DecisionRenderer(config, tokenizer, _linear_block())
            request = renderer.render_request(
                {
                    "document": "review text",
                    "questions": [
                        {"id": "s", "type": "score", "options": ["bad", "mid", "good"]}
                    ],
                }
            )
            inputs = renderer.create_input(request)
            (messages,), kwargs = tokenizer.apply_chat_template.call_args
            self.assertEqual(messages[0]["role"], "system")
            user = messages[1]["content"]
            self.assertIn("State:\nreview text", user)
            self.assertIn("A: bad\nB: mid\nC: good", user)
            self.assertEqual(kwargs["add_generation_prompt"], True)

            # handler emits all NUM_LABELS logits in one [batch, num_labels] tensor;
            # only the question's 3 count.
            outputs = SimpleNamespace(outputs=torch.tensor([[1.0, 0.0, -1.0]]))
            response = asyncio.run(renderer.render_response(request, inputs, outputs))
            probs = DecisionResponse(**response).results[0].probs
            self.assertEqual(set(probs), {"0", "1", "2"})
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
            self.assertAlmostEqual(
                probs["0"], math.exp(1.0) / (math.exp(1.0) + 1 + math.exp(-1.0)), places=6
            )


class DecisionFactoryTest(unittest.TestCase):
    def test_factory_dispatch(self):
        with tempfile.TemporaryDirectory() as d, _patch_generator():
            config = _config(d, _linear_block())
            config.task_type = TaskType.SEQ_CLASSIFICATION
            module = create_custom_module(config, MagicMock())
            self.assertIsInstance(module, DecisionModule)

    def test_factory_falls_through_without_decision_block(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text("{}")
            config = ModelConfig()
            config.ckpt_path = d
            config.task_type = TaskType.SEQ_CLASSIFICATION
            self.assertIsNone(create_decision_module(config, MagicMock()))
            # the opensource entrypoint falls back to ClassifierModule
            with patch(
                "rtp_llm.models.downstream_modules.classifier.classifier.ClassifierModule"
            ) as classifier_cls:
                module = create_custom_module(config, MagicMock())
            self.assertIs(module, classifier_cls.return_value)


if __name__ == "__main__":
    unittest.main()
