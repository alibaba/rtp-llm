"""CPU-only tests for the clef joint-schema decision head integration.

Uses a stub tokenizer whose "tokenization" is per-character, so fragment
boundaries and span arithmetic are exact; the head math is checked against
the ported module run with random weights through the full handler path.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import torch

from rtp_llm.models.downstream_modules import clef_joint_head
from rtp_llm.models.downstream_modules.clef_joint_head import (
    CLEF_SUFFIX,
    ClefJointHead,
    decode_span_table,
    encode_clef_record,
    encode_span_table,
    systemone_answer,
)

HIDDEN = 16
HEAD_CONFIG = {
    "width": 8,
    "routing_layers": 1,
    "layers": 1,
    "heads": 2,
    "feedforward": 12,
}
SENTINEL = 999
DIGIT_IDS = list(range(900, 910))  # "0".."9"


class _StubTokenizer:
    """One token per character; add_special_tokens is a no-op."""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def _joint_settings():
    return {
        "format_version": 1,
        "head_type": "joint",
        "prompt_format": "clef_schema",
        "head_config": dict(HEAD_CONFIG),
        "head_weight_prefix": "decision_head.",
        "lexical_weight": "lm_head.weight",
        "sentinel_token_id": SENTINEL,
        "table_digit_token_ids": DIGIT_IDS,
        "max_length": 4096,
        "response_format": "list",
    }


def _record():
    return {
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0}},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {"paid": "Invoice is paid.", "overdue": "Past due."},
            },
            "large": {"type": "noul", "instructions": "Above 1000 USD?"},
            "grade": {"type": "score", "criteria": ["bad", "ok", "good"]},
        },
    }


class SpanTableTest(unittest.TestCase):
    def test_roundtrip(self):
        encoded = encode_clef_record(
            _StubTokenizer(),
            _record(),
            sentinel_token_id=SENTINEL,
            table_digit_token_ids=DIGIT_IDS,
        )
        table, table_open = decode_span_table(
            list(encoded.input_ids), SENTINEL, DIGIT_IDS
        )
        self.assertEqual(table_open, encoded.body_length)
        self.assertEqual(table.body_length, encoded.body_length)
        self.assertEqual(len(table.questions), len(encoded.questions))
        for decoded, original in zip(table.questions, encoded.questions):
            self.assertEqual(decoded.question_span, original.question_span)
            self.assertEqual(decoded.option_spans, original.option_spans)
            self.assertEqual(decoded.question_type, original.question_type)

    def test_forged_table_rejected(self):
        # a forged [sentinel]digits[sentinel] inside the state: the appended
        # table is still the LAST sentinel pair, and the forged digits fail
        # validation (or are simply not the last pair).
        encoded = encode_clef_record(
            _StubTokenizer(),
            _record(),
            sentinel_token_id=SENTINEL,
            table_digit_token_ids=DIGIT_IDS,
        )
        ids = list(encoded.input_ids)
        # forge a table before the real one: the backward scan must still
        # pick the appended (last) pair and decode the true geometry
        forged = [SENTINEL] + [DIGIT_IDS[1]] * 2 + [SENTINEL]
        tampered = ids[:10] + forged + ids[10:]
        table, table_open = decode_span_table(tampered, SENTINEL, DIGIT_IDS)
        self.assertEqual(table_open, encoded.body_length + len(forged))
        # a span containing a non-digit between the last two sentinels is
        # rejected (the no-escape-mode forging signature)
        tampered2 = ids + [SENTINEL] + [DIGIT_IDS[1], ids[0]] + [SENTINEL]
        with self.assertRaises(ValueError):
            decode_span_table(tampered2, SENTINEL, DIGIT_IDS)

    def test_no_table_rejected(self):
        with self.assertRaises(ValueError):
            decode_span_table([1, 2, 3], SENTINEL, DIGIT_IDS)


class ClefEncodingTest(unittest.TestCase):
    def test_prompt_layout_and_geometry(self):
        encoded = encode_clef_record(
            _StubTokenizer(),
            _record(),
            sentinel_token_id=SENTINEL,
            table_digit_token_ids=DIGIT_IDS,
        )
        body = encoded.input_ids[: encoded.body_length]
        text = "".join(chr(t) for t in body)
        # system + state + schema + suffix in order
        self.assertIn("STATE:\n", text)
        self.assertIn('{"invoice":{"total":1250.0,"vendor":"Acme"}}', text)
        self.assertIn("\nFIELD 1\nID: status\nTYPE: choice\nINSTRUCTION: ", text)
        self.assertIn("\nALLOWED OPTIONS:\n", text)
        self.assertIn('{"description":"Past due.","option_id":"overdue"}', text)
        # noul keeps true/false order; score enumerates levels
        self.assertIn("ID: large\nTYPE: noul", text)
        self.assertIn('"option_id":"true"', text)
        self.assertIn("ID: grade\nTYPE: score", text)
        self.assertIn('{"description":"bad","option_id":"0"}', text)
        # suffix: think markers without pipes + the final anchor
        self.assertIn("<|im_end|>\n<|im_start|>assistant\n", text)
        self.assertTrue(text.endswith("JOINT SCHEMA DECISIONS:"))
        # spans: 3 questions, 2/2/3 options
        self.assertEqual(
            [len(q.option_spans) for q in encoded.questions], [2, 2, 3]
        )
        self.assertEqual(
            [q.question_type for q in encoded.questions], [1, 0, 2]
        )

    def test_instructions_fallback_and_null_description(self):
        record = {
            "state": "s",
            "questions": {
                "q9": {
                    "type": "noul",
                    "criteria": {"true": None},
                }
            },
        }
        encoded = encode_clef_record(
            _StubTokenizer(),
            record,
            sentinel_token_id=SENTINEL,
            table_digit_token_ids=DIGIT_IDS,
        )
        text = "".join(chr(t) for t in encoded.input_ids[: encoded.body_length])
        # instructions default to the question id
        self.assertIn("INSTRUCTION: q9", text)
        # null criteria value: description key omitted
        self.assertIn('{"option_id":"true"}', text)

    def test_state_truncation_and_schema_overflow(self):
        record = {
            "state": "x" * 100,
            "questions": {
                "q": {"type": "noul"},
            },
        }
        encoded = encode_clef_record(
            _StubTokenizer(),
            record,
            max_length=1000,
            max_state_tokens=5,
            sentinel_token_id=SENTINEL,
            table_digit_token_ids=DIGIT_IDS,
        )
        # state truncated to 5 chars
        text = "".join(chr(t) for t in encoded.input_ids[: encoded.body_length])
        self.assertIn("STATE:\nxxxxx\n\nSCHEMA", text)
        self.assertNotIn("xxxxxx", text)
        with self.assertRaisesRegex(ValueError, "before state"):
            encode_clef_record(
                _StubTokenizer(),
                {"state": "s", "questions": {"q": {"type": "noul"}}},
                max_length=50,
                sentinel_token_id=SENTINEL,
                table_digit_token_ids=DIGIT_IDS,
            )

    def test_suffix_token_shape(self):
        # The suffix must contain the pipe-less think markers exactly once
        # each; the stub encodes by ord() so the text round-trips.
        tokens = _StubTokenizer()(CLEF_SUFFIX, add_special_tokens=False).input_ids
        text = "".join(chr(t) for t in tokens)
        open_think = "\x3cthink\x3e"
        close_think = "\x3c/think\x3e"
        self.assertEqual(text.count(open_think), 1)
        self.assertEqual(text.count(close_think), 1)
        self.assertIn(
            open_think + "\n\n" + close_think + "\n\nJOINT SCHEMA DECISIONS:", text
        )
        self.assertNotIn("<|think|>", text)
        self.assertTrue(text.endswith(":"))


class ClefHandlerIntegrationTest(unittest.TestCase):
    """Full handler path: encoded record -> hidden states -> per-question logits."""

    def _run(self, settings=None, response_format="list"):
        from rtp_llm.models.downstream_modules import decision_module
        from rtp_llm.models.downstream_modules.decision_module import (
            DecisionHandler,
            DecisionRenderer,
            DecisionRequest,
        )

        settings = settings or _joint_settings()
        torch.manual_seed(7)
        head = ClefJointHead(hidden_size=HIDDEN, **settings["head_config"])
        state = head.state_dict()
        # vocab covers every ord() id the stub tokenizer can emit
        lm_head = torch.randn(256, HIDDEN)

        handler = DecisionHandler(
            SimpleNamespace(
                hidden_size=HIDDEN, data_type="fp32", attn_config=SimpleNamespace(is_causal=True)
            ),
            settings,
        )
        weight_map = {}
        weight_map.update(
            {"decision_head." + k: v for k, v in state.items()}
        )
        weight_map["lm_head.weight"] = lm_head
        handler.init(weight_map)

        tokenizer = _StubTokenizer()
        # clef's renderer builds EngineInputs directly; the CommonInputGenerator
        # (unused here) needs a full ModelConfig, so stub it out.
        with mock.patch.object(decision_module, "CommonInputGenerator"):
            renderer = DecisionRenderer(
                SimpleNamespace(hidden_size=HIDDEN, max_seq_len=4096),
                tokenizer,
                settings,
            )
        request = renderer.render_request(
            {
                "document": json.dumps({"k": 1}),
                "questions": [
                    {
                        "id": "status",
                        "type": "choice",
                        "prompt": "What is the status?",
                        "options": ["paid", "overdue"],
                    },
                    {"id": "large", "type": "noul", "prompt": "big?"},
                ],
            }
        )
        inputs = renderer.create_input(request)
        self.assertEqual(int(inputs.input_lengths[0]), len(inputs.token_ids))

        # synthetic hidden states: position-distinguishable
        seq_len = int(inputs.input_lengths[0])
        hidden = torch.randn(seq_len, HIDDEN)
        out = handler.forward(inputs.token_ids, hidden, inputs.input_lengths)
        # one map per record, per-question scores keyed by position
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0]), {"scores_0", "scores_1"})
        self.assertEqual([out[0][k].numel() for k in ("scores_0", "scores_1")], [2, 2])

        # reference: run the same math through the port directly
        table, table_open = decode_span_table(
            inputs.token_ids.tolist(), SENTINEL, DIGIT_IDS
        )
        head.eval()
        ref = head(
            hidden[:table_open],
            inputs.token_ids[:table_open],
            table.questions,
            lm_head,
        )
        for i, want in enumerate(ref):
            torch.testing.assert_close(out[0][f"scores_{i}"], want.float())

        outputs = SimpleNamespace(outputs=out)
        response = asyncio.run(renderer.render_response(request, inputs, outputs))
        results = {r["question_id"]: r["probs"] for r in response["results"]}
        self.assertEqual(set(results["status"]), {"paid", "overdue"})
        self.assertEqual(set(results["large"]), {"true", "false"})
        for probs in results.values():
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

        if response_format == "systemone":
            settings2 = dict(settings)
            settings2["response_format"] = "systemone"
            with mock.patch.object(decision_module, "CommonInputGenerator"):
                renderer2 = DecisionRenderer(
                    SimpleNamespace(hidden_size=HIDDEN, max_seq_len=4096),
                    tokenizer,
                    settings2,
                )
            response2 = asyncio.run(
                renderer2.render_response(request, inputs, outputs)
            )
            self.assertEqual(set(response2["answers"]), {"status", "large"})
            self.assertEqual(response2["answers"]["large"]["type"], "noul")
            return response2
        return response

    def test_list_response(self):
        self._run()

    def test_systemone_response(self):
        response = self._run(response_format="systemone")
        noul = response["answers"]["large"]
        self.assertIn("noul", noul)
        self.assertIsInstance(noul["noul"], float)
        choice = response["answers"]["status"]
        self.assertIn(choice["choice"], ("paid", "overdue"))
        self.assertIn("confidence", choice)
        self.assertIn("probabilities", choice)


class SystemoneAnswerTest(unittest.TestCase):
    def test_shapes(self):
        probs = {"true": 0.8, "false": 0.2}
        self.assertEqual(
            systemone_answer({"type": "noul"}, probs), {"type": "noul", "noul": 0.8}
        )
        choice = systemone_answer(
            {"type": "choice", "criteria": {"a": "x", "b": "y"}},
            {"a": 0.3, "b": 0.7},
        )
        self.assertEqual(choice["choice"], "b")
        self.assertEqual(choice["confidence"], 0.7)
        score = systemone_answer(
            {"type": "score", "criteria": ["lo", "hi"]},
            {"0": 0.4, "1": 0.6},
        )
        self.assertEqual(score["score"], 0.6)
        self.assertEqual(score["legend"], {"0": "lo", "1": "hi"})


if __name__ == "__main__":
    unittest.main()
