"""CPU-only tests for the decider_schema decision format.

The stub tokenizer is one token per character, so fragment boundaries and
slot arithmetic are exact; the head math is checked through the full handler
path against a manual reference.
"""

import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models.downstream_modules.clef_joint_head import decode_span_table
from rtp_llm.models.downstream_modules.decider_decision import (
    encode_decider_record,
)

HIDDEN = 16
NUM_LABELS = 12  # stub head: enough for the option counts used here
SENTINEL = 999
DIGIT_IDS = list(range(900, 910))


class _StubTokenizer:
    """One token per character, except uppercase labels which stay single.

    Deciding the label table needs every A..Z / AA.. two-letter string to be
    one token (the real BPE tokenizer does this); the stub maps them to
    dedicated ids while every other character still encodes per character.
    """

    eos_token_id = 0
    pad_token_id = 0
    _LABEL_IDS = None

    @classmethod
    def label_ids(cls):
        import string

        if cls._LABEL_IDS is None:
            names = list(string.ascii_uppercase) + [
                a + b
                for a in string.ascii_uppercase
                for b in string.ascii_uppercase
            ]
            cls._LABEL_IDS = {name: 1000 + i for i, name in enumerate(names)}
        return cls._LABEL_IDS

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        text = str(text)
        if text in self.label_ids():
            return SimpleNamespace(input_ids=[self.label_ids()[text]])
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def _decider_settings():
    return {
        "format_version": 1,
        "head_type": "linear",
        "prompt_format": "decider_schema",
        "head_weight": "decision_head.weight",
        "num_labels": NUM_LABELS,
        "codes": ["A", "B", "C", "D"],
        "labels": ["A", "B", "C", "D"],
        "temperature": 1.08,
        "sentinel_token_id": SENTINEL,
        "table_digit_token_ids": DIGIT_IDS,
        "max_ctx_tokens": 1024,
        "escape_special_tokens": True,
    }


class DeciderEncodingTest(unittest.TestCase):
    def _questions(self):
        return [
            {
                "id": "dept",
                "question": "Which department should handle this?",
                "options": ["billing", "technical", "sales"],
            },
            {
                "id": "urgent",
                "question": "How urgent is this?",
                "options": ["low", "medium", "high"],
            },
        ]

    def test_layout_and_slots(self):
        encoded = encode_decider_record(
            _StubTokenizer(),
            "My card was charged twice for the same purchase.",
            self._questions(),
            sentinel_token_id=SENTINEL,
            table_digit_token_ids=DIGIT_IDS,
        )
        body = encoded.input_ids[: encoded.body_length]
        text = "".join(chr(t) for t in body)
        # state_first: Context, then per-question blocks each ending "Answer k: ("
        self.assertTrue(text.startswith("Context:\nMy card was charged"))
        self.assertIn("\n\nQuestion 1: Which department should handle this?\nOptions:", text)
        self.assertIn("\n(A) billing\n(B) technical\n(C) sales", text)
        self.assertIn("\n\nQuestion 2: How urgent is this?\nOptions:", text)
        self.assertTrue(text.endswith("\nAnswer 2: ("))
        # both slots decoded from the appended table
        table, table_open = decode_span_table(
            list(encoded.input_ids), SENTINEL, DIGIT_IDS
        )
        self.assertEqual(table_open, encoded.body_length)
        self.assertEqual(len(table.questions), 2)
        for q, slot in zip(table.questions, encoded.slots):
            self.assertEqual(q.question_span, (slot, slot + 1))
            # the slot is the "(" token of its Answer line
            self.assertEqual(chr(body[slot]), "(")

    def test_option_count_validation(self):
        with self.assertRaisesRegex(ValueError, "2..255 options"):
            encode_decider_record(
                _StubTokenizer(),
                "ctx",
                [{"question": "q", "options": ["only"]}],
                sentinel_token_id=SENTINEL,
                table_digit_token_ids=DIGIT_IDS,
            )


class DeciderHandlerIntegrationTest(unittest.TestCase):
    def _run(self):
        from rtp_llm.models.downstream_modules.decision_module import (
            DecisionHandler,
            DecisionRenderer,
        )

        settings = _decider_settings()
        torch.manual_seed(11)
        weight = torch.randn(NUM_LABELS, HIDDEN)
        handler = DecisionHandler(
            SimpleNamespace(
                hidden_size=HIDDEN,
                data_type="fp32",
                attn_config=SimpleNamespace(is_causal=True),
            ),
            settings,
        )
        handler.init({"decision_head.weight": weight})

        tokenizer = _StubTokenizer()
        with mock.patch(
            "rtp_llm.models.downstream_modules.decision_module.CommonInputGenerator"
        ):
            renderer = DecisionRenderer(
                SimpleNamespace(hidden_size=HIDDEN, max_seq_len=4096),
                tokenizer,
                settings,
            )
        request = renderer.render_request(
            {
                "document": "My card was charged twice for the same purchase.",
                "questions": [
                    {
                        "id": "dept",
                        "type": "choice",
                        "prompt": "Which department should handle this?",
                        "options": ["billing", "technical", "sales"],
                    },
                    {
                        "id": "urgent",
                        "type": "choice",
                        "prompt": "How urgent is this?",
                        "options": ["low", "medium", "high"],
                    },
                ],
            }
        )
        inputs = renderer.create_input(request)
        self.assertEqual(int(inputs.input_lengths[0]), len(inputs.token_ids))

        seq_len = int(inputs.input_lengths[0])
        hidden = torch.randn(seq_len, HIDDEN)
        out = handler.forward(inputs.token_ids, hidden, inputs.input_lengths)
        # one map per record with position-keyed per-question slices
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0]), {"scores_0", "scores_1"})
        self.assertEqual([out[0][k].numel() for k in ("scores_0", "scores_1")], [NUM_LABELS, NUM_LABELS])

        # manual reference: slot hidden -> linear head -> /temperature
        table, table_open = decode_span_table(
            inputs.token_ids.tolist(), SENTINEL, DIGIT_IDS
        )
        for i, q in enumerate(table.questions):
            slot_state = hidden[q.question_span[0]]
            ref = (weight @ slot_state).float() / 1.08
            torch.testing.assert_close(out[0][f"scores_{i}"], ref)

        # response: first-K over the question's own options, softmax
        outputs = SimpleNamespace(outputs=out)
        response = asyncio.run(renderer.render_response(request, inputs, outputs))
        results = {r["question_id"]: r["probs"] for r in response["results"]}
        self.assertEqual(set(results["dept"]), {"billing", "technical", "sales"})
        self.assertEqual(set(results["urgent"]), {"low", "medium", "high"})
        for probs in results.values():
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)


class ConvertDeciderTest(unittest.TestCase):
    def test_convert_decider(self):
        from rtp_llm.tools import convert_decision_head as tool
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            ckpt = Path(tmp) / "decider"
            ckpt.mkdir()
            hidden = 8
            lm_head = torch.randn(256, hidden)
            save_file({"lm_head.weight": lm_head}, str(ckpt / "s.safetensors"))
            (ckpt / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe_text",
                        "vision_start_token_id": 1,
                        "vision_end_token_id": 2,
                    }
                )
            )
            (ckpt / "decider_config.json").write_text(
                json.dumps({"temperature": 1.08})
            )
            (ckpt / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 0},
                        "weight_map": {"lm_head.weight": "s.safetensors"},
                    }
                )
            )
            # tokenizer files: label-table + digit + sentinel resolution
            # label table needs 255 single-token uppercase strings; provide a
            # char-keyed vocab and patch the tokenizer loader instead.
            (ckpt / "tokenizer.json").write_text(
                json.dumps({"added_tokens": [{"content": "<|vision_pad|>", "id": 555}]})
            )
            (ckpt / "tokenizer_config.json").write_text(json.dumps({}))

            class _Tok:
                def encode(self, text, add_special_tokens=False):
                    # A..Z then AA, AB... all "single tokens" with sequential ids
                    return [100 + (ord(text[0]) - 65) + 26 * (len(text) - 1)]

            with mock.patch.object(
                tool,
                "_decider_label_table",
                return_value=[(f"L{i}", i) for i in range(255)],
            ), mock.patch.object(
                tool, "_resolve_digit_token_ids", return_value=list(range(10))
            ):
                block = tool.convert_decider(str(ckpt), str(ckpt))

            self.assertEqual(block["prompt_format"], "decider_schema")
            self.assertEqual(block["head_type"], "linear")
            self.assertEqual(block["num_labels"], 255)
            self.assertEqual(block["temperature"], 1.08)
            self.assertEqual(block["sentinel_token_id"], 555)
            config = json.loads((ckpt / "config.json").read_text())
            self.assertNotIn("vision_start_token_id", config)
            self.assertEqual(config["decision"], block)

            head = save_file  # noqa: F841 (readability)
            from safetensors.torch import load_file as lf

            stored = lf(str(ckpt / "decision_head.safetensors"))
            expected = lm_head.index_select(
                0, torch.tensor(list(range(255)), dtype=torch.long)
            )
            torch.testing.assert_close(stored["decision_head.weight"], expected)
            idx = json.loads((ckpt / "model.safetensors.index.json").read_text())
            self.assertEqual(
                idx["weight_map"]["decision_head.weight"], "decision_head.safetensors"
            )


if __name__ == "__main__":
    unittest.main()
