"""Request-to-token fixture and separate checkpoint-encoding differential."""

import hashlib
import importlib.util
import os
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.deepseekv41_renderer import DeepseekV41Renderer
from rtp_llm.openai.renderers.v41_recipe import convert_request, render_request


class RecipeReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = Path(os.environ["DSV41_REFERENCE_CHECKPOINT"])
        path = cls.checkpoint / "encoding" / "encoding.py"
        # Independently snapshotted HF 2bc89ac599031fa673cab993f1df02fc4a98c673.
        expected = "f64a67e5680a5621b9320585a9684967cd5c75a9b82e19d914cbc02845d72cab"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("checkpoint encoding differs from the pinned reference")
        spec = importlib.util.spec_from_file_location("v41_checkpoint_reference", path)
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)
        cls.tokenizer = Tokenizer.from_file(str(cls.checkpoint / "tokenizer.json"))

    def test_literal_request_token_ids(self):
        # IDs are read from the separately pinned tokenizer vocabulary, not
        # generated with the request renderer under test.
        prompt, _ = render_request(
            convert_request({"messages": [{"role": "user", "content": "hello"}]})
        )
        self.assertEqual(
            self.tokenizer.encode(prompt, add_special_tokens=False).ids,
            [0, 128803, 33310, 128804, 128822],
        )

    def test_checkpoint_effort_difference_is_explicit(self):
        messages = [{"role": "user", "content": "hello"}]
        official = self.reference.encode_messages(
            messages,
            thinking_mode="thinking",
            reasoning_effort="low",
            drop_thinking=False,
        )
        actual, _ = render_request(
            convert_request({"messages": messages, "reasoning_effort": "low"})
        )
        self.assertIn("Reasoning Effort: 25 ", official)
        self.assertIn("Reasoning Effort: 50 ", actual)
        self.assertEqual(
            actual,
            official.replace("Reasoning Effort: 25 ", "Reasoning Effort: 50 ", 1),
        )

    def test_checkpoint_integer_and_history_differential(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": "prior"},
            {"role": "system", "content": "Continue"},
        ]
        for budget in (1, 37, 75, 100):
            actual, _ = render_request(
                convert_request({"messages": messages, "reasoning_effort": budget})
            )
            expected = self.reference.encode_messages(
                messages,
                thinking_mode="thinking",
                reasoning_effort=budget,
                drop_thinking=False,
            )
            self.assertEqual(actual, expected)

    def test_renderer_preserves_canonical_inputs(self):
        tokenizer = self.tokenizer

        class Adapter:
            def encode(self, text):
                return tokenizer.encode(text, add_special_tokens=False).ids

        renderer = DeepseekV41Renderer.__new__(DeepseekV41Renderer)
        renderer.encoding_module = renderer._load_encoding_module(str(self.checkpoint))
        renderer.tokenizer = Adapter()
        renderer.think_mode = False
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {"type": "disabled"},
            }
        )
        rendered = renderer.render_chat(request)
        self.assertEqual(rendered.input_ids, [0, 128803, 33310, 128804, 128822])
        self.assertEqual(tuple(rendered.input_ids), rendered.v41_inputs.token_ids)
        self.assertEqual(
            renderer.encoding_sha256,
            hashlib.sha256(
                (self.checkpoint / "encoding" / "encoding.py").read_bytes()
            ).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
