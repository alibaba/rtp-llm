"""Opt-in K3 checkpoint tokenizer contract test.

Run with --test_env=K3_CKPT_PATH=/ssd/5/kimi-k3 on a checkpoint host.
"""

import os
import unittest
from types import SimpleNamespace

from transformers import AutoTokenizer

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.response_format_compiler import ReasoningFormat
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    kimi_k3_pending_prompt_token_count,
)


class KimiK3NativeTokenizerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        checkpoint = os.environ["K3_CKPT_PATH"]
        cls.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True
        )

    def test_xtml_transition_uses_native_special_token_encoding(self) -> None:
        transition = "<|close|>think<|sep|><|open|>response<|sep|>"
        expected = self.tokenizer.encode(transition)
        config = GenerateConfig(max_thinking_tokens=8)

        config.add_thinking_params(
            self.tokenizer,
            SimpleNamespace(
                think_mode="disabled",
                think_end_token_id=-1,
                think_end_tag="</think>",
            ),
            enable_thinking=True,
            reasoning_format=ReasoningFormat(
                tag_begin="",
                tag_end=transition,
                tag_end_native_encoding=True,
            ),
        )

        self.assertTrue(config.in_think_mode)
        self.assertEqual(config.end_think_token_ids, expected)
        self.assertEqual(
            config.structural_tag["format"]["elements"][0]["end"], transition
        )

    def test_usage_offset_matches_only_native_open_channel_suffix(self) -> None:
        think_suffix = self.tokenizer.encode("<|open|>think<|sep|>")
        response_suffix = self.tokenizer.encode("<|open|>response<|sep|>")
        prefix = [17, 18, 19]

        self.assertEqual(
            kimi_k3_pending_prompt_token_count(
                self.tokenizer, prefix + think_suffix
            ),
            len(think_suffix),
        )
        self.assertEqual(
            kimi_k3_pending_prompt_token_count(
                self.tokenizer, prefix + response_suffix
            ),
            len(response_suffix),
        )
        self.assertEqual(
            kimi_k3_pending_prompt_token_count(
                self.tokenizer, prefix + think_suffix + [20]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
