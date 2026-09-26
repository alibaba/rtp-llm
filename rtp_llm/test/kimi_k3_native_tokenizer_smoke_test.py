"""Opt-in K3 checkpoint tokenizer contract test.

Run with --test_env=K3_CKPT_PATH=/ssd/5/kimi-k3 on a checkpoint host.
"""

import json
import os
import unittest
from types import SimpleNamespace

from transformers import AutoTokenizer
import xgrammar as xgr

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.grammar_tokenizer_info import build_grammar_tokenizer_info_json
from rtp_llm.config.response_format_compiler import ReasoningFormat
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    kimi_k3_pending_prompt_token_count,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer


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

    def test_native_xgrammar_masks_eos_until_full_xtml_boundary_and_final(self) -> None:
        """Exercise the production tokenizer serializer against pinned Python xgrammar."""
        with open(os.path.join(os.environ["K3_CKPT_PATH"], "config.json")) as f:
            model_vocab_size = json.load(f)["text_config"]["vocab_size"]
        eos_id = self.tokenizer.eos_token_id
        self.assertIsInstance(eos_id, int)
        tokenizer_info_json = build_grammar_tokenizer_info_json(
            self.tokenizer,
            model_vocab_size=model_vocab_size,
            stop_token_ids=[eos_id],
        )
        tokenizer_info = xgr.TokenizerInfo.deserialize_json(tokenizer_info_json)
        transition = "<|close|>think<|sep|><|open|>response<|sep|>"
        boundary_ids = self.tokenizer.encode(transition)
        think_ids = self.tokenizer.encode("x")
        final_ids = self.tokenizer.encode("a")
        self.assertEqual(len(think_ids), 1)
        self.assertEqual(len(final_ids), 1)
        self.assertGreater(len(boundary_ids), 1)

        grammar = {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    ReasoningFormat(tag_begin="", tag_end=transition).prefix_format(1),
                    {"type": "regex", "pattern": "a"},
                ],
            },
        }
        compiled = xgr.GrammarCompiler(tokenizer_info, max_threads=2).compile_structural_tag(
            grammar
        )
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        bitmask = xgr.allocate_token_bitmask(1, model_vocab_size)

        def allows(token_id: int) -> bool:
            xgr.reset_token_bitmask(bitmask)
            matcher.fill_next_token_bitmask(bitmask)
            return bool(int(bitmask[0, token_id // 32]) & (1 << (token_id % 32)))

        self.assertTrue(allows(think_ids[0]))
        self.assertFalse(allows(eos_id))
        self.assertTrue(matcher.accept_token(think_ids[0]))

        for token_id in boundary_ids:
            self.assertTrue(allows(token_id), f"transition token {token_id} was masked")
            self.assertFalse(allows(think_ids[0]))
            self.assertFalse(allows(eos_id))
            self.assertTrue(matcher.accept_token(token_id))

        self.assertTrue(allows(final_ids[0]))
        self.assertFalse(allows(think_ids[0]))
        self.assertFalse(allows(eos_id))
        self.assertTrue(matcher.accept_token(final_ids[0]))
        self.assertTrue(matcher.is_terminated())

    def test_native_xgrammar_accepts_four_parallel_tool_calls(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Four cities"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Return weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "parallel_tool_calls": True,
                "thinking": {"type": "disabled"},
            }
        )
        config = GenerateConfig()
        KimiK3Renderer.__new__(KimiK3Renderer).apply_chat_completion_constraints(
            request, config
        )
        with open(os.path.join(os.environ["K3_CKPT_PATH"], "config.json")) as f:
            model_vocab_size = json.load(f)["text_config"]["vocab_size"]
        tokenizer_info = xgr.TokenizerInfo.deserialize_json(
            build_grammar_tokenizer_info_json(
                self.tokenizer,
                model_vocab_size=model_vocab_size,
                stop_token_ids=[self.tokenizer.eos_token_id],
            )
        )
        grammar = xgr.GrammarCompiler(tokenizer_info, max_threads=2).compile_structural_tag(
            config.structural_tag
        )
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        output = "<|close|>response<|sep|><|open|>tools<|sep|>"
        for index, city in enumerate(("A", "B", "C", "D"), 1):
            output += (
                f'<|open|>call tool="get_weather" index="{index}"<|sep|>'
                '<|open|>json type="object"<|sep|>'
                f'{{"city":"{city}"}}'
                "<|close|>json<|sep|><|close|>call<|sep|>"
            )
        output += "<|close|>tools<|sep|>"
        for token_id in self.tokenizer.encode(output):
            self.assertTrue(matcher.accept_token(token_id), f"masked token {token_id}")
        self.assertTrue(matcher.is_terminated())


if __name__ == "__main__":
    unittest.main()
