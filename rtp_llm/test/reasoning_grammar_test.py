"""Thinking boundaries through request preparation and native xgrammar masks."""

import copy
import unittest
from types import SimpleNamespace

import xgrammar as xgr

from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.response_format_compiler import ReasoningFormat


class BoundaryTokenizer:
    """Small RAW vocabulary with distinct K3 and DeepSeek control tokens."""

    def __init__(self):
        self.vocab = [chr(i) for i in range(128)] + [
            "<|open|>",
            "<|close|>",
            "<|sep|>",
            "think",
            "response",
            "message",
            "<think>",
            "</think>",
        ]
        self.ids = {text: i for i, text in enumerate(self.vocab)}
        self.pieces = sorted(self.ids, key=len, reverse=True)

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            piece = next(piece for piece in self.pieces if text.startswith(piece))
            result.append(self.ids[piece])
            text = text[len(piece) :]
        return result

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.vocab[token_id] for token_id in token_ids)


class ReasoningGrammarTest(unittest.TestCase):
    K3_BEGIN = "<|open|>think<|sep|>"
    K3_END = "<|close|>think<|sep|><|open|>response<|sep|>"

    def setUp(self):
        self.tokenizer = BoundaryTokenizer()
        self.compiler = xgr.GrammarCompiler(
            xgr.TokenizerInfo(self.tokenizer.vocab, stop_token_ids=[0]),
            max_threads=1,
        )

    def config(
        self,
        budget=64,
        mode=ThinkingMode.ENABLED,
        begin=K3_BEGIN,
        end=K3_END,
        end_token_id=-1,
        reasoning_format=None,
    ):
        config = GenerateConfig(
            max_thinking_tokens=budget, thinking_mode=mode, regex="a"
        )
        config.add_thinking_params(
            self.tokenizer,
            SimpleNamespace(
                think_mode="disabled",
                think_start_tag=begin,
                think_end_tag=end,
                think_end_token_id=end_token_id,
            ),
            reasoning_format=reasoning_format,
        )
        return config

    def matcher(self, config):
        return xgr.GrammarMatcher(
            self.compiler.compile_structural_tag(config.structural_tag)
        )

    def allows(self, matcher, text):
        token_id = 0 if text == "EOS" else self.tokenizer.ids[text]
        bitmask = xgr.allocate_token_bitmask(1, len(self.tokenizer.vocab))
        matcher.fill_next_token_bitmask(bitmask)
        return bool(int(bitmask[0, token_id // 32]) & (1 << (token_id % 32)))

    def accept(self, matcher, text):
        for token_id in self.tokenizer.encode(text):
            self.assertTrue(matcher.accept_token(token_id))

    def test_k3_natural_close_commits_to_think_end_before_budget(self):
        matcher = self.matcher(self.config())
        self.assertTrue(self.allows(matcher, "x"))
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, "x<|close|>")
        for invalid in ("response", "message", "<|open|>", "<|close|>", "x", "EOS"):
            self.assertFalse(self.allows(matcher, invalid), invalid)
        self.accept(matcher, "think<|sep|><|open|>response<|sep|>")
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, "a")
        self.assertTrue(self.allows(matcher, "EOS"))
        self.assertTrue(matcher.accept_token(0))
        self.assertTrue(matcher.is_terminated())

    def test_old_any_text_allows_wrong_close_but_new_grammar_rejects_it(self):
        config = self.config()
        old_config = copy.deepcopy(config)
        old_config.structural_tag["format"]["elements"][0]["content"].pop("excludes")
        old_matcher = self.matcher(old_config)
        self.accept(old_matcher, "<|close|>")
        self.assertTrue(self.allows(old_matcher, "response"))
        self.assertTrue(self.allows(old_matcher, "message"))
        new_matcher = self.matcher(config)
        self.accept(new_matcher, "<|close|>")
        self.assertFalse(self.allows(new_matcher, "response"))
        self.assertFalse(self.allows(new_matcher, "message"))

    def test_k3_budget_still_forces_complete_transition(self):
        matcher = self.matcher(self.config(budget=1))
        self.accept(matcher, "x")
        for token_id in self.tokenizer.encode(self.K3_END):
            self.assertFalse(self.allows(matcher, "x"))
            self.assertFalse(self.allows(matcher, "EOS"))
            self.assertTrue(matcher.accept_token(token_id))
        self.accept(matcher, "a")
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_provisional_boundary_accept_and_rollback_restore_masks(self):
        matcher = self.matcher(self.config())
        for _ in range(3):
            self.accept(matcher, "<|close|>think")
            self.assertTrue(self.allows(matcher, "<|sep|>"))
            self.assertFalse(self.allows(matcher, "response"))
            matcher.rollback(2)
            self.assertTrue(self.allows(matcher, "x"))
            self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, self.K3_END + "a")
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_deepseek_end_token_requires_configured_newline_suffix(self):
        matcher = self.matcher(self.config(begin="<think>", end="</think>\n\n"))
        self.assertFalse(self.allows(matcher, "<think>"))
        self.accept(matcher, "x</think>")
        self.assertFalse(self.allows(matcher, "x"))
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, "\n")
        self.assertFalse(self.allows(matcher, "a"))
        self.accept(matcher, "\na")
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_token_id_boundary_keeps_precedence_over_end_string(self):
        matcher = self.matcher(
            self.config(
                begin="<think>",
                end="unused string",
                end_token_id=self.tokenizer.ids["</think>"],
            )
        )
        self.assertTrue(self.allows(matcher, "u"))
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, "</think>a")
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_adaptive_and_custom_reasoning_use_same_body_exclusions(self):
        custom = ReasoningFormat(
            tag_begin=self.K3_BEGIN,
            tag_end=self.K3_END,
            content_excludes=("message",),
        )
        matcher = self.matcher(
            self.config(mode=ThinkingMode.ADAPTIVE, reasoning_format=custom)
        )
        self.accept(matcher, self.K3_BEGIN)
        self.assertFalse(self.allows(matcher, "message"))
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, "<|close|>")
        self.assertFalse(self.allows(matcher, "response"))
        self.accept(matcher, "think<|sep|><|open|>response<|sep|>a")
        self.assertTrue(self.allows(matcher, "EOS"))
        no_think = self.matcher(self.config(mode=ThinkingMode.ADAPTIVE))
        self.accept(no_think, "a")
        self.assertTrue(self.allows(no_think, "EOS"))

    def test_disabled_thinking_preserves_final_constraint(self):
        config = self.config(mode=ThinkingMode.DISABLED)
        self.assertIsNone(config.structural_tag)
        self.assertEqual(config.regex, "a")

    def test_complete_upstream_grammar_keeps_its_shape_and_gets_thinking_limits(self):
        upstream = copy.deepcopy(self.config().structural_tag)
        config = GenerateConfig(
            thinking_mode=ThinkingMode.ENABLED,
            structural_tag=upstream,
            max_thinking_tokens=1,
        )
        config._structural_tag_is_complete = True
        config.add_thinking_params(
            self.tokenizer,
            SimpleNamespace(
                think_mode="disabled",
                think_start_tag=self.K3_BEGIN,
                think_end_tag=self.K3_END,
                think_end_token_id=-1,
            ),
        )
        self.assertEqual(len(config.structural_tag["format"]["elements"]), 2)
        self.assertEqual(upstream["format"]["elements"][0]["content"]["max_tokens"], 64)
        matcher = self.matcher(config)
        self.accept(matcher, "x")
        self.assertFalse(self.allows(matcher, "x"))
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, self.K3_END + "a")
        self.assertTrue(self.allows(matcher, "EOS"))


if __name__ == "__main__":
    unittest.main()
