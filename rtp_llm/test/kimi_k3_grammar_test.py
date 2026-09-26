"""K3 model plans exercised through the native xgrammar matcher."""

import copy
import unittest
from types import SimpleNamespace

import xgrammar as xgr

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.models.kimi_k3.kimi_k3_grammar import (
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    THINK_TO_RESPONSE,
    build_kimi_k3_response_format_plan,
)

MESSAGE_CLOSE = "<|close|>message<|sep|>"


class K3Tokenizer:
    def __init__(self):
        self.vocab = [chr(i) for i in range(128)] + [
            "<|open|>",
            "<|close|>",
            "<|sep|>",
            "think",
            "response",
            "message",
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


class KimiK3GrammarTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = K3Tokenizer()
        self.compiler = xgr.GrammarCompiler(
            xgr.TokenizerInfo(self.tokenizer.vocab, stop_token_ids=[0]), max_threads=1
        )

    def prepare(self, thinking=True, budget=8, prompt=None, **kwargs):
        config = GenerateConfig(
            in_think_mode=thinking,
            thinking_mode=ThinkingMode.ENABLED if thinking else ThinkingMode.DISABLED,
            max_thinking_tokens=budget,
        )
        for field in (
            "json_schema",
            "regex",
            "ebnf",
            "structural_tag",
            "response_format",
        ):
            if field in kwargs:
                setattr(config, field, kwargs.pop(field))
        prompt = (
            prompt if prompt is not None else THINK_OPEN if thinking else RESPONSE_OPEN
        )
        plan = build_kimi_k3_response_format_plan(
            config,
            tokenizer=self.tokenizer,
            input_ids=self.tokenizer.encode(prompt),
            **kwargs
        )
        config.add_thinking_params(
            self.tokenizer,
            SimpleNamespace(
                think_mode="disabled",
                think_start_tag=THINK_OPEN,
                think_end_tag=THINK_TO_RESPONSE,
                think_end_token_id=-1,
            ),
            enable_thinking=thinking,
            prepared_plan=plan,
        )
        return config, xgr.GrammarMatcher(
            self.compiler.compile_structural_tag(config.structural_tag)
        )

    def allows(self, matcher, text):
        token_id = 0 if text == "EOS" else self.tokenizer.ids[text]
        mask = xgr.allocate_token_bitmask(1, len(self.tokenizer.vocab))
        matcher.fill_next_token_bitmask(mask)
        return bool(int(mask[0, token_id // 32]) & (1 << (token_id % 32)))

    def accept(self, matcher, text):
        for token_id in self.tokenizer.encode(text):
            self.assertTrue(matcher.accept_token(token_id), (text, token_id))

    def test_thinking_close_forces_full_transition_and_blocks_reported_loop(self):
        _, matcher = self.prepare()
        self.assertFalse(self.allows(matcher, "EOS"))
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, "x<|close|>")
        for text in ("response", "message", "<|open|>", "<|close|>"):
            self.assertFalse(self.allows(matcher, text))
        self.accept(matcher, "think<|sep|>")
        self.assertFalse(self.allows(matcher, "x"))
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, RESPONSE_OPEN + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))
        self.assertFalse(self.allows(matcher, "<|close|>"))

    def test_budget_zero_and_one_bound_only_thinking_body(self):
        for budget in (0, 1):
            with self.subTest(budget=budget):
                _, matcher = self.prepare(budget=budget)
                if budget:
                    self.accept(matcher, "x")
                self.assertFalse(self.allows(matcher, "x"))
                self.accept(
                    matcher, THINK_TO_RESPONSE + "xx" + RESPONSE_CLOSE + MESSAGE_CLOSE
                )
                self.assertTrue(self.allows(matcher, "EOS"))

    def test_disabled_thinking_still_requires_response_and_message_close(self):
        config, matcher = self.prepare(thinking=False)
        self.assertFalse(config.in_think_mode)
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, "a" + RESPONSE_CLOSE)
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_disabled_thinking_closes_an_upstream_open_think_block(self):
        config, matcher = self.prepare(thinking=False, prompt=THINK_OPEN)
        self.assertFalse(config.in_think_mode)
        self.assertFalse(self.allows(matcher, "x"))
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)

    def test_response_constraints_live_inside_the_response_block(self):
        for field, value, text in (
            (
                "json_schema",
                {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                },
                '{"q":"x"}',
            ),
            ("regex", "a", "a"),
            ("ebnf", 'root ::= "a"', "a"),
        ):
            with self.subTest(field=field):
                config, matcher = self.prepare(**{field: value})
                self.assertEqual(
                    config.structural_tag["format"]["elements"][1]["type"], "tag"
                )
                self.accept(
                    matcher, THINK_TO_RESPONSE + text + RESPONSE_CLOSE + MESSAGE_CLOSE
                )
                self.assertTrue(self.allows(matcher, "EOS"))

    def test_auto_xml_tools_remain_optional_and_constrain_arguments(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        _, matcher = self.prepare(tools=tools, parallel_tool_calls=False)
        self.accept(matcher, THINK_TO_RESPONSE + RESPONSE_CLOSE)
        self.accept(
            matcher,
            '<|open|>tools<|sep|><|open|>call tool="lookup" index="1"<|sep|>'
            '<|open|>argument key="q" type="string"<|sep|>x<|close|>argument<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|>" + MESSAGE_CLOSE,
        )
        self.assertTrue(self.allows(matcher, "EOS"))
        _, matcher = self.prepare(tools=tools)
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)

    def test_required_tools_keep_json_arguments_and_force_a_call(self):
        tail = {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": RESPONSE_CLOSE + "<|open|>tools<|sep|>",
                "content": {
                    "type": "tag",
                    "begin": '<|open|>call tool="lookup" index="1"<|sep|><|open|>json type="object"<|sep|>',
                    "content": {
                        "type": "json_schema",
                        "json_schema": {"type": "object"},
                    },
                    "end": "<|close|>json<|sep|><|close|>call<|sep|>",
                },
                "end": "<|close|>tools<|sep|>",
            },
        }
        _, matcher = self.prepare(structural_tag=tail, required_tools_tail=tail)
        self.accept(matcher, THINK_TO_RESPONSE)
        self.assertFalse(self.allows(matcher, "a"))
        self.accept(matcher, RESPONSE_CLOSE)
        self.accept(matcher, "<|open|>")
        self.assertFalse(self.allows(matcher, "message"))
        self.accept(
            matcher,
            'tools<|sep|><|open|>call tool="lookup" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|>" + MESSAGE_CLOSE,
        )

    def test_full_upstream_grammar_is_preserved_with_thinking_enabled(self):
        base, _ = self.prepare()
        original = copy.deepcopy(base.structural_tag)
        for scope in (None, "full_turn"):
            config, _ = self.prepare(structural_tag=original, scope=scope)
            self.assertTrue(config.in_think_mode)
            self.assertEqual(config.structural_tag, original)
        self.assertEqual(base.structural_tag, original)

    def test_response_tail_gets_one_thinking_transition(self):
        tail_config, _ = self.prepare(thinking=False)
        config, matcher = self.prepare(
            structural_tag=tail_config.structural_tag, scope="response_tail"
        )
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertEqual(
            config.structural_tag["format"]["elements"][0]["end"], THINK_CLOSE
        )

    def test_finalization_is_idempotent_and_plan_is_request_local(self):
        config, _ = self.prepare(budget=1)
        grammar = copy.deepcopy(config.structural_tag)
        config.finalize_response_format()
        self.assertEqual(config.structural_tag, grammar)
        other, _ = self.prepare(budget=7)
        self.assertEqual(
            config.structural_tag["format"]["elements"][0]["content"]["max_tokens"], 1
        )
        self.assertEqual(
            other.structural_tag["format"]["elements"][0]["content"]["max_tokens"], 7
        )
        self.assertNotIn("_model_response_format_plan", config.model_dump())

    def test_prompt_mismatch_and_unsupported_sampling_are_rejected(self):
        with self.assertRaisesRegex(FtRuntimeException, "prompt opens the response"):
            self.prepare(prompt=RESPONSE_OPEN)
        config = GenerateConfig(num_return_sequences=2)
        plan = build_kimi_k3_response_format_plan(config)
        with self.assertRaisesRegex(FtRuntimeException, "num_return_sequences"):
            config.finalize_response_format(prepared_plan=plan)


if __name__ == "__main__":
    unittest.main()
