"""K3 model plans exercised through the native xgrammar matcher."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import xgrammar as xgr

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.response_format_compiler import (
    ReasoningFormat,
    ResponseFormatPlan,
    validate_engine_ready,
)
from rtp_llm.config.thinking_mode import INT32_MAX
from rtp_llm.models.kimi_k3.kimi_k3_grammar import (
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    THINK_TO_RESPONSE,
    KimiK3Grammar,
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

    def __len__(self):
        return len(self.vocab)

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
        config.prepare_thinking_params(None, None)
        plan = KimiK3Grammar(**kwargs).build_plan(
            config, self.tokenizer, self.tokenizer.encode(prompt)
        )
        plan.apply_to_config(config)
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

    @staticmethod
    def token_marker(control, channel):
        return {
            "type": "sequence",
            "elements": [
                {"type": "token", "token": control},
                {"type": "const_string", "value": channel},
                {"type": "token", "token": "<|sep|>"},
            ],
        }

    def token_upstream_grammar(self, passthrough=False):
        # Serving's token builder represents channel markers as sequences,
        # rather than the string begin/end fields of the legacy tag builder.
        prefix = {
            "type": "sequence",
            "elements": [
                {
                    "type": "any_tokens",
                    "exclude_tokens": ["<|open|>", "<|close|>"],
                },
                self.token_marker("<|close|>", "think"),
            ],
        }
        suffix = (
            {"type": "any_text", "excludes": [THINK_OPEN, THINK_CLOSE]}
            if passthrough
            else {
                "type": "sequence",
                "elements": [
                    self.token_marker("<|open|>", "response"),
                    {
                        "type": "any_tokens",
                        "exclude_tokens": ["<|open|>", "<|close|>"],
                    },
                    self.token_marker("<|close|>", "response"),
                    self.token_marker("<|close|>", "message"),
                ],
            }
        )
        return {
            "type": "structural_tag",
            "format": {"type": "sequence", "elements": [prefix, suffix]},
        }

    def test_legacy_upstream_structural_tag_is_rejected_as_invalid_input(self):
        with self.assertRaisesRegex(FtRuntimeException, "requires a format object"):
            self.prepare(
                structural_tag={"structures": [], "triggers": []},
                upstream_grammar=True,
            )

    def test_token_upstream_keeps_one_think_region_and_caps_its_budget(self):
        for budget in (0, 1, 8):
            with self.subTest(budget=budget):
                upstream = self.token_upstream_grammar()
                original = copy.deepcopy(upstream)
                with patch(
                    "xgrammar.builtin_structural_tag.get_model_structural_tag",
                    side_effect=AssertionError(
                        "complete upstream grammar must be reused"
                    ),
                ):
                    config, matcher = self.prepare(
                        structural_tag=upstream, upstream_grammar=True, budget=budget
                    )
                expected = copy.deepcopy(original)
                expected["format"]["elements"][0]["elements"][0]["max_tokens"] = budget
                self.assertEqual(config.structural_tag, expected)
                self.assertEqual(upstream, original)
                self.assertFalse(self.allows(matcher, "<|open|>"))
                self.accept(matcher, "x" * budget)
                self.assertFalse(self.allows(matcher, "x"))
                self.accept(matcher, THINK_TO_RESPONSE + "answer")
                self.assertFalse(self.allows(matcher, "<|open|>"))
                self.accept(matcher, RESPONSE_CLOSE + MESSAGE_CLOSE)
                self.assertTrue(self.allows(matcher, "EOS"))

    def test_token_upstream_preserves_token_exclusion_semantics(self):
        upstream = self.token_upstream_grammar()
        upstream["format"]["elements"][0]["elements"][0]["max_tokens"] = 1
        config, matcher = self.prepare(
            structural_tag=upstream, upstream_grammar=True, budget=8
        )
        self.assertEqual(config.structural_tag, upstream)
        self.accept(matcher, "x")
        self.assertFalse(self.allows(matcher, "x"))
        self.accept(matcher, THINK_TO_RESPONSE)
        # Ordinary characters spelling a control marker remain text; only
        # the dedicated control token is reserved for the channel boundary.
        for character in "<|open|>":
            self.assertTrue(matcher.accept_token(self.tokenizer.ids[character]))
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_token_think_prefix_completes_an_unstructured_suffix(self):
        _, matcher = self.prepare(
            structural_tag=self.token_upstream_grammar(passthrough=True),
            upstream_grammar=True,
            budget=1,
        )
        self.accept(matcher, "x")
        self.assertFalse(self.allows(matcher, "x"))
        self.accept(matcher, THINK_TO_RESPONSE + "answer" + RESPONSE_CLOSE)
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_token_upstream_empty_and_optional_thinking_keep_their_structure(self):
        for empty in (False, True):
            with self.subTest(empty=empty):
                upstream = self.token_upstream_grammar()
                prefix = upstream["format"]["elements"][0]
                if empty:
                    prefix["elements"] = prefix["elements"][1:]
                else:
                    prefix["elements"].insert(0, self.token_marker("<|open|>", "think"))
                    upstream["format"]["elements"][0] = {
                        "type": "optional",
                        "content": prefix,
                    }
                config, matcher = self.prepare(
                    structural_tag=upstream, upstream_grammar=True, budget=1
                )
                if not empty:
                    self.accept(matcher, THINK_OPEN + "x")
                    self.assertFalse(self.allows(matcher, "x"))
                self.accept(
                    matcher, THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE
                )
                self.assertTrue(self.allows(matcher, "EOS"))
                if not empty:
                    no_think = xgr.GrammarMatcher(
                        self.compiler.compile_structural_tag(config.structural_tag)
                    )
                    self.accept(
                        no_think, RESPONSE_OPEN + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE
                    )
                    self.assertTrue(self.allows(no_think, "EOS"))

    def test_token_upstream_preserves_response_schema_and_tool_tail(self):
        upstream = self.token_upstream_grammar()
        response = upstream["format"]["elements"][1]["elements"]
        response[1] = {
            "type": "json_schema",
            "json_schema": {"type": "object", "additionalProperties": False},
            "style": "json",
        }
        tool_call = (
            '<|open|>tools<|sep|><|open|>call tool="lookup" index="1"<|sep|>'
            '<|open|>argument key="q" type="string"<|sep|>x'
            "<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        )
        response.insert(-1, {"type": "const_string", "value": tool_call})
        expected = copy.deepcopy(response)
        config, matcher = self.prepare(
            structural_tag=upstream, upstream_grammar=True, budget=1
        )
        self.assertEqual(
            config.structural_tag["format"]["elements"][1]["elements"], expected
        )
        self.accept(matcher, "x" + THINK_TO_RESPONSE + "{}" + RESPONSE_CLOSE)
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, tool_call + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_token_upstream_disabled_thinking_keeps_the_open_response(self):
        upstream = self.token_upstream_grammar()
        upstream["format"] = upstream["format"]["elements"][1]
        upstream["format"]["elements"].pop(0)
        config, matcher = self.prepare(
            thinking=False, structural_tag=upstream, upstream_grammar=True
        )
        self.assertEqual(config.structural_tag, upstream)
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, "answer" + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

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

    def test_negative_budget_keeps_k3_unbounded_thinking(self):
        config, matcher = self.prepare(budget=-1)
        self.assertEqual(config.max_thinking_tokens, INT32_MAX)
        self.accept(
            matcher, "xx" + THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE
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

    def test_upstream_grammar_is_request_local_and_budget_is_capped(self):
        base, _ = self.prepare()
        original = copy.deepcopy(base.structural_tag)
        config, matcher = self.prepare(
            structural_tag=original, upstream_grammar=True, budget=1
        )
        self.assertTrue(config.in_think_mode)
        self.accept(matcher, "x")
        self.assertFalse(self.allows(matcher, "x"))
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))
        self.assertEqual(base.structural_tag, original)
        config, _ = self.prepare(
            structural_tag=original, upstream_grammar=True, budget=64
        )
        self.assertEqual(
            config.structural_tag["format"]["elements"][0]["content"]["max_tokens"], 8
        )

    def test_upstream_response_without_thinking_uses_the_shared_think_prefix(self):
        response, _ = self.prepare(thinking=False)
        config, matcher = self.prepare(
            structural_tag=response.structural_tag, upstream_grammar=True, budget=1
        )
        self.assertEqual(len(config.structural_tag["format"]["elements"]), 3)
        self.accept(matcher, "x")
        self.assertFalse(self.allows(matcher, "x"))
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_model_preparation_reuses_upstream_without_common_compilation(self):
        base, _ = self.prepare()
        with patch.object(
            ResponseFormatPlan, "compile", wraps=ResponseFormatPlan.compile
        ) as compile_plan, patch(
            "xgrammar.builtin_structural_tag.get_model_structural_tag"
        ) as build_model:
            config, matcher = self.prepare(
                structural_tag=base.structural_tag, upstream_grammar=True, budget=1
            )
            original = copy.deepcopy(config.structural_tag)
            validate_engine_ready(config)
            self.assertEqual(config.structural_tag, original)
            compile_plan.assert_not_called()
            build_model.assert_not_called()
        self.accept(matcher, "x")
        self.assertFalse(self.allows(matcher, "x"))
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_model_preparation_handles_fixed_and_adaptive_thinking(self):
        for mode in (
            ThinkingMode.ENABLED,
            ThinkingMode.DISABLED,
            ThinkingMode.ADAPTIVE,
        ):
            with self.subTest(mode=mode):
                config = GenerateConfig(
                    thinking_mode=mode,
                    in_think_mode=mode == ThinkingMode.ENABLED,
                    max_thinking_tokens=1,
                    regex="a",
                )
                config.prepare_thinking_params(None, None)
                plan = KimiK3Grammar().build_plan(config, self.tokenizer)
                plan.apply_to_config(config)
                matcher = xgr.GrammarMatcher(
                    self.compiler.compile_structural_tag(config.structural_tag)
                )
                if mode != ThinkingMode.DISABLED:
                    if mode == ThinkingMode.ADAPTIVE:
                        self.accept(matcher, THINK_OPEN)
                    self.accept(matcher, "x")
                    self.assertFalse(self.allows(matcher, "x"))
                    self.assertFalse(self.allows(matcher, "<|open|>"))
                    self.accept(matcher, THINK_TO_RESPONSE)
                self.accept(matcher, "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)
                self.assertTrue(self.allows(matcher, "EOS"))
                if mode == ThinkingMode.ADAPTIVE:
                    no_think = xgr.GrammarMatcher(
                        self.compiler.compile_structural_tag(config.structural_tag)
                    )
                    self.accept(
                        no_think, RESPONSE_OPEN + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE
                    )
                    self.assertTrue(self.allows(no_think, "EOS"))

    def passthrough_grammar(self, prefix=None):
        return {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    prefix
                    or {
                        "type": "tag",
                        "begin": "",
                        "content": {"type": "any_text"},
                        "end": THINK_CLOSE,
                    },
                    {"type": "any_text", "excludes": [THINK_OPEN, THINK_CLOSE]},
                ],
            },
        }

    def test_upstream_passthrough_closes_channels_and_preserves_free_tools(self):
        for budget in (0, 1, 8):
            with self.subTest(budget=budget):
                _, matcher = self.prepare(
                    structural_tag=self.passthrough_grammar(),
                    upstream_grammar=True,
                    budget=budget,
                )
                self.assertFalse(self.allows(matcher, "EOS"))
                self.assertFalse(self.allows(matcher, "<|open|>"))
                for _ in range(budget):
                    self.accept(matcher, "x")
                self.assertFalse(self.allows(matcher, "x"))
                self.accept(matcher, "<|close|>")
                self.assertFalse(self.allows(matcher, "response"))
                self.accept(matcher, "think<|sep|>")
                self.assertFalse(self.allows(matcher, "x"))
                self.accept(matcher, RESPONSE_OPEN + "answer")
                self.assertFalse(self.allows(matcher, "<|open|>"))
                self.accept(matcher, "<|close|>")
                self.assertFalse(self.allows(matcher, "think"))
                self.assertFalse(self.allows(matcher, "message"))
                self.accept(matcher, "response<|sep|>")
                self.assertFalse(self.allows(matcher, "EOS"))
                self.accept(matcher, "<|open|>")
                self.assertFalse(self.allows(matcher, "think"))
                self.assertFalse(self.allows(matcher, "response"))
                self.accept(
                    matcher,
                    'tools<|sep|><|open|>call tool="lookup" index="1"<|sep|>'
                    '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
                    "<|close|>call<|sep|><|close|>tools<|sep|>" + MESSAGE_CLOSE,
                )
                self.assertTrue(self.allows(matcher, "EOS"))
                self.assertFalse(self.allows(matcher, "<|close|>"))

    def test_upstream_empty_optional_and_combined_think_boundaries(self):
        prefixes = [
            {"type": "const_string", "value": THINK_CLOSE},
            {
                "type": "optional",
                "content": self.passthrough_grammar()["format"]["elements"][0],
            },
            {
                "type": "tag",
                "begin": "",
                "content": {"type": "any_text"},
                "end": THINK_TO_RESPONSE,
            },
        ]
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                _, matcher = self.prepare(
                    structural_tag=self.passthrough_grammar(prefix),
                    upstream_grammar=True,
                )
                self.accept(matcher, THINK_TO_RESPONSE + RESPONSE_CLOSE + MESSAGE_CLOSE)
                self.assertTrue(self.allows(matcher, "EOS"))

    def test_upstream_json_response_excludes_controls_without_editing_schema(self):
        schema = {"type": "object", "properties": {"type": {"type": "string"}}}
        base, _ = self.prepare(json_schema=schema)
        original = copy.deepcopy(base.structural_tag)
        original["format"]["elements"][0]["content"].pop("excludes")
        original["format"]["elements"][1]["content"].pop("excludes")
        config, matcher = self.prepare(structural_tag=original, upstream_grammar=True)
        self.assertEqual(
            config.structural_tag["format"]["elements"][1]["content"]["json_schema"],
            schema,
        )
        self.accept(matcher, THINK_TO_RESPONSE + '{"type":"')
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.assertFalse(self.allows(matcher, "<|close|>"))
        self.accept(matcher, 'x"}' + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_upstream_masks_recover_after_speculative_rollback(self):
        _, matcher = self.prepare(
            structural_tag=self.passthrough_grammar(), upstream_grammar=True, budget=1
        )
        self.accept(matcher, "x")
        self.assertFalse(self.allows(matcher, "x"))
        self.accept(matcher, THINK_TO_RESPONSE)
        matcher.rollback(len(self.tokenizer.encode(THINK_TO_RESPONSE)))
        self.assertFalse(self.allows(matcher, "x"))
        self.assertFalse(self.allows(matcher, "EOS"))
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_prepared_grammar_is_engine_ready_and_request_local(self):
        config, _ = self.prepare(budget=1)
        grammar = copy.deepcopy(config.structural_tag)
        validate_engine_ready(config)
        self.assertEqual(config.structural_tag, grammar)
        other, _ = self.prepare(budget=7)
        self.assertEqual(
            config.structural_tag["format"]["elements"][0]["content"]["max_tokens"], 1
        )
        self.assertEqual(
            other.structural_tag["format"]["elements"][0]["content"]["max_tokens"], 7
        )
        self.assertNotIn("_grammar_format", config.model_dump())

    def test_prompt_mismatch_and_unsupported_sampling_are_rejected(self):
        with self.assertRaisesRegex(FtRuntimeException, "prompt opens the response"):
            self.prepare(prompt=RESPONSE_OPEN)
        config = GenerateConfig(num_return_sequences=2)
        with self.assertRaisesRegex(FtRuntimeException, "num_return_sequences"):
            config.prepare_thinking_params(None, False)
            KimiK3Grammar().build_plan(config, self.tokenizer)

    def test_model_plan_leaves_grammar_fields_for_common_finalization(self):
        from rtp_llm.config.response_format import ResponseFormat

        config = GenerateConfig(
            thinking_mode=ThinkingMode.ENABLED,
            in_think_mode=True,
            response_format=ResponseFormat(type="json_object"),
        )
        plan = KimiK3Grammar().build_plan(config, self.tokenizer)
        self.assertEqual(config.response_format.type, "json_object")
        self.assertIsNone(config.structural_tag)
        self.assertFalse(config._reasoning_envelope_applied)
        plan.apply_to_config(config)
        self.assertIsNone(config.response_format)
        self.assertEqual(config.structural_tag, plan.engine_constraint.value)
        config.finalize_response_format()
        self.assertEqual(config.structural_tag, plan.engine_constraint.value)

    def test_openai_entry_prepares_model_grammar_once(self):
        from rtp_llm.openai.api_datatype import ChatCompletionRequest
        from rtp_llm.openai.openai_endpoint import OpenaiEndpoint

        prepared = []

        def build_plan(request, config, input_ids):
            plan = KimiK3Grammar().build_plan(config, self.tokenizer, input_ids)
            prepared.append(plan)
            return plan

        renderer = SimpleNamespace(
            resolve_thinking_mode=lambda request: ThinkingMode.ENABLED,
            tokenize_words=lambda words: [],
            apply_chat_completion_constraints=lambda request, config: None,
            get_reasoning_format=lambda: ReasoningFormat("", THINK_CLOSE),
            build_chat_completion_response_format_plan=build_plan,
        )
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.chat_renderer = renderer
        endpoint.tokenizer = self.tokenizer
        endpoint.stop_words_str_list = []
        endpoint.stop_words_id_list = []
        endpoint.generate_env_config = None
        request = ChatCompletionRequest(
            messages=[], response_format={"type": "json_object"}
        )
        with patch.object(
            GenerateConfig,
            "prepare_thinking_params",
            autospec=True,
            side_effect=GenerateConfig.prepare_thinking_params,
        ) as prepare_params, patch.object(
            GenerateConfig,
            "validate",
            autospec=True,
            side_effect=GenerateConfig.validate,
        ) as validate, patch.object(
            ResponseFormatPlan,
            "apply_to_config",
            autospec=True,
            side_effect=ResponseFormatPlan.apply_to_config,
        ) as apply_plan:
            config = endpoint._extract_generation_config(
                request, self.tokenizer.encode(THINK_OPEN), renderer
            )
        prepare_params.assert_called_once()
        validate.assert_called_once()
        apply_plan.assert_called_once()
        self.assertEqual(len(prepared), 1)
        self.assertEqual(config.structural_tag, prepared[0].engine_constraint.value)
        matcher = xgr.GrammarMatcher(
            self.compiler.compile_structural_tag(config.structural_tag)
        )
        self.accept(matcher, THINK_TO_RESPONSE + "{}" + RESPONSE_CLOSE + MESSAGE_CLOSE)
        self.assertTrue(self.allows(matcher, "EOS"))

    def test_openai_partial_continues_existing_channel_without_open_marker(self):
        from rtp_llm.openai.api_datatype import (
            ChatCompletionRequest,
            ChatCompletionTokenLogprob,
        )
        from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
        from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
        from rtp_llm.openai.renderers.kimi_k3_renderer import (
            K3XtmlDecoder,
            KimiK3Renderer,
        )

        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                opening = THINK_OPEN if thinking else RESPONSE_OPEN
                renderer = KimiK3Renderer.__new__(KimiK3Renderer)
                renderer.tokenizer = self.tokenizer
                endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
                endpoint.chat_renderer = renderer
                endpoint.tokenizer = self.tokenizer
                endpoint.model_type = "kimi_k3"
                endpoint.stop_words_str_list = []
                endpoint.stop_words_id_list = []
                endpoint.generate_env_config = None
                request = ChatCompletionRequest(
                    messages=[
                        {"role": "assistant", "content": "prefix", "partial": True}
                    ],
                    enable_thinking=thinking,
                    logprobs=True,
                )
                with patch.object(
                    renderer,
                    "render_chat",
                    return_value=RenderedInputs(self.tokenizer.encode(opening)),
                ):
                    rendered = endpoint.render_chat(request)
                self.assertEqual(
                    rendered.input_ids, self.tokenizer.encode(opening + "prefix")
                )
                config = endpoint._extract_generation_config(
                    request, rendered.input_ids, renderer
                )
                matcher = xgr.GrammarMatcher(
                    self.compiler.compile_structural_tag(config.structural_tag)
                )
                # The generated continuation starts with text in the channel
                # already opened before the assistant prefix.
                self.accept(matcher, "tail")
                self.accept(
                    matcher,
                    (THINK_TO_RESPONSE if thinking else "")
                    + RESPONSE_CLOSE
                    + MESSAGE_CLOSE,
                )
                self.assertTrue(self.allows(matcher, "EOS"))
                parsed = K3XtmlDecoder(thinking, request).decode(
                    "tail",
                    [
                        ChatCompletionTokenLogprob(
                            token=token,
                            bytes=[ord(token)],
                            logprob=-0.1,
                            top_logprobs=[],
                        )
                        for token in "tail"
                    ],
                    flush=True,
                )
                channel = "reasoning_content" if thinking else "content"
                self.assertEqual(getattr(parsed.delta, channel), "tail")
                self.assertEqual(
                    bytes(
                        byte
                        for token in parsed.channel_logprobs[channel]
                        for byte in token.bytes
                    ),
                    b"tail",
                )

    def test_model_plan_installation_does_not_rebuild_grammar(self):
        config = GenerateConfig(
            thinking_mode=ThinkingMode.ENABLED, in_think_mode=True, regex="a"
        )
        plan = KimiK3Grammar().build_plan(config, self.tokenizer)
        with patch(
            "xgrammar.builtin_structural_tag.get_model_structural_tag",
            side_effect=AssertionError("installed grammar must not be rebuilt"),
        ):
            plan.apply_to_config(config)
            plan.apply_to_config(config)
        self.assertEqual(config.structural_tag, plan.engine_constraint.value)
        self.assertEqual(plan.final_constraint.name, "regex")


if __name__ == "__main__":
    unittest.main()
