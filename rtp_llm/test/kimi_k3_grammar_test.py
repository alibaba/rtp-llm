"""K3 model plans exercised through the native xgrammar matcher."""

import copy
import unittest
from unittest.mock import patch

import xgrammar as xgr

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.response_format import ResponseFormat
from rtp_llm.config.response_format_compiler import (
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
RESPONSE_END = RESPONSE_CLOSE + MESSAGE_CLOSE
TOOLS_OPEN = "<|open|>tools<|sep|>"
TOOLS_CLOSE = "<|close|>tools<|sep|>"


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

    def convert_tokens_to_ids(self, text):
        return self.ids.get(text)

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

    def assert_completion_accepted(self, matcher, text):
        self.accept(matcher, text)
        self.assertTrue(self.allows(matcher, "EOS"), text)

    def assert_completion_rejected(self, matcher, text):
        for token_id in self.tokenizer.encode(text):
            if not matcher.accept_token(token_id):
                return
        self.assertFalse(self.allows(matcher, "EOS"), text)

    @staticmethod
    def lookup_tools():
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string", "enum": ["x"]}},
                        "required": ["q"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    @staticmethod
    def xml_call(name="lookup", key="q", value="x", index=1, value_type="string"):
        return (
            f'<|open|>call tool="{name}" index="{index}"<|sep|>'
            f'<|open|>argument key="{key}" type="{value_type}"<|sep|>{value}'
            "<|close|>argument<|sep|><|close|>call<|sep|>"
        )

    @staticmethod
    def required_tool_tail(schema):
        return {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": RESPONSE_CLOSE + TOOLS_OPEN,
                "content": {
                    "type": "tag",
                    "begin": '<|open|>call tool="lookup" index="1"<|sep|><|open|>json type="object"<|sep|>',
                    "content": {"type": "json_schema", "json_schema": schema},
                    "end": "<|close|>json<|sep|><|close|>call<|sep|>",
                },
                "end": TOOLS_CLOSE,
            },
        }

    def test_all_public_response_formats_accept_and_reject_body_content(self):
        # Both registered K3 versions share this grammar class. Exercise each
        # public format through native masks, with and without a think channel.
        cases = (
            (None, "a", None),
            ({"type": "text"}, "a", None),
            ({"type": "json_object"}, "{}", '"x"'),
            ({"type": "json_object"}, "[]", "1"),
            (
                {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": {"enum": ["a"]}},
                },
                '"a"',
                '"b"',
            ),
            ({"type": "regex", "pattern": "a|b"}, "b", "c"),
            ({"type": "ebnf", "grammar": 'root ::= "a"'}, "a", "b"),
            (
                {
                    "type": "structural_tag",
                    "structural_tag": {
                        "type": "structural_tag",
                        "format": {
                            "type": "tag",
                            "begin": "[",
                            "content": {"type": "regex", "pattern": "a"},
                            "end": "]",
                        },
                    },
                },
                "[a]",
                "[b]",
            ),
        )
        for thinking in (False, True):
            prefix = THINK_TO_RESPONSE if thinking else ""
            for response_format, valid, invalid in cases:
                with self.subTest(thinking=thinking, response_format=response_format):
                    kwargs = (
                        {
                            "response_format": ResponseFormat.model_validate(
                                response_format
                            )
                        }
                        if response_format is not None
                        else {}
                    )
                    config, matcher = self.prepare(thinking=thinking, **kwargs)
                    self.assertIsNone(config.response_format)
                    self.assertIsNotNone(config.structural_tag)
                    self.assert_completion_accepted(
                        matcher, prefix + valid + RESPONSE_END
                    )
                    if invalid is not None:
                        _, matcher = self.prepare(thinking=thinking, **kwargs)
                        self.assert_completion_rejected(
                            matcher, prefix + invalid + RESPONSE_END
                        )

    def test_plain_text_allows_empty_body_but_requires_protocol_termination(self):
        for thinking in (False, True):
            for response_format in (None, ResponseFormat(type="text")):
                with self.subTest(thinking=thinking, response_format=response_format):
                    _, matcher = self.prepare(
                        thinking=thinking, response_format=response_format
                    )
                    if thinking:
                        self.accept(matcher, THINK_TO_RESPONSE)
                    self.assertFalse(self.allows(matcher, "EOS"))
                    self.assert_completion_accepted(
                        matcher, RESPONSE_CLOSE + MESSAGE_CLOSE
                    )

    def test_auto_tools_accept_xml_and_preserve_each_response_format(self):
        for thinking in (False, True):
            for response_format, body in (
                (ResponseFormat(type="text"), "a"),
                (ResponseFormat(type="json_object"), "{}"),
            ):
                with self.subTest(
                    thinking=thinking, response_format=response_format.type
                ):
                    _, matcher = self.prepare(
                        thinking=thinking,
                        response_format=response_format,
                        tools=self.lookup_tools(),
                    )
                    prefix = THINK_TO_RESPONSE if thinking else ""
                    self.assert_completion_accepted(
                        matcher,
                        prefix
                        + body
                        + RESPONSE_CLOSE
                        + TOOLS_OPEN
                        + self.xml_call()
                        + TOOLS_CLOSE
                        + MESSAGE_CLOSE,
                    )

    def test_auto_tool_schema_rejects_unknown_names_keys_types_and_values(self):
        invalid_calls = (
            self.xml_call(name="unknown"),
            self.xml_call(key="unknown"),
            self.xml_call(value="y"),
            self.xml_call(value="1", value_type="number"),
            '<|open|>call tool="lookup" index="1"<|sep|><|close|>call<|sep|>',
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                _, matcher = self.prepare(thinking=False, tools=self.lookup_tools())
                self.assert_completion_rejected(
                    matcher,
                    RESPONSE_CLOSE + TOOLS_OPEN + call + TOOLS_CLOSE + MESSAGE_CLOSE,
                )

    def test_no_tools_cannot_open_tools_block(self):
        for tools in (None, []):
            with self.subTest(tools=tools):
                _, matcher = self.prepare(thinking=False, tools=tools)
                self.assert_completion_rejected(
                    matcher,
                    RESPONSE_CLOSE
                    + TOOLS_OPEN
                    + self.xml_call()
                    + TOOLS_CLOSE
                    + MESSAGE_CLOSE,
                )

    def test_auto_parallel_tools_controls_second_call(self):
        text = (
            RESPONSE_CLOSE
            + TOOLS_OPEN
            + self.xml_call(index=1)
            + self.xml_call(index=2)
            + TOOLS_CLOSE
            + MESSAGE_CLOSE
        )
        _, parallel = self.prepare(
            thinking=False, tools=self.lookup_tools(), parallel_tool_calls=True
        )
        self.assert_completion_accepted(parallel, text)
        _, single = self.prepare(
            thinking=False, tools=self.lookup_tools(), parallel_tool_calls=False
        )
        self.assert_completion_rejected(single, text)

    def test_prompt_without_open_channel_requires_a_new_channel(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                _, matcher = self.prepare(thinking=thinking, prompt="history")
                self.assertFalse(self.allows(matcher, "x"))
                opening = THINK_OPEN if thinking else RESPONSE_OPEN
                transition = THINK_TO_RESPONSE if thinking else ""
                self.assert_completion_accepted(
                    matcher,
                    opening + transition + "a" + RESPONSE_END,
                )

    def test_partial_open_channels_continue_body_without_reopening(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                opening = THINK_OPEN if thinking else RESPONSE_OPEN
                _, matcher = self.prepare(
                    thinking=thinking, prompt="history" + opening + "prefix"
                )
                self.assertFalse(self.allows(matcher, "<|open|>"))
                transition = THINK_TO_RESPONSE if thinking else ""
                self.assert_completion_accepted(
                    matcher, "x" + transition + "a" + RESPONSE_END
                )

    def test_closed_history_does_not_reuse_its_last_think_channel(self):
        history = THINK_OPEN + "x" + THINK_TO_RESPONSE + "a" + RESPONSE_END
        _, matcher = self.prepare(thinking=False, prompt=history)
        self.assertFalse(self.allows(matcher, "x"))
        self.assert_completion_accepted(matcher, RESPONSE_OPEN + "a" + RESPONSE_END)

    def test_control_markers_cannot_escape_plain_response_or_json_string(self):
        for kwargs, body_prefix in (
            ({}, "a"),
            ({"response_format": ResponseFormat(type="json_object")}, '{"q":"a'),
        ):
            with self.subTest(kwargs=kwargs):
                config, matcher = self.prepare(thinking=False, **kwargs)
                self.assertEqual(config.end_think_token_ids, [])
                self.accept(matcher, body_prefix)
                self.assertFalse(self.allows(matcher, "<|open|>"))
                if body_prefix.startswith("{"):
                    self.assertFalse(self.allows(matcher, "<|close|>"))
                else:
                    self.accept(matcher, "<|close|>")
                    self.assertFalse(self.allows(matcher, "think"))
                    self.assertFalse(self.allows(matcher, "message"))

    def test_control_markers_cannot_escape_nested_structural_tag_text(self):
        response_format = ResponseFormat(
            type="structural_tag",
            structural_tag={
                "type": "structural_tag",
                "format": {
                    "type": "sequence",
                    "elements": [
                        {"type": "const_string", "value": "result:"},
                        {
                            "type": "tag",
                            "begin": "[",
                            "content": {"type": "any_text"},
                            "end": "]",
                        },
                    ],
                },
            },
        )
        _, matcher = self.prepare(thinking=False, response_format=response_format)
        self.accept(matcher, "result:[a")
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.assertFalse(self.allows(matcher, "<|close|>"))
        self.assert_completion_accepted(matcher, "]" + RESPONSE_END)

    def test_control_markers_are_excluded_in_tag_lists(self):
        tag = {
            "type": "tag",
            "begin": "[field]",
            "end": "[/field]",
            "content": {"type": "any_text"},
        }
        for kind, fields in (
            ("triggered_tags", {"triggers": ["[field]"]}),
            ("tags_with_separator", {"separator": ","}),
        ):
            with self.subTest(kind=kind):
                config, matcher = self.prepare(
                    thinking=False,
                    response_format=ResponseFormat(
                        type="structural_tag",
                        structural_tag={
                            "type": "structural_tag",
                            "format": {"type": kind, "tags": [tag], **fields},
                        },
                    ),
                )
                if kind == "triggered_tags":
                    self.assertFalse(self.allows(matcher, "<|open|>"))
                self.assertEqual(config.end_think_token_ids, [])
                self.accept(matcher, "[field]a")
                self.assertFalse(self.allows(matcher, "<|open|>"))
                self.assertFalse(self.allows(matcher, "<|close|>"))
                self.assert_completion_accepted(matcher, "[/field]" + RESPONSE_END)

    def test_auto_tool_string_cannot_inject_channels(self):
        tools = self.lookup_tools()
        del tools[0]["function"]["parameters"]["properties"]["q"]["enum"]
        _, matcher = self.prepare(thinking=False, tools=tools)
        self.accept(
            matcher,
            RESPONSE_CLOSE
            + '<|open|>tools<|sep|><|open|>call tool="lookup" index="1"<|sep|>'
            + '<|open|>argument key="q" type="string"<|sep|>a',
        )
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.assertFalse(self.allows(matcher, "<|sep|>"))
        self.accept(matcher, "<|close|>")
        self.assertFalse(self.allows(matcher, "think"))
        self.assertFalse(self.allows(matcher, "message"))
        self.assert_completion_accepted(
            matcher,
            "argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>" + MESSAGE_CLOSE,
        )

    def test_beam_and_multiple_return_sequences_are_rejected_without_thinking(self):
        for kwargs in (
            {"num_beams": 2},
            {"variable_num_beams": [1, 2]},
            {"num_return_sequences": 2},
        ):
            with self.subTest(kwargs=kwargs):
                config = GenerateConfig(
                    in_think_mode=False, thinking_mode=ThinkingMode.DISABLED, **kwargs
                )
                with self.assertRaisesRegex(FtRuntimeException, "beam search"):
                    KimiK3Grammar().build_plan(
                        config, self.tokenizer, self.tokenizer.encode(RESPONSE_OPEN)
                    )

    def test_builder_failure_is_reported_as_invalid_request(self):
        # Malformed public tools are rejected by the OpenAI request model before
        # reaching this class. A dependency's construction error must still be
        # converted into a request error rather than an internal server error.
        with patch(
            "xgrammar.builtin_structural_tag.get_model_structural_tag",
            side_effect=ValueError("unsupported tool schema"),
        ):
            with self.assertRaisesRegex(
                FtRuntimeException,
                "invalid Kimi K3 tool schema: unsupported tool schema",
            ) as error:
                self.prepare(thinking=False, tools=self.lookup_tools())
        self.assertEqual(error.exception.exception_type, ExceptionType.INVALID_PARAMS)

    def test_required_tool_tail_constrains_json_and_cannot_be_skipped(self):
        schema = self.lookup_tools()[0]["function"]["parameters"]
        tail = self.required_tool_tail(schema)
        for thinking in (False, True):
            prefix = THINK_TO_RESPONSE if thinking else ""
            for arguments in ('{"q":"x"}', "{}", '{"q":"y"}', '{"q":1}'):
                with self.subTest(thinking=thinking, arguments=arguments):
                    _, matcher = self.prepare(
                        thinking=thinking, structural_tag=tail, required_tools_tail=tail
                    )
                    text = (
                        prefix
                        + RESPONSE_CLOSE
                        + '<|open|>tools<|sep|><|open|>call tool="lookup" index="1"<|sep|>'
                        + '<|open|>json type="object"<|sep|>'
                        + arguments
                        + "<|close|>json<|sep|><|close|>call<|sep|>"
                        + TOOLS_CLOSE
                        + MESSAGE_CLOSE
                    )
                    if arguments == '{"q":"x"}':
                        self.assert_completion_accepted(matcher, text)
                    else:
                        self.assert_completion_rejected(matcher, text)
            for invalid in (
                prefix + "a" + RESPONSE_END,
                prefix + RESPONSE_END,
            ):
                with self.subTest(thinking=thinking, invalid=invalid):
                    _, matcher = self.prepare(
                        thinking=thinking, structural_tag=tail, required_tools_tail=tail
                    )
                    self.assert_completion_rejected(matcher, invalid)

    def test_conflicting_response_constraints_are_rejected_before_sampling(self):
        for constraints in (
            {"response_format": ResponseFormat(type="json_object"), "regex": "a"},
            {"regex": "a", "ebnf": 'root ::= "a"'},
        ):
            with self.subTest(constraints=constraints):
                with self.assertRaises(FtRuntimeException):
                    self.prepare(thinking=False, **constraints)

    def test_nested_bounded_structural_tag_is_rejected(self):
        response_format = ResponseFormat(
            type="structural_tag",
            structural_tag={
                "type": "structural_tag",
                "format": {"type": "any_text", "max_tokens": 1},
            },
        )
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                with self.assertRaisesRegex(FtRuntimeException, "already contains"):
                    self.prepare(thinking=thinking, response_format=response_format)

    def test_structural_tag_token_and_alternate_endings_keep_response_boundaries(self):
        for ending, body in (
            ({"type": "token", "token": "]"}, "[a]"),
            ({"type": "token", "token": self.tokenizer.ids["]"]}, "[a]"),
            (["]", ">"], "[a>"),
        ):
            with self.subTest(ending=ending):
                response_format = ResponseFormat(
                    type="structural_tag",
                    structural_tag={
                        "type": "structural_tag",
                        "format": {
                            "type": "tag",
                            "begin": "[",
                            "content": {"type": "regex", "pattern": "a"},
                            "end": ending,
                        },
                    },
                )
                _, matcher = self.prepare(
                    thinking=False, response_format=response_format
                )
                self.accept(matcher, body)
                self.assertFalse(self.allows(matcher, "EOS"))
                self.assert_completion_accepted(matcher, RESPONSE_CLOSE + MESSAGE_CLOSE)

    def test_structural_tag_free_text_suffix_is_not_a_thinking_budget(self):
        for suffix, body in (
            (
                [
                    {
                        "type": "optional",
                        "content": {"type": "const_string", "value": "!"},
                    },
                    {"type": "const_string", "value": "]"},
                ],
                "[xxxx!]",
            ),
            ([{"type": "regex", "pattern": "b"}], "[xxxxb"),
        ):
            with self.subTest(suffix=suffix):
                response_format = ResponseFormat(
                    type="structural_tag",
                    structural_tag={
                        "type": "structural_tag",
                        "format": {
                            "type": "sequence",
                            "elements": [
                                {"type": "const_string", "value": "["},
                                {"type": "any_text"},
                                *suffix,
                            ],
                        },
                    },
                )
                # Disabled thinking installs a zero think budget. A caller's
                # free response region must remain free to produce text.
                _, matcher = self.prepare(
                    thinking=False, response_format=response_format
                )
                self.assert_completion_accepted(matcher, body + RESPONSE_END)

    def test_malformed_structural_tag_regions_are_invalid_requests(self):
        formats = (
            {"type": "sequence"},
            {"type": "sequence", "elements": []},
            {"type": "sequence", "elements": [None]},
            {"type": "sequence", "elements": [{"type": "tag", "content": 1}]},
            {"type": "tag", "begin": "[", "content": None, "end": "]"},
            {"type": "tag", "begin": "[", "end": THINK_CLOSE},
            {
                "type": "tag",
                "begin": "[",
                "content": [{"type": "any_text"}],
                "end": THINK_CLOSE,
            },
        )
        for format_node in formats:
            with self.subTest(format=format_node):
                response_format = ResponseFormat(
                    type="structural_tag",
                    structural_tag={"type": "structural_tag", "format": format_node},
                )
                with self.assertRaises(FtRuntimeException) as error:
                    self.prepare(thinking=False, response_format=response_format)
                self.assertEqual(
                    error.exception.exception_type, ExceptionType.INVALID_PARAMS
                )

    def test_structural_tag_excludes_must_be_a_list_of_strings(self):
        for excludes in (None, "x", 1, {}, [None], [{}]):
            with self.subTest(excludes=excludes):
                response_format = ResponseFormat(
                    type="structural_tag",
                    structural_tag={
                        "type": "structural_tag",
                        "format": {"type": "any_text", "excludes": excludes},
                    },
                )
                with self.assertRaises(FtRuntimeException) as error:
                    self.prepare(thinking=False, response_format=response_format)
                self.assertEqual(
                    error.exception.exception_type, ExceptionType.INVALID_PARAMS
                )

    def test_adaptive_plan_with_an_open_think_prompt_uses_the_bounded_channel(self):
        config = GenerateConfig(
            thinking_mode=ThinkingMode.ADAPTIVE,
            in_think_mode=False,
            max_thinking_tokens=1,
        )
        plan = KimiK3Grammar().build_plan(
            config, self.tokenizer, self.tokenizer.encode(THINK_OPEN)
        )
        plan.apply_to_config(config)
        matcher = xgr.GrammarMatcher(
            self.compiler.compile_structural_tag(config.structural_tag)
        )
        self.assertTrue(config.in_think_mode)
        self.accept(matcher, "x")
        self.assertFalse(self.allows(matcher, "x"))
        self.assert_completion_accepted(
            matcher, THINK_TO_RESPONSE + "answer" + RESPONSE_END
        )

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
        self.assert_completion_accepted(matcher, RESPONSE_OPEN + "a" + RESPONSE_END)
        self.assertFalse(self.allows(matcher, "<|close|>"))

    def test_budget_zero_and_one_bound_only_thinking_body(self):
        for budget in (0, 1):
            with self.subTest(budget=budget):
                _, matcher = self.prepare(budget=budget)
                if budget:
                    self.accept(matcher, "x")
                self.assertFalse(self.allows(matcher, "x"))
                self.assert_completion_accepted(
                    matcher, THINK_TO_RESPONSE + "xx" + RESPONSE_END
                )

    def test_negative_budget_keeps_k3_unbounded_thinking(self):
        config, matcher = self.prepare(budget=-1)
        self.assertEqual(config.max_thinking_tokens, INT32_MAX)
        self.assert_completion_accepted(
            matcher, "xx" + THINK_TO_RESPONSE + "a" + RESPONSE_END
        )

    def test_disabled_thinking_still_requires_response_and_message_close(self):
        config, matcher = self.prepare(thinking=False)
        self.assertFalse(config.in_think_mode)
        self.assertFalse(self.allows(matcher, "<|open|>"))
        self.accept(matcher, "a" + RESPONSE_CLOSE)
        self.assertFalse(self.allows(matcher, "EOS"))
        self.assert_completion_accepted(matcher, MESSAGE_CLOSE)

    def test_disabled_thinking_closes_an_upstream_open_think_block(self):
        config, matcher = self.prepare(thinking=False, prompt=THINK_OPEN)
        self.assertFalse(config.in_think_mode)
        self.assertFalse(self.allows(matcher, "x"))
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_END)

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
                self.assert_completion_accepted(
                    matcher, THINK_TO_RESPONSE + text + RESPONSE_END
                )

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
        self.assert_completion_accepted(
            matcher,
            '<|open|>tools<|sep|><|open|>call tool="lookup" index="1"<|sep|>'
            '<|open|>argument key="q" type="string"<|sep|>x<|close|>argument<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|>" + MESSAGE_CLOSE,
        )
        _, matcher = self.prepare(tools=tools)
        self.accept(matcher, THINK_TO_RESPONSE + "a" + RESPONSE_END)

    def test_required_tools_keep_json_arguments_and_force_a_call(self):
        tail = self.required_tool_tail({"type": "object"})
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
                self.assert_completion_accepted(matcher, "a" + RESPONSE_END)
                if mode == ThinkingMode.ADAPTIVE:
                    no_think = xgr.GrammarMatcher(
                        self.compiler.compile_structural_tag(config.structural_tag)
                    )
                    self.assert_completion_accepted(
                        no_think, RESPONSE_OPEN + "a" + RESPONSE_END
                    )

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
        config = GenerateConfig(
            num_return_sequences=2, thinking_mode=ThinkingMode.DISABLED
        )
        with self.assertRaisesRegex(FtRuntimeException, "num_return_sequences"):
            KimiK3Grammar().build_plan(config, self.tokenizer)

    def test_building_plan_preserves_request_grammar_fields(self):
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

    def test_openai_entry_prepares_model_grammar_once(self):
        from rtp_llm.config.py_config_modules import GenerateEnvConfig
        from rtp_llm.openai.api_datatype import ChatCompletionRequest
        from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
        from rtp_llm.openai.renderers.custom_renderer import RendererParams
        from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer

        generate_env_config = GenerateEnvConfig()
        generate_env_config.max_thinking_tokens = 8
        renderer = KimiK3Renderer(
            self.tokenizer,
            RendererParams("kimi_k3", 256, 0, []),
            generate_env_config,
        )
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.chat_renderer = renderer
        endpoint.tokenizer = self.tokenizer
        endpoint.stop_words_str_list = []
        endpoint.stop_words_id_list = []
        endpoint.generate_env_config = generate_env_config
        request = ChatCompletionRequest(
            messages=[], enable_thinking=True, response_format={"type": "json_object"}
        )
        with patch.object(
            renderer,
            "_prepare_request_constraints",
            wraps=renderer._prepare_request_constraints,
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
        ) as apply_plan, patch.object(
            renderer,
            "build_chat_completion_response_format_plan",
            wraps=renderer.build_chat_completion_response_format_plan,
        ) as build_plan:
            config = endpoint._extract_generation_config(
                request, self.tokenizer.encode(THINK_OPEN), renderer
            )
        prepare_params.assert_called_once()
        validate.assert_called_once()
        apply_plan.assert_called_once()
        build_plan.assert_called_once()
        self.assertEqual(config.max_thinking_tokens, 8)
        installed = copy.deepcopy(config.structural_tag)
        self.assertIsNotNone(config.finalize_response_format())
        self.assertEqual(config.structural_tag, installed)
        matcher = xgr.GrammarMatcher(
            self.compiler.compile_structural_tag(config.structural_tag)
        )
        self.assert_completion_accepted(
            matcher, THINK_TO_RESPONSE + "{}" + RESPONSE_END
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
