#!/usr/bin/env python3

import json
import unittest
from unittest.mock import Mock

import torch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, UsageInfo
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject
from rtp_llm.openai.renderers.kimi_k3_renderer import (
    KimiK3Renderer,
    _KimiK3StreamStatus,
    _KimiK3StreamStatusSync,
    _uses_reasoning_channel,
)


class KimiK3RendererTest(unittest.TestCase):
    def request(self, enable_thinking: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def bash_tool() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }

    def test_pretokenized_tool_metadata_selects_full_xtml_grammar(self) -> None:
        config = GenerateConfig(
            in_think_mode=True,
            max_thinking_tokens=1048576,
            response_format={"type": "text"},
        )

        source = KimiK3Renderer.apply_pretokenized_chat_request_constraints(
            {"tools": [self.bash_tool()], "tool_choice": "auto"},
            config,
            thinking=True,
        )

        tag = json.loads(config.structural_tag)
        tools_tag = tag["format"]["elements"][2]["content"]
        call_tag = tools_tag["content"]["tags"][0]
        self.assertEqual(tools_tag["begin"], "<|open|>tools<|sep|>")
        self.assertEqual(call_tag["begin"], '<|open|>call tool="bash" index="')
        self.assertEqual(call_tag["content"]["elements"][2]["style"], "kimi_k3_xml")
        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.end_think_token_ids, [])
        self.assertEqual(config.max_thinking_tokens, 0)
        self.assertEqual(source, "pretokenized_request_tools")

    def test_pretokenized_missing_metadata_selects_boundary_fallback(self) -> None:
        config = GenerateConfig(in_think_mode=True, max_thinking_tokens=1048576)

        source = KimiK3Renderer.apply_pretokenized_chat_request_constraints(
            {}, config, thinking=True
        )

        self.assertEqual(source, "common_prompt_tail_fallback")
        self.assertIsNone(config.structural_tag)
        self.assertTrue(config.in_think_mode)

    def test_pretokenized_tool_choice_none_selects_no_tools_grammar(self) -> None:
        config = GenerateConfig(in_think_mode=True, max_thinking_tokens=1048576)

        KimiK3Renderer.apply_pretokenized_chat_request_constraints(
            {"tools": [self.bash_tool()], "tool_choice": "none"},
            config,
            thinking=True,
        )

        tag = json.loads(config.structural_tag)
        serialized = json.dumps(tag, ensure_ascii=False)
        self.assertNotIn("<|open|>tools<|sep|>", serialized)
        response = tag["format"]["elements"][1]
        self.assertEqual(response["content"]["type"], "sequence")
        self.assertEqual(
            response["content"]["elements"][0],
            {"type": "regex", "pattern": r"[^<\s]"},
        )
        self.assertFalse(config.in_think_mode)

    def test_pretokenized_explicit_empty_tools_auto_selects_no_tools_grammar(
        self,
    ) -> None:
        config = GenerateConfig(in_think_mode=True, max_thinking_tokens=1048576)

        source = KimiK3Renderer.apply_pretokenized_chat_request_constraints(
            {"tools": [], "tool_choice": "auto"},
            config,
            thinking=True,
        )

        self.assertEqual(source, "pretokenized_request_no_tools")
        self.assertIsNotNone(config.structural_tag)
        tag = json.loads(config.structural_tag)
        serialized = json.dumps(tag, ensure_ascii=False)
        self.assertNotIn("<|open|>tools<|sep|>", serialized)
        self.assertEqual(tag["format"]["elements"][1]["content"]["type"], "sequence")
        self.assertFalse(config.in_think_mode)

    def test_pretokenized_required_tool_supersedes_response_format(self) -> None:
        config = GenerateConfig(
            in_think_mode=True,
            response_format={"type": "json_object"},
        )

        KimiK3Renderer.apply_pretokenized_chat_request_constraints(
            {"tools": [self.bash_tool()], "tool_choice": "required"},
            config,
            thinking=True,
        )

        tag = json.loads(config.structural_tag)
        response = tag["format"]["elements"][1]
        self.assertEqual(response["content"]["type"], "any_text")
        self.assertIsNone(config.response_format)

    def test_pretokenized_required_tool_without_tools_is_rejected(self) -> None:
        config = GenerateConfig(in_think_mode=True)

        with self.assertRaises(ValueError):
            KimiK3Renderer.apply_pretokenized_chat_request_constraints(
                {"tool_choice": "required"}, config, thinking=True
            )

    def test_pretokenized_malformed_tools_metadata_is_rejected(self) -> None:
        config = GenerateConfig(in_think_mode=True)

        with self.assertRaisesRegex(ValueError, "tools metadata must be a list"):
            KimiK3Renderer.apply_pretokenized_chat_request_constraints(
                {"tools": "not-a-list", "tool_choice": "auto"},
                config,
                thinking=True,
            )

    def test_pretokenized_parallel_tool_calls_false_limits_calls(self) -> None:
        config = GenerateConfig(in_think_mode=True)

        KimiK3Renderer.apply_pretokenized_chat_request_constraints(
            {
                "tools": [self.bash_tool()],
                "tool_choice": "required",
                "parallel_tool_calls": False,
            },
            config,
            thinking=True,
        )

        tools = json.loads(config.structural_tag)["format"]["elements"][2]
        self.assertTrue(tools["elements"][1]["stop_after_first"])

    def test_thinking_xtml_is_split_across_stream_chunks(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=True))
        chunks = [
            "reasoning<|close|>thi",
            "nk<|sep|><|open|>respo",
            "nse<|sep|>ANSWER: B<|close|>res",
            "ponse<|sep|><|close|>message<|sep|>",
        ]
        deltas = [KimiK3Renderer._parse_xtml_delta(status, chunk) for chunk in chunks]

        self.assertEqual(
            "".join(delta.reasoning_content or "" for delta in deltas), "reasoning"
        )
        self.assertEqual("".join(delta.content or "" for delta in deltas), "ANSWER: B")
        self.assertTrue(status.response_closed)
        self.assertEqual(status.xtml_pending, "")

    def test_blocking_status_uses_same_xtml_parser_as_async_path(self) -> None:
        class Tokenizer:
            pieces = {
                1: "reasoning",
                2: "<|close|>think<|sep|><|open|>response<|sep|>",
                3: "answer<|close|>response<|sep|><|close|>message<|sep|>",
            }

            def decode(self, token_ids):
                return "".join(self.pieces[token_id] for token_id in token_ids)

            def encode(self, text):
                return []

        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = Tokenizer()
        renderer.eos_token_id = 99
        renderer.max_seq_len = 1024
        renderer.extra_stop_words = []
        renderer.extra_stop_word_ids_list = []
        renderer.stop_words_id_list = []
        status = _KimiK3StreamStatusSync(self.request(enable_thinking=True))

        delta = renderer._update_single_status_sync(
            status,
            input_len=10,
            output_len=3,
            reuse_len=0,
            all_probs=torch.empty(0),
            output_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            max_new_tokens=3,
            stop_words_str=[],
            stop_word_slice_list=[],
            is_streaming=False,
        )

        self.assertEqual(delta.output_str.reasoning_content, "reasoning")
        self.assertEqual(delta.output_str.content, "answer")
        self.assertTrue(status.response_closed)

        complete = json.loads(
            renderer.collect_complete_response(
                [
                    renderer._generate_first_sync(1),
                    renderer._generate_stream_response_sync([delta]),
                    renderer._generate_final_sync([status], [10], [3], [0]),
                ]
            )
        )
        message = complete["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "reasoning")
        self.assertEqual(message["content"], "answer")
        self.assertNotIn("<|close|>", json.dumps(message))

    def test_cpp_constraint_bridge_returns_only_renderer_changes(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        request_json = self.request(enable_thinking=True).model_dump_json(
            exclude_unset=True
        )
        config = GenerateConfig(
            temperature=0.7,
            top_p=1.0,
            in_think_mode=True,
            max_thinking_tokens=123,
            trace_id="preserve-native-state",
        )

        updates_json, cleared = renderer.apply_chat_completion_constraints_from_json(
            request_json, config.model_dump_json()
        )
        updates = json.loads(updates_json)

        self.assertEqual(updates["temperature"], 1.0)
        self.assertEqual(updates["top_p"], 0.95)
        self.assertIn("structural_tag", updates)
        self.assertFalse(updates["in_think_mode"])
        self.assertEqual(updates["max_thinking_tokens"], 0)
        self.assertNotIn("trace_id", updates)
        self.assertEqual(cleared, [])

    def test_non_thinking_terminal_envelope_is_removed(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        delta = KimiK3Renderer._parse_xtml_delta(
            status,
            "ANSWER: D<|close|>response<|sep|><|close|>message<|sep|>",
        )

        self.assertEqual(delta.reasoning_content, "")
        self.assertEqual(delta.content, "ANSWER: D")
        self.assertTrue(status.response_closed)

    def test_official_thinking_disabled_uses_response_channel(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            thinking={"type": "disabled"},
        )
        status = _KimiK3StreamStatus(request)

        self.assertFalse(_uses_reasoning_channel(request))
        self.assertFalse(status.in_reasoning)
        self.assertFalse(
            KimiK3Renderer._template_kwargs(request, request.model_dump(mode="json"))[
                "thinking"
            ]
        )

    def test_structured_output_uses_response_channel(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            response_format={"type": "json_object"},
        )
        status = _KimiK3StreamStatus(request)
        delta = KimiK3Renderer._parse_xtml_delta(status, '{"answer":true}', flush=True)

        self.assertFalse(_uses_reasoning_channel(request))
        self.assertEqual(delta.reasoning_content, "")
        self.assertEqual(delta.content, '{"answer":true}')

    def test_explicit_reasoning_effort_overrides_structured_output_default(
        self,
    ) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            response_format={"type": "json_object"},
            reasoning_effort="max",
        )

        kwargs = KimiK3Renderer._template_kwargs(
            request, request.model_dump(exclude_none=True, mode="json")
        )

        self.assertTrue(kwargs["thinking"])
        self.assertTrue(_uses_reasoning_channel(request))
        self.assertEqual(kwargs["thinking_effort"], "max")

    def test_reasoning_effort_none_uses_response_without_forwarding_effort(
        self,
    ) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            reasoning_effort="none",
        )

        kwargs = KimiK3Renderer._template_kwargs(
            request, request.model_dump(exclude_none=True, mode="json")
        )

        self.assertFalse(kwargs["thinking"])
        self.assertNotIn("thinking_effort", kwargs)
        self.assertFalse(_uses_reasoning_channel(request))

        config = GenerateConfig(in_think_mode=True, max_thinking_tokens=42)
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.apply_chat_completion_constraints(request, config)
        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.max_thinking_tokens, 0)

    def test_thinking_request_constrains_think_block_with_grammar(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(in_think_mode=True, max_thinking_tokens=100)

        renderer.apply_chat_completion_constraints(
            self.request(enable_thinking=True), config
        )

        # The think block lives inside the grammar, so the engine must not run
        # its own think-phase gating: stop tokens stay grammar-masked until
        # <|close|>message<|sep|> closes the turn.
        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.begin_think_token_ids, [])
        self.assertEqual(config.end_think_token_ids, [])
        self.assertEqual(config.max_thinking_tokens, 0)

        tag = json.loads(config.structural_tag)
        self.assertEqual(tag["type"], "structural_tag")
        elements = tag["format"]["elements"]
        self.assertEqual(elements[0]["begin"], "")
        self.assertEqual(elements[0]["content"]["excludes"], ["<|open|>", "<|close|>"])
        self.assertEqual(elements[0]["end"], "<|close|>think<|sep|>")
        self.assertEqual(elements[1]["begin"], "<|open|>response<|sep|>")
        self.assertEqual(
            elements[1]["content"],
            {"type": "any_text", "excludes": ["<|open|>", "<|close|>"]},
        )
        self.assertEqual(elements[1]["end"], "<|close|>response<|sep|>")
        self.assertEqual(elements[-1]["value"], "<|close|>message<|sep|>")

    def test_non_thinking_request_omits_think_element(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(in_think_mode=True)

        renderer.apply_chat_completion_constraints(
            self.request(enable_thinking=False), config
        )

        self.assertFalse(config.in_think_mode)
        tag = json.loads(config.structural_tag)
        elements = tag["format"]["elements"]
        self.assertEqual(elements[0]["begin"], "")
        self.assertEqual(elements[0]["end"], "<|close|>response<|sep|>")
        self.assertEqual(elements[-1]["value"], "<|close|>message<|sep|>")

    def test_thinking_effort_precedes_legacy_reasoning_effort(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            thinking={"type": "enabled", "effort": "low"},
            reasoning_effort="max",
        )

        kwargs = KimiK3Renderer._template_kwargs(
            request, request.model_dump(exclude_none=True, mode="json")
        )
        self.assertEqual(kwargs["thinking_effort"], "low")

    def test_pretokenized_chat_constraints_declare_stateful_completion_guard(
        self,
    ) -> None:
        constraints = KimiK3Renderer.pretokenized_chat_constraints()

        self.assertEqual(
            constraints["reasoning"]["prompt_tail"], "<|open|>think<|sep|>"
        )
        self.assertEqual(
            constraints["response"]["prompt_tail"],
            "<|open|>response<|sep|>",
        )
        reasoning = constraints["reasoning"]["completion_guard"]
        response = constraints["response"]["completion_guard"]
        self.assertEqual(reasoning["think_close"], "<|close|>think<|sep|>")
        self.assertEqual(reasoning["response_open"], "<|open|>response<|sep|>")
        self.assertEqual(response["think_close"], "")
        self.assertEqual(response["response_open"], "")
        for phase in (reasoning, response):
            self.assertEqual(phase["response_close"], "<|close|>response<|sep|>")
            self.assertEqual(phase["tools_open"], "<|open|>tools<|sep|>")
            self.assertEqual(phase["tools_close"], "<|close|>tools<|sep|>")
            self.assertEqual(phase["message_close"], "<|close|>message<|sep|>")
        for phase in ("reasoning", "response"):
            self.assertNotIn("structural_tag", constraints[phase])

    def test_dynamic_tools_stay_in_messages(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                return [1, 2, 3] if kwargs["tokenize"] else "prompt"

        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "system", "content": "", "tools": [tool]},
                    {"role": "user", "content": "hello"},
                ]
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = Tokenizer()
        renderer.max_seq_len = 0

        rendered = renderer.render_chat(request)

        self.assertEqual(rendered.input_ids, [1, 2, 3])
        self.assertEqual(rendered.rendered_prompt, "")
        self.assertEqual(len(renderer.tokenizer.calls), 1)
        for messages, kwargs in renderer.tokenizer.calls:
            self.assertEqual(messages[0]["tools"], [tool])
            self.assertIsNone(kwargs["tools"])

    @staticmethod
    def _weather_tool(name: str = "get_weather") -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": "Return weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }

    def test_required_tool_choice_builds_native_xtml_constraint(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig()

        renderer.apply_chat_completion_constraints(request, config)

        tag = json.loads(config.structural_tag)
        elements = tag["format"]["elements"]
        self.assertEqual(elements[0]["begin"], "")
        self.assertEqual(elements[0]["end"], "<|close|>response<|sep|>")
        tools = elements[1]["elements"]
        self.assertEqual(tools[0]["value"], "<|open|>tools<|sep|>")
        self.assertEqual(tools[2]["value"], "<|close|>tools<|sep|>")
        self.assertTrue(tools[1]["at_least_one"])
        self.assertFalse(tools[1]["stop_after_first"])
        call = tools[1]["tags"][0]
        self.assertIn('tool="get_weather" index="', call["begin"])
        self.assertEqual(call["content"]["elements"][0]["pattern"], r"\d+")
        self.assertEqual(call["content"]["elements"][2]["style"], "kimi_k3_xml")
        self.assertEqual(
            call["content"]["elements"][2]["json_schema"],
            self._weather_tool()["function"]["parameters"],
        )
        self.assertEqual(elements[-1]["value"], "<|close|>message<|sep|>")

    def test_parallel_tool_calls_false_limits_constraint_to_one_call(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig()

        renderer.apply_chat_completion_constraints(request, config)

        tools = json.loads(config.structural_tag)["format"]["elements"][1]
        self.assertTrue(tools["elements"][1]["stop_after_first"])

    def test_named_tool_choice_filters_constraint_alternatives(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Search"}],
                "tools": [self._weather_tool(), self._weather_tool("search")],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "search"},
                },
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig()

        renderer.apply_chat_completion_constraints(request, config)

        tools = json.loads(config.structural_tag)["format"]["elements"][1]
        self.assertEqual(tools["type"], "sequence")
        call = tools["elements"][1]
        self.assertIn('tool="search"', call["begin"])

    def test_dynamic_required_tool_is_in_constraint(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "system", "content": "", "tools": [self._weather_tool()]},
                    {"role": "user", "content": "Weather?"},
                ],
                "tool_choice": "required",
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig()

        renderer.apply_chat_completion_constraints(request, config)

        tags = json.loads(config.structural_tag)["format"]["elements"][1]["elements"][
            1
        ]["tags"]
        self.assertEqual(len(tags), 1)
        self.assertIn('tool="get_weather"', tags[0]["begin"])

    def test_thinking_constraint_includes_think_element_before_tools(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "thinking": {"type": "enabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(in_think_mode=True, end_think_token_ids=[9])

        renderer.apply_chat_completion_constraints(request, config)

        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.end_think_token_ids, [])
        elements = json.loads(config.structural_tag)["format"]["elements"]
        self.assertEqual(elements[0]["end"], "<|close|>think<|sep|>")
        self.assertEqual(elements[1]["begin"], "<|open|>response<|sep|>")
        self.assertEqual(elements[2]["type"], "sequence")
        self.assertEqual(elements[-1]["value"], "<|close|>message<|sep|>")

    def test_response_format_composes_into_response_body(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Answer in JSON."}],
                "response_format": {"type": "json_object"},
                "reasoning_effort": "max",
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(json_schema='{"type":"object"}')

        renderer.apply_chat_completion_constraints(request, config)

        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.end_think_token_ids, [])
        self.assertIsNone(config.json_schema)
        elements = json.loads(config.structural_tag)["format"]["elements"]
        self.assertEqual(elements[0]["end"], "<|close|>think<|sep|>")
        self.assertEqual(
            elements[1]["content"],
            {"type": "json_schema", "json_schema": {"type": "object"}},
        )
        self.assertEqual(elements[1]["end"], "<|close|>response<|sep|>")

    def test_omitted_thinking_enables_think_element(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(in_think_mode=False)

        renderer.apply_chat_completion_constraints(request, config)

        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.end_think_token_ids, [])
        elements = json.loads(config.structural_tag)["format"]["elements"]
        self.assertEqual(elements[0]["end"], "<|close|>think<|sep|>")
        self.assertEqual(elements[1]["begin"], "<|open|>response<|sep|>")

    def test_k3_sampling_contract_accepts_supported_profiles(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = Mock()
        renderer.tokenizer.encode.return_value = [101, 102, 103]
        cases = [
            ({"thinking": {"type": "enabled"}, "temperature": 0.0}, 0.0, 0.95),
            ({"thinking": {"type": "enabled"}, "temperature": 0.6}, 0.6, 0.95),
            ({"thinking": {"type": "enabled"}, "temperature": 1.0}, 1.0, 0.95),
            (
                {
                    "thinking": {"type": "enabled", "effort": "max"},
                    "temperature": 0.95,
                    "top_p": 1.0,
                },
                0.95,
                1.0,
            ),
            ({"thinking": {"type": "disabled"}}, 0.6, 0.95),
        ]
        for overrides, expected_temperature, expected_top_p in cases:
            with self.subTest(overrides=overrides):
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "question"}],
                        **overrides,
                    }
                )
                config = GenerateConfig()
                renderer.apply_chat_completion_constraints(request, config)
                self.assertEqual(config.temperature, expected_temperature)
                self.assertEqual(config.top_p, expected_top_p)

    def test_k3_sampling_contract_rejects_kvv_invalid_values(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        cases = [
            ("temperature", 1.1),
            ("temperature", 2.0),
            ("temperature", -0.1),
            ("top_p", 0.8),
            ("presence_penalty", 0.5),
            ("frequency_penalty", 0.5),
            ("n", 2),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value), self.assertRaisesRegex(
                FtRuntimeException, name
            ):
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "question"}],
                        "thinking": {"type": "enabled"},
                        name: value,
                    }
                )
                renderer.apply_chat_completion_constraints(request, GenerateConfig())

    def test_k3_sampling_contract_accepts_top_p_one_for_all_modes(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        for thinking in ({"type": "enabled"}, {"type": "disabled"}):
            with self.subTest(thinking=thinking):
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "question"}],
                        "thinking": thinking,
                        "top_p": 1.0,
                    }
                )
                config = GenerateConfig()
                renderer.apply_chat_completion_constraints(request, config)
                self.assertEqual(config.top_p, 1.0)

    def test_existing_grammar_composes_as_response_body(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(json_schema={"type": "object"})

        renderer.apply_chat_completion_constraints(request, config)

        self.assertIsNone(config.json_schema)
        elements = json.loads(config.structural_tag)["format"]["elements"]
        self.assertEqual(
            elements[0]["content"],
            {"type": "json_schema", "json_schema": {"type": "object"}},
        )
        self.assertEqual(elements[1]["type"], "sequence")

    def test_forced_tool_choice_supersedes_response_format_grammar(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "response_format": {"type": "json_object"},
                "reasoning_effort": "max",
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(json_schema='{"type":"object"}')

        renderer.apply_chat_completion_constraints(request, config)

        self.assertIsNone(config.json_schema)
        elements = json.loads(config.structural_tag)["format"]["elements"]
        self.assertEqual(elements[0]["end"], "<|close|>think<|sep|>")
        self.assertEqual(
            elements[1]["content"],
            {"type": "any_text", "excludes": ["<|open|>", "<|close|>"]},
        )
        self.assertEqual(elements[2]["type"], "sequence")
        call_schema = elements[2]["elements"][1]["tags"][0]["content"]["elements"][2]
        self.assertEqual(call_schema["type"], "json_schema")
        self.assertEqual(
            call_schema["json_schema"], self._weather_tool()["function"]["parameters"]
        )

    def test_pending_generation_channel_is_excluded_from_usage(self) -> None:
        class Tokenizer:
            def encode(self, text: str, add_special_tokens: bool) -> list[int]:
                if add_special_tokens:
                    raise AssertionError("generation channel text must stay ordinary")
                return {"response": [11], "think": [12]}[text]

        tokenizer = Tokenizer()
        pending_tokens = KimiK3Renderer._pending_prompt_token_count(
            tokenizer, thinking=False
        )
        response = StreamResponseObject(
            usage=UsageInfo(prompt_tokens=39, completion_tokens=7, total_tokens=46)
        )

        KimiK3Renderer._subtract_pending_prompt_tokens(response, pending_tokens)

        usage = response.usage
        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(pending_tokens, 3)
        self.assertEqual(usage.prompt_tokens, 36)
        self.assertEqual(usage.completion_tokens, 7)
        self.assertEqual(usage.total_tokens, 43)

    def test_invalid_channel_marker_remains_visible(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = "answer<|close|>think<|sep|>still-visible"
        delta = KimiK3Renderer._parse_xtml_delta(status, text, flush=True)

        self.assertEqual(delta.content, text)
        self.assertFalse(status.response_closed)

    def _run_chunks(self, status, chunks):
        deltas = []
        for index, chunk in enumerate(chunks):
            deltas.append(
                KimiK3Renderer._parse_xtml_delta(
                    status, chunk, flush=index == len(chunks) - 1
                )
            )
        return deltas

    def _all_tool_calls(self, deltas):
        calls = []
        for delta in deltas:
            if delta.tool_calls:
                calls.extend(delta.tool_calls)
        return calls

    def test_tools_channel_json_block_is_parsed(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = (
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{"city": "北京", "days": 3}'
            "<|close|>json<|sep|>"
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].index, 0)
        self.assertEqual(calls[0].type, "function")
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertTrue(calls[0].id.startswith("call_"))
        self.assertEqual(
            json.loads(calls[0].function.arguments), {"city": "北京", "days": 3}
        )
        self.assertEqual(status.tool_calls_seen, 1)
        self.assertEqual("".join(d.content or "" for d in deltas), "")

    def test_tools_channel_typed_arguments_are_coerced(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = (
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="search" index="1"<|sep|>'
            '<|open|>argument key="query" type="string"<|sep|>天气<|close|>argument<|sep|>'
            '<|open|>argument key="limit" type="number"<|sep|>5<|close|>argument<|sep|>'
            '<|open|>argument key="fresh" type="boolean"<|sep|>true<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"query": "天气", "limit": 5, "fresh": True},
        )

    def test_tools_channel_split_across_stream_chunks(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        chunks = [
            "OK<|close|>respo",
            "nse<|sep|><|open|>to",
            'ols<|sep|><|open|>call tool="get_wea',
            'ther" index="1"<|sep|><|open|>json type="object"<|sep|>{"ci',
            'ty": "杭州"}<|close|>js',
            "on<|sep|><|close|>call<|sep|><|close|>too",
            "ls<|sep|>",
        ]
        deltas = self._run_chunks(status, chunks)

        self.assertEqual("".join(d.content or "" for d in deltas), "OK")
        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_weather")
        self.assertEqual(json.loads(calls[0].function.arguments), {"city": "杭州"})

    def test_tools_channel_multiple_calls(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = (
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="a" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            '<|open|>call tool="b" index="2"<|sep|>'
            '<|open|>json type="object"<|sep|>{"k": 1}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        calls = self._all_tool_calls(deltas)
        self.assertEqual([c.index for c in calls], [0, 1])
        self.assertEqual([c.function.name for c in calls], ["a", "b"])

    def test_tools_channel_unclosed_is_flushed(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        deltas = self._run_chunks(
            status,
            [
                "<|close|>response<|sep|><|open|>tools<|sep|>"
                '<|open|>call tool="a" index="1"<|sep|>'
                '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
                "<|close|>call<|sep|>"
            ],
        )

        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "a")

    def test_tools_channel_escaped_attributes(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        text = (
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="a&amp;b&quot;c" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        calls = self._all_tool_calls(deltas)
        self.assertEqual(calls[0].function.name, 'a&b"c')

    def test_thinking_then_tools_channel(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=True))
        text = (
            "think<|close|>think<|sep|><|open|>response<|sep|>"
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{"city": "上海"}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        deltas = self._run_chunks(status, [text])

        self.assertEqual(deltas[0].reasoning_content, "think")
        calls = self._all_tool_calls(deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].function.arguments), {"city": "上海"})

    def test_no_tools_channel_keeps_empty_tool_calls(self) -> None:
        status = _KimiK3StreamStatus(self.request(enable_thinking=False))
        deltas = self._run_chunks(
            status, ["ANSWER<|close|>response<|sep|><|end_of_msg|>"]
        )

        self.assertEqual(self._all_tool_calls(deltas), [])
        self.assertEqual(status.tool_calls_seen, 0)
        self.assertEqual("".join(d.content or "" for d in deltas), "ANSWER")


if __name__ == "__main__":
    unittest.main()
