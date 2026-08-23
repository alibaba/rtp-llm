#!/usr/bin/env python3

import json
import unittest
from unittest.mock import Mock

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, UsageInfo
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject
from rtp_llm.openai.renderers.kimi_k3_renderer import (
    KimiK3Renderer,
    _KimiK3StreamStatus,
    _uses_reasoning_channel,
)


class KimiK3RendererTest(unittest.TestCase):
    def request(self, enable_thinking: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            enable_thinking=enable_thinking,
        )

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

    def test_thinking_budget_uses_k3_xtml_transition_without_grammar(self) -> None:
        """Without grammar constraints, end_think_token_ids should NOT be set."""
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = Mock()
        renderer.tokenizer.encode.return_value = [101, 102, 103]
        config = GenerateConfig(max_thinking_tokens=100)

        renderer.apply_chat_completion_constraints(
            self.request(enable_thinking=True), config
        )

        # No grammar constraint → end_think_token_ids stays default (empty)
        self.assertEqual(config.end_think_token_ids, [])
        renderer.tokenizer.encode.assert_not_called()

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
        fmt = tag["format"]
        self.assertEqual(
            fmt["begin"],
            "<|close|>response<|sep|><|open|>tools<|sep|>",
        )
        self.assertEqual(fmt["end"], "<|close|>tools<|sep|>")
        self.assertTrue(fmt["content"]["at_least_one"])
        self.assertTrue(fmt["content"]["stop_after_first"])
        call = fmt["content"]["tags"][0]
        self.assertIn('tool="get_weather" index="1"', call["begin"])
        self.assertEqual(
            call["content"]["json_schema"],
            self._weather_tool()["function"]["parameters"],
        )

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

        tags = json.loads(config.structural_tag)["format"]["content"]["tags"]
        self.assertEqual(len(tags), 1)
        self.assertIn('tool="search"', tags[0]["begin"])

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

        tags = json.loads(config.structural_tag)["format"]["content"]["tags"]
        self.assertEqual(len(tags), 1)
        self.assertIn('tool="get_weather"', tags[0]["begin"])

    def test_thinking_constraint_switches_after_full_xtml_boundary(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.encoded = []

            def encode(self, text: str, add_special_tokens: bool) -> list[int]:
                self.encoded.append((text, add_special_tokens))
                return [101, 102, 103]

        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "thinking": {"type": "enabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = Tokenizer()
        config = GenerateConfig(in_think_mode=True, end_think_token_ids=[9])

        renderer.apply_chat_completion_constraints(request, config)

        self.assertEqual(config.end_think_token_ids, [101, 102, 103])
        self.assertEqual(
            renderer.tokenizer.encoded,
            [(KimiK3Renderer._THINK_TO_RESPONSE, False)],
        )

    def test_response_format_switches_after_full_xtml_boundary(self) -> None:
        class Tokenizer:
            def encode(self, text: str, add_special_tokens: bool) -> list[int]:
                self.encoded = (text, add_special_tokens)
                return [101, 102, 103]

        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Answer in JSON."}],
                "response_format": {"type": "json_object"},
                "reasoning_effort": "max",
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = Tokenizer()
        config = GenerateConfig(json_schema='{"type":"object"}')

        renderer.apply_chat_completion_constraints(request, config)

        self.assertEqual(config.end_think_token_ids, [101, 102, 103])
        self.assertEqual(
            renderer.tokenizer.encoded,
            (KimiK3Renderer._THINK_TO_RESPONSE, False),
        )

    def test_omitted_thinking_enables_forced_tool_constraint(self) -> None:
        class Tokenizer:
            def encode(self, text: str, add_special_tokens: bool) -> list[int]:
                self.encoded = (text, add_special_tokens)
                return [101, 102, 103]

        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = Tokenizer()
        config = GenerateConfig(in_think_mode=False)

        renderer.apply_chat_completion_constraints(request, config)

        self.assertTrue(config.in_think_mode)
        self.assertEqual(config.end_think_token_ids, [101, 102, 103])
        self.assertEqual(
            renderer.tokenizer.encoded,
            (KimiK3Renderer._THINK_TO_RESPONSE, False),
        )

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

    def test_forced_tool_choice_rejects_existing_grammar(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)

        with self.assertRaisesRegex(
            FtRuntimeException, "conflicts with existing grammar constraint"
        ):
            renderer.apply_chat_completion_constraints(
                request, GenerateConfig(json_schema={"type": "object"})
            )

    def test_forced_tool_choice_supersedes_response_format_grammar(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(json_schema='{"type":"object"}')

        renderer.apply_chat_completion_constraints(request, config)

        self.assertIsNone(config.json_schema)
        self.assertIsNotNone(config.structural_tag)

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
