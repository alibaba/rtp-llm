#!/usr/bin/env python3

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    validate_kimi_k3_tool_history,
)
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    FinisheReason,
    PromptTokensDetails,
    TopLogprob,
    UsageInfo,
)
from rtp_llm.openai.renderers.custom_renderer import (
    RenderedInputs,
    StreamResponseObject,
)
from rtp_llm.openai.renderers.kimi_k3_renderer import (
    K3RequestOptions,
    KimiK3Renderer,
    _K3StreamResponseObject,
    _KimiK3StreamStatus,
)


class KimiK3RendererTest(unittest.TestCase):
    def request(self, enable_thinking: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            enable_thinking=enable_thinking,
        )

    def status(self, thinking: bool) -> _KimiK3StreamStatus:
        return _KimiK3StreamStatus(self.request(thinking), thinking=thinking)

    @staticmethod
    def _history_call(call_id: str, name="lookup", arguments='{"q":"x"}'):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    @staticmethod
    def _logprob(token: str, value: float = -0.1) -> ChatCompletionTokenLogprob:
        return ChatCompletionTokenLogprob(
            token=token,
            logprob=value,
            bytes=list(token.encode("utf-8")),
            top_logprobs=[
                TopLogprob(
                    token=token,
                    logprob=value,
                    bytes=list(token.encode("utf-8")),
                )
            ],
        )

    def test_tool_history_accepts_parallel_results_in_any_order(self) -> None:
        validate_kimi_k3_tool_history(
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        self._history_call("call_a"),
                        self._history_call("call_b"),
                    ],
                },
                {"role": "tool", "tool_call_id": "call_b", "content": "b"},
                {"role": "tool", "tool_call_id": "call_a", "content": "a"},
                {"role": "user", "content": "continue"},
            ]
        )

    def test_tool_history_rejects_malformed_calls_and_results(self) -> None:
        cases = {
            "missing id": [
                {"role": "assistant", "tool_calls": [self._history_call("")]}
            ],
            "missing name": [
                {
                    "role": "assistant",
                    "tool_calls": [self._history_call("call_a", name=None)],
                }
            ],
            "missing arguments": [
                {
                    "role": "assistant",
                    "tool_calls": [self._history_call("call_a", arguments=None)],
                }
            ],
            "invalid arguments": [
                {
                    "role": "assistant",
                    "tool_calls": [self._history_call("call_a", arguments="{")],
                }
            ],
            "non-object arguments": [
                {
                    "role": "assistant",
                    "tool_calls": [self._history_call("call_a", arguments="[]")],
                }
            ],
            "unexpected result": [
                {"role": "tool", "tool_call_id": "call_a", "content": "a"}
            ],
        }
        for name, messages in cases.items():
            with self.subTest(name=name), self.assertRaises(FtRuntimeException):
                validate_kimi_k3_tool_history(messages)

    def test_tool_history_requires_all_results_before_next_turn(self) -> None:
        with self.assertRaisesRegex(FtRuntimeException, "advances the conversation"):
            validate_kimi_k3_tool_history(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            self._history_call("call_a"),
                            self._history_call("call_b"),
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_a", "content": "a"},
                    {"role": "user", "content": "continue"},
                ]
            )

    def test_tool_history_rejects_duplicate_call_ids_and_results(self) -> None:
        duplicate_calls = [
            {
                "role": "assistant",
                "tool_calls": [
                    self._history_call("call_a"),
                    self._history_call("call_a"),
                ],
            }
        ]
        with self.assertRaisesRegex(FtRuntimeException, "duplicates"):
            validate_kimi_k3_tool_history(duplicate_calls)

        duplicate_results = [
            {
                "role": "assistant",
                "tool_calls": [self._history_call("call_a")],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "a"},
            {"role": "tool", "tool_call_id": "call_a", "content": "again"},
        ]
        with self.assertRaisesRegex(FtRuntimeException, "does not match"):
            validate_kimi_k3_tool_history(duplicate_results)

    def test_thinking_xtml_is_split_across_stream_chunks(self) -> None:
        status = self.status(True)
        chunks = [
            "reasoning<|close|>thi",
            "nk<|sep|><|open|>respo",
            "nse<|sep|>ANSWER: B<|close|>res",
            "ponse<|sep|><|close|>message<|sep|>",
        ]
        deltas = []
        for chunk in chunks:
            deltas.extend(
                KimiK3Renderer._split_exclusive_delta(
                    KimiK3Renderer._parse_xtml_delta(status, chunk)
                )
            )

        self.assertEqual(
            "".join(delta.reasoning_content or "" for delta in deltas), "reasoning"
        )
        self.assertEqual("".join(delta.content or "" for delta in deltas), "ANSWER: B")
        self.assertTrue(status.xtml_decoder.response_closed)
        self.assertEqual(status.xtml_decoder.xtml_pending, "")

        serialized = [delta.model_dump(exclude_none=True) for delta in deltas]
        for payload in serialized:
            present = {
                key
                for key in ("content", "reasoning_content", "tool_calls")
                if key in payload
            }
            self.assertLessEqual(len(present), 1, payload)

    def test_empty_reasoning_chunk_is_not_a_boundary(self) -> None:
        status = self.status(True)

        delta = KimiK3Renderer._parse_xtml_delta(status, "")

        self.assertEqual(delta.model_dump(exclude_none=True), {})

    def test_transition_without_reasoning_does_not_emit_early_boundary(self) -> None:
        status = self.status(True)

        deltas = KimiK3Renderer._split_exclusive_delta(
            KimiK3Renderer._parse_xtml_delta(
                status, KimiK3Renderer._THINK_TO_RESPONSE + "answer"
            )
        )

        self.assertEqual(
            [delta.model_dump(exclude_none=True) for delta in deltas],
            [{"content": "answer"}],
        )

    def test_transition_after_reasoning_emits_boundary_before_content(self) -> None:
        status = self.status(True)
        reasoning = KimiK3Renderer._parse_xtml_delta(status, "reasoning")
        transition = KimiK3Renderer._parse_xtml_delta(
            status, KimiK3Renderer._THINK_TO_RESPONSE + "answer"
        )

        deltas = KimiK3Renderer._split_exclusive_delta(transition)

        self.assertEqual(
            reasoning.model_dump(exclude_none=True),
            {"reasoning_content": "reasoning"},
        )
        self.assertEqual(
            [delta.model_dump(exclude_none=True) for delta in deltas],
            [{"reasoning_content": ""}, {"content": "answer"}],
        )

    def test_mixed_response_inserts_boundary_before_content(self) -> None:
        response = StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        reasoning_content="reasoning", content="answer"
                    ),
                )
            ],
        )

        frames = KimiK3Renderer._split_exclusive_response(response)

        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].choices[0].delta.reasoning_content, "reasoning")
        self.assertIsNone(frames[0].choices[0].finish_reason)
        self.assertIsNone(frames[0].usage)
        self.assertEqual(frames[1].choices[0].delta.reasoning_content, "")
        self.assertEqual(frames[2].choices[0].delta.content, "answer")
        self.assertIsNone(frames[2].choices[0].finish_reason)

    def test_mixed_response_splits_logprobs_by_visible_channel(self) -> None:
        reasoning_logprob = self._logprob("reasoning", -0.2)
        content_logprob = self._logprob("answer", -0.3)
        response = _K3StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        reasoning_content="reasoning", content="answer"
                    ),
                )
            ],
            channel_logprobs={
                0: {
                    "reasoning_content": [reasoning_logprob],
                    "content": [content_logprob],
                    "tool_calls": [],
                }
            },
        )

        frames = KimiK3Renderer._split_exclusive_response(response)

        self.assertEqual(len(frames), 3)
        self.assertEqual(
            frames[0].choices[0].logprobs.content, [reasoning_logprob]
        )
        self.assertIsNone(frames[1].choices[0].logprobs)
        self.assertEqual(frames[2].choices[0].logprobs.content, [content_logprob])

    def test_logprobs_state_machine_filters_split_xtml_markers(self) -> None:
        status = self.status(True)
        reasoning_logprob = self._logprob("reasoning", -0.2)
        content_logprob = self._logprob("answer", -0.3)

        first = KimiK3Renderer._split_logprobs_by_channel(
            status,
            [reasoning_logprob, self._logprob("<|close|>")],
        )
        second = KimiK3Renderer._split_logprobs_by_channel(
            status,
            [
                self._logprob("think"),
                self._logprob("<|sep|>"),
                self._logprob("<|open|>"),
                self._logprob("response"),
                self._logprob("<|sep|>"),
                content_logprob,
                self._logprob("<|close|>"),
                self._logprob("response"),
                self._logprob("<|sep|>"),
                self._logprob("<|open|>"),
                self._logprob("tools"),
                self._logprob("<|sep|>"),
                self._logprob("tool-body"),
                self._logprob("<|close|>"),
                self._logprob("tools"),
                self._logprob("<|sep|>"),
                self._logprob("<|end_of_msg|>"),
            ],
            flush=True,
        )

        self.assertEqual(first["reasoning_content"], [reasoning_logprob])
        self.assertEqual(first["content"], [])
        self.assertEqual(second["reasoning_content"], [])
        self.assertEqual(second["content"], [content_logprob])
        self.assertEqual(second["tool_calls"], [])
        self.assertEqual(status.xtml_decoder.logprobs_marker_pending, [])

    def test_role_frame_drops_empty_reasoning_and_logprobs(self) -> None:
        response = StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role="assistant", content="", reasoning_content=""
                    ),
                    logprobs=ChoiceLogprobs(content=[]),
                )
            ]
        )

        frames = KimiK3Renderer._split_exclusive_response(response)

        self.assertEqual(len(frames), 1)
        choice = frames[0].choices[0]
        self.assertEqual(
            choice.delta.model_dump(exclude_none=True),
            {"role": "assistant", "content": ""},
        )
        self.assertIsNone(choice.logprobs)

    def test_multi_choice_final_is_split_with_per_choice_usage(self) -> None:
        aggregate = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        response = StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=index,
                    delta=DeltaMessage(content=""),
                    finish_reason=FinisheReason.stop,
                    usage=UsageInfo(
                        prompt_tokens=10,
                        completion_tokens=completion_tokens,
                        total_tokens=10 + completion_tokens,
                    ),
                )
                for index, completion_tokens in enumerate((2, 3))
            ],
            usage=aggregate,
        )

        frames = KimiK3Renderer._split_exclusive_response(response)

        self.assertEqual(len(frames), 2)
        self.assertEqual([frame.choices[0].index for frame in frames], [0, 1])
        self.assertTrue(all(frame.usage is None for frame in frames))
        self.assertTrue(
            all(
                frame.choices[0].delta.model_dump(exclude_none=True) == {}
                for frame in frames
            )
        )
        self.assertEqual(
            [frame.choices[0].usage.completion_tokens for frame in frames],
            [2, 3],
        )

    def test_generate_final_computes_usage_for_each_choice(self) -> None:
        renderer = object.__new__(KimiK3Renderer)
        buffers = [
            SimpleNamespace(
                output=SimpleNamespace(
                    aux_info=SimpleNamespace(
                        input_len=10,
                        output_len=output_tokens,
                        reuse_len=0,
                        multimodal_lengths={},
                    )
                ),
                finish_reason=FinisheReason.stop,
            )
            for output_tokens in (2, 3)
        ]
        think_statuses = [
            SimpleNamespace(enable_think_mode=False, think_tokens=0),
            SimpleNamespace(enable_think_mode=False, think_tokens=0),
        ]

        response = asyncio.run(
            renderer._generate_final(
                buffers, self.request(enable_thinking=False), think_statuses
            )
        )

        self.assertEqual(response.usage.completion_tokens, 5)
        self.assertEqual(
            [choice.usage.completion_tokens for choice in response.choices],
            [2, 3],
        )
        self.assertEqual(
            [choice.usage.total_tokens for choice in response.choices],
            [12, 13],
        )

    def test_stream_options_include_usage_is_explicit(self) -> None:
        default_request = self.request(enable_thinking=False)
        enabled_request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            stream=True,
            stream_options={"include_usage": True},
        )

        self.assertFalse(KimiK3Renderer._stream_includes_usage(default_request))
        self.assertTrue(KimiK3Renderer._stream_includes_usage(enabled_request))

    def test_usage_summary_is_only_emitted_when_requested(self) -> None:
        usage = UsageInfo(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        response = StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=FinisheReason.stop,
                    usage=copy.deepcopy(usage),
                )
            ],
            usage=usage,
        )

        without_summary = KimiK3Renderer._format_stream_frames(response, False)
        with_summary = KimiK3Renderer._format_stream_frames(response, True)

        self.assertEqual(len(without_summary), 1)
        self.assertEqual(len(with_summary), 2)
        self.assertEqual(with_summary[-1].choices, [])
        self.assertEqual(with_summary[-1].usage, usage)

    def test_non_thinking_terminal_envelope_is_removed(self) -> None:
        status = self.status(False)
        delta = KimiK3Renderer._parse_xtml_delta(
            status,
            "ANSWER: D<|close|>response<|sep|><|close|>message<|sep|>",
        )

        self.assertIsNone(delta.reasoning_content)
        self.assertEqual(delta.content, "ANSWER: D")
        self.assertTrue(status.xtml_decoder.response_closed)

    def test_official_thinking_disabled_uses_response_channel(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            thinking={"type": "disabled"},
        )
        options = K3RequestOptions.from_request(request)
        status = _KimiK3StreamStatus(request, thinking=options.thinking)

        self.assertFalse(options.thinking)
        self.assertFalse(status.xtml_decoder.in_reasoning)
        self.assertFalse(
            KimiK3Renderer._template_kwargs(options, request.model_dump(mode="json"))[
                "thinking"
            ]
        )

    def test_structured_output_uses_response_channel(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            response_format={"type": "json_object"},
        )
        options = K3RequestOptions.from_request(request)
        status = _KimiK3StreamStatus(request, thinking=options.thinking)
        delta = KimiK3Renderer._parse_xtml_delta(status, '{"answer":true}', flush=True)

        self.assertFalse(options.thinking)
        self.assertIsNone(delta.reasoning_content)
        self.assertEqual(delta.content, '{"answer":true}')

    def test_explicit_reasoning_effort_overrides_structured_output_default(
        self,
    ) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            response_format={"type": "json_object"},
            reasoning_effort="max",
        )

        options = K3RequestOptions.from_request(request)
        kwargs = KimiK3Renderer._template_kwargs(
            options, request.model_dump(exclude_none=True, mode="json")
        )

        self.assertTrue(kwargs["thinking"])
        self.assertTrue(options.thinking)
        self.assertEqual(kwargs["thinking_effort"], "max")

    def test_response_status_reuses_generate_config_thinking_mode(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        request = self.request(enable_thinking=False)
        config = GenerateConfig(in_think_mode=True)

        thinking = renderer._response_thinking_enabled(request, config)
        statuses = asyncio.run(
            renderer._create_response_status_list(2, request, thinking)
        )

        self.assertTrue(thinking)
        self.assertEqual(len(statuses), 2)
        self.assertTrue(all(status.xtml_decoder.in_reasoning for status in statuses))

    def test_reasoning_effort_accepts_k3_native_levels(self) -> None:
        for effort in ("low", "high", "max"):
            with self.subTest(effort=effort):
                request = ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="question")],
                    reasoning_effort=effort,
                )
                options = K3RequestOptions.from_request(request)

                kwargs = KimiK3Renderer._template_kwargs(
                    options, request.model_dump(exclude_none=True, mode="json")
                )

                self.assertTrue(kwargs["thinking"])
                self.assertEqual(kwargs["thinking_effort"], effort)

    def test_non_native_reasoning_effort_is_rejected(self) -> None:
        for effort in ("none", "minimal", "medium", "xhigh", "invalid"):
            with self.subTest(effort=effort):
                request = ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="question")],
                    reasoning_effort=effort,
                )

                with self.assertRaisesRegex(
                    FtRuntimeException,
                    "'low', 'high', 'max'",
                ):
                    K3RequestOptions.from_request(request)

    def test_invalid_reasoning_effort_is_rejected_when_thinking_is_disabled(
        self,
    ) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            thinking={"type": "disabled"},
            reasoning_effort="none",
        )

        with self.assertRaisesRegex(
            FtRuntimeException,
            "'low', 'high', 'max'",
        ):
            K3RequestOptions.from_request(request)

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
        options = K3RequestOptions.from_request(request)

        kwargs = KimiK3Renderer._template_kwargs(
            options, request.model_dump(exclude_none=True, mode="json")
        )
        self.assertEqual(kwargs["thinking_effort"], "low")

    def test_request_options_merge_is_shared_by_render_and_constraints(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "question"}],
                "chat_template_kwargs": {"thinking": False, "request_key": 1},
                "extra_configs": {
                    "chat_template_kwargs": {"extra_key": 2},
                },
            }
        )
        options = K3RequestOptions.from_request(request)
        kwargs = KimiK3Renderer._template_kwargs(
            options, request.model_dump(exclude_none=True, mode="json")
        )
        rendered_inputs = RenderedInputs(input_ids=[], renderer_context=options)
        config = GenerateConfig(in_think_mode=True, max_thinking_tokens=42)
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)

        with patch.object(
            K3RequestOptions,
            "from_request",
            side_effect=AssertionError("request options were parsed twice"),
        ):
            renderer.apply_rendered_chat_completion_constraints(
                request, config, rendered_inputs
            )

        self.assertFalse(kwargs["thinking"])
        self.assertEqual(kwargs["request_key"], 1)
        self.assertEqual(kwargs["extra_key"], 2)
        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.max_thinking_tokens, 0)

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
        renderer.vit_config = VitConfig()

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
            def encode(self, text: str) -> list[int]:
                return {
                    "<|open|>response<|sep|>": [10, 11, 13],
                    "<|open|>think<|sep|>": [10, 12, 13],
                }[text]

        tokenizer = Tokenizer()
        pending_tokens = KimiK3Renderer._pending_prompt_token_count(
            tokenizer, thinking=False
        )
        response = StreamResponseObject(
            usage=UsageInfo(
                prompt_tokens=39,
                completion_tokens=7,
                total_tokens=46,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=38),
            )
        )

        KimiK3Renderer._subtract_pending_prompt_tokens(response, pending_tokens)

        usage = response.usage
        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(pending_tokens, 3)
        self.assertEqual(usage.prompt_tokens, 36)
        self.assertEqual(usage.completion_tokens, 7)
        self.assertEqual(usage.total_tokens, 43)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 36)

    def test_invalid_channel_marker_remains_visible(self) -> None:
        status = self.status(False)
        text = "answer<|close|>think<|sep|>still-visible"
        delta = KimiK3Renderer._parse_xtml_delta(status, text, flush=True)

        self.assertEqual(delta.content, text)
        self.assertFalse(status.xtml_decoder.response_closed)

    def _run_chunks(self, status, chunks):
        deltas = []
        for index, chunk in enumerate(chunks):
            deltas.extend(
                KimiK3Renderer._split_exclusive_delta(
                    KimiK3Renderer._parse_xtml_delta(
                        status, chunk, flush=index == len(chunks) - 1
                    )
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
        status = self.status(False)
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
        self.assertEqual(status.xtml_decoder.tool_calls_seen, 1)
        self.assertEqual("".join(d.content or "" for d in deltas), "")

    def test_tools_channel_typed_arguments_are_coerced(self) -> None:
        status = self.status(False)
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
        status = self.status(False)
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
        status = self.status(False)
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
        status = self.status(False)
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
        status = self.status(False)
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
        status = self.status(True)
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
        status = self.status(False)
        deltas = self._run_chunks(
            status, ["ANSWER<|close|>response<|sep|><|end_of_msg|>"]
        )

        self.assertEqual(self._all_tool_calls(deltas), [])
        self.assertEqual(status.xtml_decoder.tool_calls_seen, 0)
        self.assertEqual("".join(d.content or "" for d in deltas), "ANSWER")


if __name__ == "__main__":
    unittest.main()
