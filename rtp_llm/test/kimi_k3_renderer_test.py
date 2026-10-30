#!/usr/bin/env python3

import asyncio
import copy
import json
import re
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import xgrammar as xgr

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.models.kimi_k3.kimi_k3_grammar import THINK_OPEN
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
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject, ThinkStatus
from rtp_llm.openai.renderers.kimi_k3_renderer import (
    K3RequestOptions,
    KimiK3Renderer,
    _K3StreamResponseObject,
    _KimiK3StreamStatus,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput


class KimiK3RendererTest(unittest.TestCase):
    def request(self, enable_thinking: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            enable_thinking=enable_thinking,
        )

    def status(self, thinking: bool) -> _KimiK3StreamStatus:
        return _KimiK3StreamStatus(self.request(thinking), thinking=thinking)

    @staticmethod
    def _decode_delta(status, text: str, *, flush: bool = False) -> DeltaMessage:
        return status.xtml_decoder.decode(text, None, flush=flush).delta

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
                KimiK3Renderer._split_exclusive_delta(self._decode_delta(status, chunk))
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

        delta = self._decode_delta(status, "")

        self.assertEqual(delta.model_dump(exclude_none=True), {})

    def test_transition_without_reasoning_does_not_emit_early_boundary(self) -> None:
        status = self.status(True)

        deltas = KimiK3Renderer._split_exclusive_delta(
            self._decode_delta(status, KimiK3Renderer._THINK_TO_RESPONSE + "answer")
        )

        self.assertEqual(
            [delta.model_dump(exclude_none=True) for delta in deltas],
            [{"content": "answer"}],
        )

    def test_transition_after_reasoning_emits_boundary_before_content(self) -> None:
        status = self.status(True)
        reasoning = self._decode_delta(status, "reasoning")
        transition = self._decode_delta(
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
                    delta=DeltaMessage(reasoning_content="reasoning", content="answer"),
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
                    delta=DeltaMessage(reasoning_content="reasoning", content="answer"),
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
        self.assertEqual(frames[0].choices[0].logprobs.content, [reasoning_logprob])
        self.assertIsNone(frames[1].choices[0].logprobs)
        self.assertEqual(frames[2].choices[0].logprobs.content, [content_logprob])

    def test_logprobs_state_machine_filters_split_xtml_markers(self) -> None:
        status = self.status(True)
        reasoning_logprob = self._logprob("reasoning", -0.2)
        content_logprob = self._logprob("answer", -0.3)

        first = status.xtml_decoder.decode(
            "",
            [reasoning_logprob, self._logprob("<|close|>")],
        ).channel_logprobs
        second = status.xtml_decoder.decode(
            "",
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
        ).channel_logprobs

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

    def test_multi_choice_final_is_split_with_per_choice_and_aggregate_usage(
        self,
    ) -> None:
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
        self.assertEqual([frame.usage for frame in frames], [None, aggregate])
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

    def test_split_frames_preserve_step_usage_for_frontend_accounting(self) -> None:
        usage = UsageInfo(prompt_tokens=10, completion_tokens=3, total_tokens=13)
        response = StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        reasoning_content="think",
                        content="answer",
                    ),
                )
            ],
            usage=usage,
        )

        frames = KimiK3Renderer._split_exclusive_response(response)

        self.assertGreater(len(frames), 1)
        self.assertEqual(
            [frame.usage for frame in frames],
            [None] * (len(frames) - 1) + [usage],
        )
        self.assertEqual(frames[-1].usage.completion_tokens, 3)

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
                has_real_aux_info=True,
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

    def test_empty_choice_usage_summary_is_additional_when_requested(self) -> None:
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
        self.assertEqual(without_summary[0].usage, usage)
        self.assertEqual(with_summary[0].usage, usage)
        self.assertEqual(with_summary[-1].choices, [])
        self.assertEqual(with_summary[-1].usage, usage)

    def test_non_thinking_terminal_envelope_is_removed(self) -> None:
        status = self.status(False)
        delta = self._decode_delta(
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
        self.assertFalse(KimiK3Renderer._template_kwargs(options, request)["thinking"])

    def test_tool_collection_does_not_serialize_conversation(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            functions=[
                {
                    "name": "lookup",
                    "description": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
        )
        with patch.object(
            ChatCompletionRequest,
            "model_dump",
            side_effect=AssertionError("conversation serialization"),
        ):
            tools = KimiK3Renderer._all_tools(request)
        self.assertEqual(tools[0]["function"]["name"], "lookup")

    def test_structured_output_uses_response_channel(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            response_format={"type": "json_object"},
        )
        options = K3RequestOptions.from_request(request)
        status = _KimiK3StreamStatus(request, thinking=options.thinking)
        delta = self._decode_delta(status, '{"answer":true}', flush=True)

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
        kwargs = KimiK3Renderer._template_kwargs(options, request)

        self.assertTrue(kwargs["thinking"])
        self.assertTrue(options.thinking)
        self.assertEqual(kwargs["thinking_effort"], "max")

    def test_response_status_reuses_generate_config_thinking_mode(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        request = self.request(enable_thinking=True)
        statuses = asyncio.run(renderer._create_status_list(2, request))

        self.assertEqual(renderer.resolve_thinking_mode(request), ThinkingMode.ENABLED)
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

                kwargs = KimiK3Renderer._template_kwargs(options, request)

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

    def test_integer_reasoning_effort_is_rejected_by_the_k3_contract(self):
        for thinking in ("enabled", "disabled"):
            with self.subTest(thinking=thinking):
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "question"}],
                        "reasoning_effort": 1,
                        "thinking": {"type": thinking},
                    }
                )
                self.assertEqual(request.reasoning_effort, 1)
                with self.assertRaisesRegex(FtRuntimeException, "must be a string"):
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

        renderer._prepare_request_constraints(
            self.request(enable_thinking=True), config
        )

        # No grammar constraint → end_think_token_ids stays default (empty)
        self.assertEqual(config.end_think_token_ids, [])
        renderer.tokenizer.encode.assert_not_called()

    def test_openai_completion_limit_distinguishes_default_and_explicit_value(
        self,
    ) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        for value, expected in ((None, 131072), (128, 128), (1048576, 131072)):
            with self.subTest(value=value):
                request_data = {
                    "messages": [{"role": "user", "content": "question"}],
                    "enable_thinking": True,
                }
                if value is not None:
                    request_data["max_completion_tokens"] = value
                request = ChatCompletionRequest.model_validate(request_data)
                config = GenerateConfig(max_new_tokens=value or 32000)

                renderer._prepare_request_constraints(request, config)

                self.assertEqual(config.max_new_tokens, expected)

    def test_openai_completion_limit_rejects_explicit_non_positive_values(
        self,
    ) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        for name in ("max_completion_tokens", "max_tokens"):
            for value in (0, -1):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(
                    FtRuntimeException, name
                ):
                    renderer._prepare_request_constraints(
                        ChatCompletionRequest.model_validate(
                            {
                                "messages": [{"role": "user", "content": "question"}],
                                name: value,
                            }
                        ),
                        GenerateConfig(),
                    )

    def test_thinking_effort_precedes_legacy_reasoning_effort(self) -> None:
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            thinking={"type": "enabled", "effort": "low"},
            reasoning_effort="max",
        )
        options = K3RequestOptions.from_request(request)

        kwargs = KimiK3Renderer._template_kwargs(options, request)
        self.assertEqual(kwargs["thinking_effort"], "low")

    def test_thinking_effort_accepts_k3_native_levels(self) -> None:
        for effort in ("low", "high", "max"):
            with self.subTest(effort=effort):
                request = ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="question")],
                    thinking={"type": "enabled", "effort": effort},
                )
                options = K3RequestOptions.from_request(request)
                self.assertEqual(options.template_kwargs["thinking_effort"], effort)

    def test_thinking_effort_values_are_validated_by_k3(self) -> None:
        for mode in ("enabled", "disabled"):
            for effort in ("medium", "xhigh", "invalid", 75):
                with self.subTest(mode=mode, effort=effort):
                    request = ChatCompletionRequest(
                        messages=[ChatMessage(role="user", content="question")],
                        thinking={"type": mode, "effort": effort},
                    )
                    self.assertEqual(request.thinking.effort, effort)
                    with self.assertRaisesRegex(FtRuntimeException, "thinking.effort"):
                        K3RequestOptions.from_request(request)

    def test_request_options_merge_controls_render_and_constraints(self) -> None:
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
        kwargs = KimiK3Renderer._template_kwargs(options, request)
        config = GenerateConfig(in_think_mode=True, max_thinking_tokens=42)
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)

        renderer._prepare_request_constraints(request, config)

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
                "parallel_tool_calls": False,
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig()

        renderer._prepare_request_constraints(request, config)

        tag = config.structural_tag
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
                "parallel_tool_calls": False,
                "thinking": {"type": "disabled"},
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig()

        renderer._prepare_request_constraints(request, config)

        tags = config.structural_tag["format"]["content"]["tags"]
        self.assertEqual(len(tags), 1)
        self.assertIn('tool="search"', tags[0]["begin"])

    def test_parallel_required_tool_choice_uses_consecutive_xtml_indices(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather in four cities?"}],
                "tools": [self._weather_tool(), self._weather_tool("search")],
                "tool_choice": "required",
                "parallel_tool_calls": True,
                "thinking": {"type": "disabled"},
            }
        )
        config = GenerateConfig()
        KimiK3Renderer.__new__(KimiK3Renderer)._prepare_request_constraints(
            request, config
        )

        content = config.structural_tag["format"]["content"]
        for index in range(1, 5):
            self.assertEqual(content["type"], "sequence")
            choice = content["elements"][0]
            self.assertEqual(choice["type"], "or")
            self.assertEqual(len(choice["elements"]), 2)
            for tag in choice["elements"]:
                self.assertIn(f'index="{index}"', tag["begin"])
            content = content["elements"][1]["content"]

    def test_thinking_grammar_switches_after_full_xtml_boundary(self) -> None:
        cases = {
            "required_tool": {
                "tools": [self._weather_tool()],
                "tool_choice": "required",
                "thinking": {"type": "enabled"},
            },
            "json_response": {
                "response_format": {"type": "json_object"},
                "reasoning_effort": "max",
            },
        }
        for name, fields in cases.items():
            with self.subTest(constraint=name):
                request = ChatCompletionRequest.model_validate(
                    {"messages": [{"role": "user", "content": "question"}], **fields}
                )
                renderer = KimiK3Renderer.__new__(KimiK3Renderer)
                renderer.tokenizer = Mock()
                renderer.tokenizer.encode.return_value = [101, 102, 103]
                renderer.generate_env_config = SimpleNamespace(
                    think_mode="disabled",
                    think_end_token_id=-1,
                    think_end_tag=KimiK3Renderer._THINK_TO_RESPONSE,
                )
                if name == "required_tool":
                    config = GenerateConfig(max_thinking_tokens=100)
                else:
                    config = GenerateConfig(response_format=request.response_format)

                renderer.try_finalize_chat_completion_response_format(
                    request, config, enable_thinking=True, input_ids=None
                )

                self.assertEqual(config.end_think_token_ids, [101, 102, 103])
                boundary = KimiK3Renderer._THINK_TO_RESPONSE
                renderer.tokenizer.encode.assert_called_with(boundary)
                elements = config.structural_tag["format"]["elements"]
                if name == "required_tool":
                    self.assertEqual(
                        renderer.tokenizer.encode.call_args_list,
                        [call(THINK_OPEN), call(boundary)],
                    )
                    self.assertEqual(elements[0]["end"], "<|close|>think<|sep|>")
                    self.assertEqual(elements[1]["type"], "tag")
                else:
                    self.assertEqual(elements[1]["content"]["type"], "json_schema")

    def test_omitted_thinking_enables_forced_tool_constraint(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [self._weather_tool()],
                "tool_choice": "required",
            }
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        config = GenerateConfig(in_think_mode=False)

        renderer._prepare_request_constraints(request, config)

        self.assertTrue(config.in_think_mode)
        self.assertEqual(config.end_think_token_ids, [])
        self.assertEqual(config.structural_tag["type"], "structural_tag")

    def test_forced_tool_choice_overrides_explicit_response_format(self) -> None:
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        request_fields = {
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [self._weather_tool()],
            "tool_choice": "required",
            "thinking": {"type": "disabled"},
        }
        schema = {"type": "object"}
        with self.assertRaisesRegex(
            FtRuntimeException, "conflicts with existing grammar constraint"
        ):
            renderer._prepare_request_constraints(
                ChatCompletionRequest.model_validate(request_fields),
                GenerateConfig(json_schema=schema),
            )

        request = ChatCompletionRequest.model_validate(
            {**request_fields, "response_format": {"type": "json_object"}}
        )
        config = GenerateConfig(json_schema=json.dumps(schema))
        renderer._prepare_request_constraints(request, config)
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
        delta = self._decode_delta(status, text, flush=True)

        self.assertEqual(delta.content, text)
        self.assertFalse(status.xtml_decoder.response_closed)

    def _run_chunks(self, status, chunks):
        deltas = []
        for index, chunk in enumerate(chunks):
            deltas.extend(
                KimiK3Renderer._split_exclusive_delta(
                    self._decode_delta(status, chunk, flush=index == len(chunks) - 1)
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
        self.assertRegex(calls[0].id, r"^get_weather_0_[0-9a-f]{8}$")
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
        self.assertEqual([int(c.id.rsplit("_", 2)[1]) for c in calls], [0, 1])
        self.assertEqual(len({c.id.rsplit("_", 1)[1] for c in calls}), 2)

    def test_tool_ids_continue_global_history_across_legacy_and_modern_formats(self):
        """The cases follow K3 Chimp acceptance execution 90494's ID contract."""
        cases = [
            ([], 0),
            (["get_weather_0", "get_air_quality_1", "get_weather_3"], 4),
            (["get_weather_0_aaaaaaaa", "get_weather_1_bbbbbbbb"], 2),
            (["get_weather:1", "functions.get_air_quality:2", "get_weather_4"], 5),
            (["get_weather_0_aaaabbbb", "get_air_quality_9_ccccdddd"], 10),
            (
                ["get_weather_999999999999999999999999_abcdef12"],
                1000000000000000000000000,
            ),
            (
                [
                    "get_weather_xx_zzzzzzzz",
                    "get_air_quality_1.5_1234abcd",
                    "get_weather_0_12345",
                    "other_tool_9_deadbeef",
                ],
                0,
            ),
        ]
        names = ["get_weather", "get_air_quality", "get_weather", "get_local_time"]
        tool_body = "".join(
            f'<|open|>call tool="{name}" index="{index}"<|sep|>'
            '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
            for index, name in enumerate(names, 1)
        )
        text = (
            "<|close|>response<|sep|><|open|>tools<|sep|>"
            + tool_body
            + "<|close|>tools<|sep|>"
        )

        for history_ids, start_index in cases:
            with self.subTest(history_ids=history_ids):
                history = []
                if history_ids:
                    history.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                self._history_call(
                                    call_id,
                                    name=(
                                        "get_air_quality"
                                        if "air_quality" in call_id
                                        else "get_weather"
                                    ),
                                )
                                for call_id in history_ids
                            ],
                        }
                    )
                    history.extend(
                        {"role": "tool", "tool_call_id": call_id, "content": "ok"}
                        for call_id in history_ids
                    )
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [
                            {"role": "user", "content": "question"},
                            *history,
                            {"role": "user", "content": "continue"},
                        ],
                        "enable_thinking": False,
                    }
                )
                status = _KimiK3StreamStatus(request, thinking=False)
                calls = self._all_tool_calls(self._run_chunks(status, [text]))
                self.assertEqual(len(calls), len(names))
                suffixes = set()
                for offset, (name, call) in enumerate(zip(names, calls)):
                    match = re.fullmatch(
                        rf"{re.escape(name)}_([0-9]+)_([0-9a-f]{{8}})", call.id
                    )
                    self.assertIsNotNone(match, call.id)
                    self.assertEqual(int(match.group(1)), start_index + offset)
                    self.assertNotIn(match.group(2), suffixes)
                    self.assertNotIn(call.id, history_ids)
                    suffixes.add(match.group(2))

    def test_tool_id_suffix_does_not_reuse_history_hex(self):
        history_id = "get_weather_3_aaaaaaaa"
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [self._history_call(history_id, "get_weather")],
                    },
                    {"role": "tool", "tool_call_id": history_id, "content": "ok"},
                    {"role": "user", "content": "continue"},
                ],
                "enable_thinking": False,
            }
        )
        status = _KimiK3StreamStatus(request, thinking=False)
        text = (
            "<|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{}<|close|>json<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|>"
        )
        with patch(
            "rtp_llm.openai.renderers.kimi_k3_renderer.uuid.uuid4",
            side_effect=[
                SimpleNamespace(hex="aaaaaaaa000000000000000000000000"),
                SimpleNamespace(hex="bbbbbbbb000000000000000000000000"),
            ],
        ):
            calls = self._all_tool_calls(self._run_chunks(status, [text]))
        self.assertEqual([call.id for call in calls], ["get_weather_4_bbbbbbbb"])

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

    def test_thinking_control_precedence_ignores_conflicting_lower_priorities(self):
        cases = (
            ({}, True),
            ({"response_format": {"type": "text"}}, True),
            ({"response_format": {"type": "json_object"}}, False),
            ({"json_format": True}, False),
            ({"reasoning_effort": " HIGH ", "json_format": True}, True),
            ({"thinking_budget": 0, "reasoning_effort": "high"}, False),
            (
                {
                    "extra_configs": {"max_thinking_tokens": 0},
                    "reasoning_effort": "high",
                },
                False,
            ),
            (
                {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "thinking_budget": 0,
                },
                True,
            ),
            (
                {"chat_template_kwargs": {"thinking": False, "enable_thinking": True}},
                False,
            ),
            (
                {"enable_thinking": False, "chat_template_kwargs": {"thinking": True}},
                False,
            ),
            (
                {"enable_thinking": True, "thinking_budget": 0, "json_format": True},
                True,
            ),
            ({"thinking": {"type": "enabled"}, "enable_thinking": False}, True),
            ({"thinking": {"type": "disabled"}, "enable_thinking": True}, False),
            ({"chat_template_kwargs": {"thinking_mode": "disabled"}}, False),
            ({"extra_configs": {"thinking_mode": ThinkingMode.DISABLED}}, False),
            (
                {
                    "chat_template_kwargs": {"thinking_mode": "disabled"},
                    "extra_configs": {"chat_template_kwargs": {"topical": True}},
                },
                False,
            ),
            (
                {
                    "chat_template_kwargs": {"thinking_mode": "enabled"},
                    "json_format": True,
                },
                True,
            ),
            (
                {
                    "chat_template_kwargs": {"thinking_mode": "enabled"},
                    "extra_configs": {"thinking_mode": ThinkingMode.DISABLED},
                },
                False,
            ),
            (
                {
                    "chat_template_kwargs": {"thinking_mode": "disabled"},
                    "enable_thinking": True,
                },
                True,
            ),
        )
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        for controls, expected in cases:
            with self.subTest(controls=controls):
                request = ChatCompletionRequest.model_validate(
                    {"messages": [{"role": "user", "content": "question"}], **controls}
                )
                options = K3RequestOptions.from_request(request)
                self.assertEqual(options.thinking, expected)
                self.assertEqual(
                    renderer.resolve_thinking_mode(request),
                    ThinkingMode.ENABLED if expected else ThinkingMode.DISABLED,
                )
                self.assertEqual(options.template_kwargs["thinking"], expected)
                self.assertNotIn("enable_thinking", options.template_kwargs)
                self.assertNotIn("thinking_mode", options.template_kwargs)

    def test_template_controls_merge_without_mutating_the_request(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "question"}],
                "chat_template_kwargs": {
                    "thinking": True,
                    "thinking_effort": "stale",
                    "source": "top",
                    "top_only": 1,
                },
                "extra_configs": {
                    "chat_template_kwargs": {"thinking": False, "source": "nested"}
                },
            }
        )
        original = request.model_dump()
        options = K3RequestOptions.from_request(request)
        kwargs = KimiK3Renderer._template_kwargs(options, request)
        self.assertEqual(kwargs, {"thinking": False, "source": "nested", "top_only": 1})
        self.assertEqual(request.model_dump(), original)
        with self.assertRaises(TypeError):
            options.template_kwargs["thinking"] = True
        kwargs["source"] = "changed"
        self.assertEqual(options.template_kwargs["source"], "nested")

    def test_both_template_thinking_aliases_reject_non_boolean_values(self):
        for field in ("thinking", "enable_thinking"):
            for value in (0, 1, "false", None, []):
                with self.subTest(field=field, value=value):
                    request = ChatCompletionRequest.model_validate(
                        {
                            "messages": [{"role": "user", "content": "question"}],
                            "chat_template_kwargs": {field: value},
                        }
                    )
                    with self.assertRaisesRegex(
                        FtRuntimeException, "must be a boolean"
                    ):
                        K3RequestOptions.from_request(request)

    @staticmethod
    def _renderer_with_tokenizer():
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        renderer.tokenizer = Mock()
        renderer.tokenizer.apply_chat_template.return_value = [101, 102]
        renderer.max_seq_len = 0
        renderer.vit_config = VitConfig()
        renderer.stop_words_id_list = []
        renderer.extra_stop_words = []
        renderer.extra_stop_word_ids_list = []
        return renderer

    def test_dynamic_tool_validation_rejects_malformed_declarations_before_render(self):
        valid_tool = self._weather_tool()
        cases = [
            ({"role": "user", "content": "", "tools": [valid_tool]}, "system message"),
            (
                {"role": "system", "content": "not empty", "tools": [valid_tool]},
                "empty content",
            ),
            (
                {"role": "system", "tools": [{"type": "other", "function": {}}]},
                "function object",
            ),
            (
                {"role": "system", "tools": [{"type": "function", "function": None}]},
                "function object",
            ),
        ]
        for name in (None, "", "bad-name", "1bad", "x" * 257):
            tool = copy.deepcopy(valid_tool)
            tool["function"]["name"] = name
            cases.append(({"role": "system", "tools": [tool]}, "name is invalid"))
        for parameters in (None, [], "object"):
            tool = copy.deepcopy(valid_tool)
            tool["function"]["parameters"] = parameters
            cases.append(
                ({"role": "system", "tools": [tool]}, "parameters must be an object")
            )
        for strict in (0, "true", None):
            tool = copy.deepcopy(valid_tool)
            tool["function"]["strict"] = strict
            cases.append(
                ({"role": "system", "tools": [tool]}, "strict must be a boolean")
            )
        for message, error in cases:
            with self.subTest(message=message):
                request = ChatCompletionRequest.model_validate({"messages": [message]})
                renderer = self._renderer_with_tokenizer()
                with self.assertRaisesRegex(FtRuntimeException, error):
                    renderer.render_chat(request)
                renderer.tokenizer.apply_chat_template.assert_not_called()

    def test_dynamic_tool_name_uniqueness_across_messages_and_global_tools(self):
        tool = self._weather_tool()
        for request_data in (
            {
                "tools": [tool, tool],
                "messages": [{"role": "user", "content": "question"}],
            },
            {"tools": [tool], "messages": [{"role": "system", "tools": [tool]}]},
            {
                "messages": [
                    {"role": "system", "tools": [tool]},
                    {"role": "system", "tools": [tool]},
                ]
            },
        ):
            with self.subTest(request=request_data):
                renderer = self._renderer_with_tokenizer()
                request = ChatCompletionRequest.model_validate(request_data)
                with self.assertRaisesRegex(
                    FtRuntimeException, "tool names must be unique"
                ):
                    renderer.render_chat(request)
                renderer.tokenizer.apply_chat_template.assert_not_called()

    def test_dynamic_tools_allow_empty_system_content_and_boundary_names(self):
        for content in (None, "", []):
            for strict in (True, False):
                with self.subTest(content=content, strict=strict):
                    tool = self._weather_tool("_" + "x" * 255)
                    tool["function"]["strict"] = strict
                    request = ChatCompletionRequest.model_validate(
                        {
                            "messages": [
                                {"role": "system", "content": content, "tools": [tool]}
                            ]
                        }
                    )
                    renderer = self._renderer_with_tokenizer()
                    self.assertEqual(
                        renderer.render_chat(request).input_ids, [101, 102]
                    )
                    messages = renderer.tokenizer.apply_chat_template.call_args.args[0]
                    self.assertEqual(messages[0]["tools"], [tool])

    def test_tools_take_precedence_over_legacy_functions_without_request_mutation(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "question"}],
                "tools": [self._weather_tool("modern")],
                "functions": [self._weather_tool("legacy")["function"]],
            }
        )
        original = request.model_dump()
        tools = KimiK3Renderer._all_tools(request)
        self.assertEqual([tool["function"]["name"] for tool in tools], ["modern"])
        tools[0]["function"]["name"] = "mutated"
        self.assertEqual(request.model_dump(), original)

    def test_auto_none_and_omitted_choice_route_native_tools_and_parallel_policy(self):
        for choice in (None, "auto", "none"):
            for parallel in (None, False, True):
                with self.subTest(choice=choice, parallel=parallel):
                    request = ChatCompletionRequest.model_validate(
                        {
                            "messages": [
                                {
                                    "role": "system",
                                    "tools": [self._weather_tool("dynamic")],
                                },
                                {"role": "user", "content": "question"},
                            ],
                            "functions": [self._weather_tool("legacy")["function"]],
                            "tool_choice": choice,
                            "parallel_tool_calls": parallel,
                        }
                    )
                    renderer = self._renderer_with_tokenizer()
                    config = GenerateConfig()
                    renderer._prepare_request_constraints(request, config)
                    self.assertIsNone(config.structural_tag)
                    with patch(
                        "rtp_llm.openai.renderers.kimi_k3_renderer.KimiK3Grammar"
                    ) as grammar:
                        renderer.build_chat_completion_response_format_plan(
                            request, config, [101]
                        )
                    kwargs = grammar.call_args.kwargs
                    names = [tool["function"]["name"] for tool in kwargs["tools"] or []]
                    self.assertEqual(
                        names, [] if choice == "none" else ["legacy", "dynamic"]
                    )
                    self.assertEqual(
                        kwargs["parallel_tool_calls"], parallel is not False
                    )
                    self.assertIsNone(kwargs["required_tools_tail"])
                    grammar.return_value.build_plan.assert_called_once_with(
                        config, renderer.tokenizer, [101]
                    )

    def test_forced_tools_default_parallel_matches_explicit_true_in_native_matcher(
        self,
    ):
        compiler = xgr.GrammarCompiler(
            xgr.TokenizerInfo([chr(i) for i in range(128)], stop_token_ids=[0]),
            max_threads=1,
        )
        call = (
            '<|open|>call tool="get_weather" index="{index}"<|sep|>'
            '<|open|>json type="object"<|sep|>{{"city":"Paris"}}'
            "<|close|>json<|sep|><|close|>call<|sep|>"
        )
        for choice in (
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ):
            for parallel in (None, False, True):
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "Weather?"}],
                        "tools": [self._weather_tool()],
                        "tool_choice": choice,
                        "parallel_tool_calls": parallel,
                    }
                )
                grammar = compiler.compile_structural_tag(
                    KimiK3Renderer._build_tool_call_structural_tag(request)
                )
                for count in (1, 2):
                    with self.subTest(choice=choice, parallel=parallel, count=count):
                        text = (
                            KimiK3Renderer._RESPONSE_CLOSE
                            + KimiK3Renderer._TOOLS_OPEN
                            + "".join(
                                call.format(index=index)
                                for index in range(1, count + 1)
                            )
                            + KimiK3Renderer._TOOLS_CLOSE
                        )
                        matcher = xgr.GrammarMatcher(grammar)
                        accepted = all(matcher.accept_token(ord(char)) for char in text)
                        accepted = accepted and matcher.accept_token(0)
                        self.assertEqual(accepted, count == 1 or parallel is not False)

    def test_legacy_and_dynamic_tool_names_must_be_unique(self):
        tool = self._weather_tool("lookup")
        for data in (
            {
                "messages": [{"role": "system", "tools": [tool]}],
                "functions": [tool["function"]],
            },
            {
                "messages": [{"role": "user", "content": "question"}],
                "functions": [tool["function"], tool["function"]],
            },
        ):
            with self.subTest(request=data):
                request = ChatCompletionRequest.model_validate(data)
                renderer = self._renderer_with_tokenizer()
                with self.assertRaisesRegex(
                    FtRuntimeException, "tool names must be unique"
                ):
                    renderer.render_chat(request)
                renderer.tokenizer.apply_chat_template.assert_not_called()

    def test_unknown_named_choice_is_rejected_with_legacy_or_dynamic_tools(self):
        for source in ("legacy", "dynamic"):
            with self.subTest(source=source):
                tool = self._weather_tool("lookup")
                data = {
                    "messages": [{"role": "user", "content": "question"}],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "missing"},
                    },
                }
                if source == "legacy":
                    data["functions"] = [tool["function"]]
                else:
                    data["messages"].insert(0, {"role": "system", "tools": [tool]})
                with self.assertRaisesRegex(ValueError, "not in tools"):
                    ChatCompletionRequest.model_validate(data)

    def test_partial_assistant_and_image_placeholders_reach_the_native_template(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/a.png"},
                                "preprocess_config": {
                                    "mm_timeout_ms": 8500,
                                    "resized_width": 64,
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/b.png"},
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "The first image",
                        "partial": True,
                    },
                ],
                "thinking": {"type": "disabled"},
            }
        )
        original = request.model_dump()
        renderer = self._renderer_with_tokenizer()
        rendered = renderer.render_chat(request)
        args, kwargs = renderer.tokenizer.apply_chat_template.call_args
        self.assertEqual(
            args[0][-1],
            {"role": "assistant", "content": "The first image", "partial": True},
        )
        self.assertEqual(kwargs["image_prompts"], ["<|media_pad|>", "<|media_pad|>"])
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertTrue(kwargs["tokenize"])
        self.assertEqual(
            [item.url for item in rendered.multimodal_inputs],
            ["https://example.com/a.png", "https://example.com/b.png"],
        )
        self.assertEqual(
            [
                item.mm_preprocess_config.mm_timeout_ms
                for item in rendered.multimodal_inputs
            ],
            [8500, -1],
        )
        self.assertEqual(
            [item.mm_preprocess_config.width for item in rendered.multimodal_inputs],
            [-1, -1],
        )
        self.assertEqual(request.model_dump(), original)

    def test_xtml_content_reasoning_and_tools_are_invariant_at_every_split(self):
        call = (
            '<|open|>call tool="search" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{"query":"北京","nested":[1,true]}'
            "<|close|>json<|sep|><|close|>call<|sep|>"
        )
        for thinking in (False, True):
            prefix = "分析🙂" + KimiK3Renderer._THINK_TO_RESPONSE if thinking else ""
            text = (
                prefix
                + "回答"
                + KimiK3Renderer._RESPONSE_CLOSE
                + KimiK3Renderer._TOOLS_OPEN
                + call
                + KimiK3Renderer._TOOLS_CLOSE
            )
            chunkings = [[text[:cut], text[cut:]] for cut in range(len(text) + 1)]
            chunkings.append(list(text))
            for chunks in chunkings:
                with self.subTest(thinking=thinking, lengths=list(map(len, chunks))):
                    status = self.status(thinking)
                    deltas = self._run_chunks(status, chunks)
                    self.assertEqual(
                        "".join(d.reasoning_content or "" for d in deltas),
                        "分析🙂" if thinking else "",
                    )
                    self.assertEqual("".join(d.content or "" for d in deltas), "回答")
                    calls = self._all_tool_calls(deltas)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].index, 0)
                    self.assertEqual(calls[0].function.name, "search")
                    self.assertEqual(
                        json.loads(calls[0].function.arguments),
                        {"query": "北京", "nested": [1, True]},
                    )
                    self.assertEqual(status.xtml_decoder.tool_calls_seen, 1)

    def test_incomplete_xtml_prefix_is_preserved_on_flush(self):
        for thinking, marker in (
            (True, KimiK3Renderer._THINK_TO_RESPONSE),
            (False, KimiK3Renderer._RESPONSE_CLOSE),
        ):
            for length in range(1, len(marker)):
                with self.subTest(thinking=thinking, length=length):
                    status = self.status(thinking)
                    deltas = self._run_chunks(status, ["body" + marker[:length], ""])
                    field = "reasoning_content" if thinking else "content"
                    self.assertEqual(
                        "".join(getattr(d, field) or "" for d in deltas),
                        "body" + marker[:length],
                    )
                    self.assertEqual(status.xtml_decoder.xtml_pending, "")

    def test_logprobs_preserve_incomplete_and_invalid_markers_on_flush(self):
        for tokens in (
            ["<|close|>"],
            ["<|close|>", "response"],
            ["<|close|>", "not-a-channel", "body"],
            ["<|close|>", "response", "not-a-separator", "body"],
        ):
            for thinking in (False, True):
                with self.subTest(tokens=tokens, thinking=thinking):
                    status = self.status(thinking)
                    all_probs = []
                    for token in tokens:
                        result = status.xtml_decoder.decode("", [self._logprob(token)])
                        all_probs.extend(
                            result.channel_logprobs[
                                "reasoning_content" if thinking else "content"
                            ]
                        )
                    flushed = status.xtml_decoder.decode("", None, flush=True)
                    all_probs.extend(
                        flushed.channel_logprobs[
                            "reasoning_content" if thinking else "content"
                        ]
                    )
                    self.assertEqual([item.token for item in all_probs], tokens)
                    self.assertEqual(status.xtml_decoder.logprobs_marker_pending, [])

    def test_pending_prompt_usage_rejects_underflow_in_aggregate_or_choice(self):
        for prompt, total in ((2, 10), (10, 2)):
            for per_choice in (False, True):
                with self.subTest(prompt=prompt, total=total, per_choice=per_choice):
                    usage = UsageInfo(
                        prompt_tokens=prompt, completion_tokens=0, total_tokens=total
                    )
                    response = StreamResponseObject(
                        choices=(
                            [
                                ChatCompletionResponseStreamChoice(
                                    index=0, delta=DeltaMessage(), usage=usage
                                )
                            ]
                            if per_choice
                            else []
                        ),
                        usage=None if per_choice else usage,
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "shorter than its pending generation prompt"
                    ):
                        KimiK3Renderer._subtract_pending_prompt_tokens(response, 3)

    def test_image_parts_preserve_order_and_do_not_mutate_messages(self):
        messages = [
            {"role": "system", "content": "instructions"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/a.png"},
                    },
                    {"type": "text", "text": "second"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            },
        ]
        original = copy.deepcopy(messages)
        rewritten, mm_input = KimiK3Renderer._collect_and_rewrite(messages)
        self.assertEqual(messages, original)
        self.assertEqual(
            mm_input.urls, ["https://example.com/a.png", "data:image/png;base64,AA=="]
        )
        self.assertEqual(
            [part["type"] for part in rewritten[1]["content"]],
            ["text", "image", "text", "image"],
        )
        self.assertEqual(rewritten[1]["content"][2], {"type": "text", "text": "second"})

    def test_invalid_multimodal_content_is_rejected_before_tokenization(self):
        cases = [
            ({"role": "user", "content": ["text"]}, "must be objects"),
            (
                {"role": "user", "content": [{"type": "audio_url"}]},
                "only text and image_url",
            ),
            (
                {"role": "user", "content": [{"type": "unknown"}]},
                "only text and image_url",
            ),
        ]
        for role in ("system", "assistant", "tool"):
            cases.append(
                (
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/a.png"},
                            }
                        ],
                    },
                    "only in user",
                )
            )
        for image_url in (
            None,
            "https://example.com/a.png",
            {},
            {"url": ""},
            {"url": 123},
        ):
            cases.append(
                (
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": image_url}],
                    },
                    "non-empty URL",
                )
            )
        for message, error in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                FtRuntimeException, error
            ):
                KimiK3Renderer._collect_and_rewrite([message])

    def test_tokenizer_single_conversation_result_validation(self):
        for value in ([1, 2], [[1, 2]], []):
            with self.subTest(value=value):
                expected = [1, 2] if value else []
                self.assertEqual(KimiK3Renderer._as_token_ids(value), expected)
        with self.assertRaisesRegex(ValueError, "token batch"):
            KimiK3Renderer._as_token_ids([[1], [2]])
        for value in (None, "1 2", [1, "2"], (1, 2), [[[]]]):
            with self.subTest(value=value), self.assertRaisesRegex(
                TypeError, "List\\[int\\]"
            ):
                KimiK3Renderer._as_token_ids(value)

    def test_tokenizer_tensor_results_preserve_single_conversation_ids(self):
        for value in (torch.tensor([11, 12]), torch.tensor([[11, 12]])):
            with self.subTest(shape=tuple(value.shape)):
                self.assertEqual(KimiK3Renderer._as_token_ids(value), [11, 12])
        with self.assertRaisesRegex(ValueError, "token batch"):
            KimiK3Renderer._as_token_ids(torch.tensor([[11], [12]]))

    @staticmethod
    def _backend_output(ids, output_len, *, finished=False):
        return GenerateOutput(
            output_ids=torch.tensor([ids], dtype=torch.int32),
            finished=finished,
            aux_info=AuxInfo(input_len=10, output_len=output_len),
        )

    @staticmethod
    def _compact_backend_output(ids, output_len, *, offset=0, scores=()):
        output = KimiK3RendererTest._backend_output(ids, output_len)
        output.logprobs_offset = offset
        output.logprobs_count = len(scores)
        if scores:
            output.token_logprobs = torch.tensor(scores)
        return output

    def test_compact_logprobs_start_at_answer_after_thinking(self):
        renderer = self._renderer_with_tokenizer()
        renderer.max_seq_len = 10000
        renderer.eos_token_id = 0
        tokens = {
            1: "reasoning",
            2: "<|close|>",
            3: "think",
            4: "<|sep|>",
            5: "<|open|>",
            6: "response",
            7: "answer",
        }
        renderer.tokenizer.decode.side_effect = lambda ids: "".join(
            tokens[token_id] for token_id in ids
        )
        status = self.status(True)
        status.request.logprobs = True

        reasoning = asyncio.run(
            renderer._update_single_status(
                status,
                self._compact_backend_output([1], 1, offset=1),
                1000,
                [],
                [],
                True,
            )
        )
        self.assertEqual(reasoning.output_str.reasoning_content, "reasoning")
        self.assertFalse(reasoning.channel_logprobs["reasoning_content"])

        tokens_to_emit = [2, 3, 4, 5, 6, 4, 7, 2, 6, 4]
        answer = asyncio.run(
            renderer._update_single_status(
                status,
                self._compact_backend_output(
                    tokens_to_emit,
                    1 + len(tokens_to_emit),
                    offset=6,
                    scores=(-1.25,) * 4,
                ),
                1000,
                [],
                [],
                True,
            )
        )
        response = asyncio.run(
            renderer._generate_stream_response(
                [answer], [ThinkStatus(is_streaming=True)]
            )
        )
        boundary, frame = renderer._split_exclusive_response(response)
        self.assertEqual(boundary.choices[0].delta.reasoning_content, "")
        self.assertIsNone(boundary.choices[0].logprobs)
        self.assertEqual(frame.choices[0].delta.content, "answer")
        self.assertEqual(
            [(item.token, item.logprob) for item in frame.choices[0].logprobs.content],
            [("answer", -1.25)],
        )

    def test_compact_flush_recovers_valid_text_and_drops_incomplete_utf8(self):
        renderer = self._renderer_with_tokenizer()
        renderer.max_seq_len = 10000
        renderer.eos_token_id = 0
        renderer.tokenizer.decode.side_effect = lambda ids: "".join(
            {1: "answer", 2: "\ufffd"}[token_id] for token_id in ids
        )
        status = self.status(False)
        status.request.logprobs = True
        pending = asyncio.run(
            renderer._update_single_status(
                status,
                self._compact_backend_output([1, 2], 2, scores=(-1.25, -2.0)),
                1000,
                [],
                [],
                True,
            )
        )
        self.assertEqual(pending.output_str, "")
        response = asyncio.run(
            renderer._flush_buffer([status], [], True, [ThinkStatus(is_streaming=True)])
        )
        (frame,) = renderer._split_exclusive_response(response)
        self.assertEqual(frame.choices[0].delta.content, "answer")
        self.assertEqual(
            [item.token for item in frame.choices[0].logprobs.content], ["answer"]
        )
        self.assertFalse(status.pending_logprobs)

    def test_pending_utf8_backend_tokens_do_not_enter_the_xtml_decoder_early(self):
        renderer = self._renderer_with_tokenizer()
        renderer.max_seq_len = 10000
        renderer.eos_token_id = 0
        renderer.tokenizer.decode.side_effect = lambda ids: {
            (): "",
            (200,): "\ufffd",
            (200, 201): "汉",
        }[tuple(ids)]
        status = self.status(False)
        status.stop_words_list = [[0]]

        pending = asyncio.run(
            renderer._update_single_status(
                status, self._backend_output([200], 1), 1000, [], [], True
            )
        )
        self.assertEqual(pending.output_str, "")
        self.assertEqual(status.xtml_decoder.xtml_pending, "")
        self.assertFalse(status.xtml_decoder.response_closed)
        pending_response = asyncio.run(
            renderer._generate_stream_response(
                [pending], [ThinkStatus(is_streaming=True)]
            )
        )
        self.assertEqual(pending_response.choices[0].delta.content, "")
        self.assertEqual(pending_response.channel_logprobs, {})

        complete = asyncio.run(
            renderer._update_single_status(
                status,
                self._backend_output([201, 0], 3, finished=True),
                1000,
                [],
                [],
                True,
            )
        )
        self.assertEqual(complete.output_str.content, "汉")
        self.assertEqual(status.responded_string, "汉")
        self.assertEqual(status.finish_reason, FinisheReason.stop)

    def test_finished_backend_tool_call_promotes_stop_and_is_not_emitted_twice(self):
        renderer = self._renderer_with_tokenizer()
        renderer.max_seq_len = 10000
        renderer.eos_token_id = 0
        renderer.tokenizer.decode.side_effect = lambda ids: "".join(map(chr, ids))
        status = self.status(False)
        status.stop_words_list = [[0]]
        text = (
            KimiK3Renderer._RESPONSE_CLOSE
            + KimiK3Renderer._TOOLS_OPEN
            + '<|open|>call tool="lookup" index="1"<|sep|>'
            + '<|open|>json type="object"<|sep|>{"q":"x"}'
            + "<|close|>json<|sep|><|close|>call<|sep|>"
            + KimiK3Renderer._TOOLS_CLOSE
        )
        output = self._backend_output(
            [*map(ord, text), 0], len(text) + 1, finished=True
        )

        delta = asyncio.run(
            renderer._update_single_status(status, output, 1000, [], [], True)
        )

        self.assertEqual(status.finish_reason, FinisheReason.tool_calls)
        self.assertEqual(status.xtml_decoder.tool_calls_seen, 1)
        self.assertIsNone(delta.output_str.content)
        self.assertEqual(delta.output_str.tool_calls[0].function.name, "lookup")
        self.assertEqual(
            json.loads(delta.output_str.tool_calls[0].function.arguments), {"q": "x"}
        )
        duplicate = asyncio.run(
            renderer._update_single_status(status, output, 1000, [], [], True)
        )
        self.assertEqual(duplicate.output_str, "")
        self.assertEqual(status.xtml_decoder.tool_calls_seen, 1)


if __name__ == "__main__":
    unittest.main()
