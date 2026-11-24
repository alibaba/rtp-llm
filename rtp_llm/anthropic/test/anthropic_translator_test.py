import asyncio
import base64
import json
from types import SimpleNamespace
from unittest import TestCase, main

from pydantic import ValidationError
from rtp_llm.anthropic import (
    AnthropicCountTokensRequest,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicOutputTextBlock,
    AnthropicStreamTranslator,
    AnthropicTranslationError,
    AnthropicUsage,
    build_anthropic_usage,
    exception_to_anthropic_http,
    translate_anthropic_request,
    translate_openai_response,
)
from rtp_llm.anthropic.translator import _parse_tool_arguments
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ThinkingMode
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ContentPartTypeEnum,
    DebugInfo,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    PromptTokensDetails,
    RoleEnum,
    ToolCall,
    UsageInfo,
)
from rtp_llm.openai.renderers.v41_recipe import resolve_thinking
from rtp_llm.openai.renderers.v41_stream import V41StreamParser
from rtp_llm.utils.concurrency_controller import ConcurrencyException


def _parse_sse_events(raw_events):
    parsed = []
    for raw in raw_events:
        lines = [line for line in raw.strip().split("\n") if line]
        assert len(lines) == 2
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        parsed.append(
            (lines[0][len("event: ") :], json.loads(lines[1][len("data: ") :]))
        )
    return parsed


class AnthropicTranslatorTest(TestCase):
    def test_request_validation_and_constraints(self):
        # Missing max_tokens allowed for count_tokens, rejected for messages
        count_req = AnthropicCountTokensRequest(
            model="claude-test",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIsNone(count_req.max_tokens)

        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="claude-test",
                messages=[{"role": "user", "content": "hi"}],
            )

        # Boolean / non-positive / >INT32_MAX max_tokens rejected
        for bad_max in (0, -5, True, 2_147_483_648):
            with self.subTest(bad_max=bad_max):
                with self.assertRaises(ValidationError):
                    AnthropicMessagesRequest(
                        model="claude-test",
                        max_tokens=bad_max,
                        messages=[{"role": "user", "content": "hi"}],
                    )

        # Empty messages, invalid roles, and non-user first message rejected
        for bad_msgs in (
            [],
            [{"role": "system", "content": "nope"}],
            [{"role": "assistant", "content": "first cannot be assistant"}],
        ):
            with self.subTest(bad_msgs=bad_msgs):
                with self.assertRaises(ValidationError):
                    AnthropicMessagesRequest(
                        model="m", max_tokens=16, messages=bad_msgs
                    )

        # Misplaced block types rejected
        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="m",
                max_tokens=16,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}
                        ],
                    }
                ],
            )
        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="m",
                max_tokens=16,
                messages=[
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tu_1",
                                "content": "ok",
                            }
                        ],
                    },
                ],
            )
        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="m",
                max_tokens=16,
                messages=[
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/a.png",
                                },
                            }
                        ],
                    },
                ],
            )

        # Thinking budget checks
        for bad_thinking in (
            {"type": "enabled", "budget_tokens": 64},
            {"type": "enabled", "budget_tokens": -10},
            {"type": "enabled", "budget_tokens": 2_147_483_648},
            {"type": "disabled", "budget_tokens": 32},
            {"type": "adaptive", "budget_tokens": 0},
            {"type": "adaptive", "budget_tokens": -1},
        ):
            with self.subTest(bad_thinking=bad_thinking):
                with self.assertRaises(ValidationError):
                    AnthropicMessagesRequest(
                        model="m",
                        max_tokens=64,
                        messages=[{"role": "user", "content": "hi"}],
                        thinking=bad_thinking,
                    )

        # Numeric & sequence bounds (temperature, top_p, top_k, seed, stop_sequences)
        for bad_kwargs in (
            {"temperature": 1.1},
            {"temperature": 2.0},
            {"temperature": -0.1},
            {"temperature": float("nan")},
            {"top_p": 1.1},
            {"top_p": -0.1},
            {"top_k": -1},
            {"top_k": 2_147_483_648},
            {"seed": 2**63},
            {"seed": -(2**63) - 1},
            {"stop_sequences": [""]},
            {"stop_sequences": ["x" * 1025]},
            {"stop_sequences": [f"s{i}" for i in range(17)]},
            {"output_config": {"format": {"type": "json_schema"}}},
            {"output_config": {"format": {"type": "json_schema", "schema": {}}}},
            {"output_config": {"format": {"type": "invalid"}}},
        ):
            with self.subTest(bad_kwargs=bad_kwargs):
                with self.assertRaises(ValidationError):
                    AnthropicMessagesRequest(
                        model="m",
                        max_tokens=64,
                        messages=[{"role": "user", "content": "hi"}],
                        **bad_kwargs,
                    )

        # Tool choice checks
        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="m",
                max_tokens=64,
                messages=[{"role": "user", "content": "hi"}],
                tool_choice={"type": "any"},
            )
        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="m",
                max_tokens=64,
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"name": "weather", "input_schema": {"type": "object"}}],
                tool_choice={"type": "tool", "name": "missing"},
            )

    def test_image_source_security_validation(self):
        bad_sources = [
            {"type": "url", "url": "/etc/passwd"},
            {"type": "url", "url": "file:///etc/passwd"},
            {"type": "url", "url": "oss://internal-bucket/secret.png"},
            {"type": "url", "url": "data:image/png;base64,QUJD"},
            {"type": "url", "url": "https:///no-host.png"},
            {"type": "url", "url": "https://example.com/" + "a" * 9000},
            {"type": "base64", "media_type": "text/plain", "data": "QUJD"},
            {"type": "base64", "media_type": "image/png,evil", "data": "QUJD"},
            {"type": "base64", "media_type": "image/png", "data": "not-valid-b64!"},
            {"type": "base64", "media_type": "image/png", "data": "   "},
        ]
        for source in bad_sources:
            with self.subTest(source=source):
                with self.assertRaises(ValidationError):
                    AnthropicMessagesRequest(
                        model="m",
                        max_tokens=32,
                        messages=[
                            {
                                "role": "user",
                                "content": [{"type": "image", "source": source}],
                            }
                        ],
                    )

        good_b64 = base64.b64encode(b"png-bytes").decode("ascii")
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=32,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": f" {good_b64[:4]}\n{good_b64[4:]} ",
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/a.png",
                            },
                        },
                    ],
                }
            ],
        )
        translated = translate_anthropic_request(req)
        parts = translated.messages[0].content
        self.assertEqual(parts[0].image_url.url, f"data:image/png;base64,{good_b64}")
        self.assertEqual(parts[1].image_url.url, "https://example.com/a.png")

    def test_translate_multiturn_tool_and_system_and_thinking_and_multimodal(self):
        req = AnthropicMessagesRequest(
            model="qwen3-test",
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": "System part 1",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "System part 2"},
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect image:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "QUJD",
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/a.jpg",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Checking tools",
                            "signature": "sig",
                        },
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "Let me look that up."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Hangzhou"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": '{"temp": 22}'}],
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "Summarize now."},
                    ],
                },
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Lookup city weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tool_choice={
                "type": "tool",
                "name": "get_weather",
                "disable_parallel_tool_use": True,
            },
            thinking={"type": "enabled", "budget_tokens": 128},
            temperature=0.0,
            top_p=0.9,
            top_k=1,
            stop_sequences=["DONE"],
            output_config={
                "format": {"type": "json_schema", "schema": {"type": "object"}},
                "effort": "high",
            },
            metadata={"user_id": "user-42"},
            service_tier="auto",
        )

        openai_req = translate_anthropic_request(req)
        self.assertEqual(openai_req.model, "qwen3-test")
        self.assertEqual(openai_req.max_tokens, 512)
        self.assertEqual(openai_req.temperature, 0.0)
        self.assertEqual(openai_req.top_p, 0.9)
        self.assertEqual(openai_req.top_k, 1)
        self.assertEqual(openai_req.stop, ["DONE"])
        self.assertEqual(openai_req.user, "user-42")
        self.assertTrue(openai_req.enable_thinking)
        self.assertEqual(openai_req.thinking_budget, 128)
        self.assertEqual(openai_req.chat_template_kwargs, {"enable_thinking": True})
        self.assertIsNone(openai_req.extra_configs)
        self.assertEqual(openai_req.get_explicit_thinking_mode(), ThinkingMode.ENABLED)
        self.assertFalse(openai_req.parallel_tool_calls)
        self.assertEqual(openai_req.reasoning_effort, "high")
        self.assertEqual(
            openai_req.tool_choice,
            {"type": "function", "function": {"name": "get_weather"}},
        )
        self.assertEqual(openai_req.response_format.type, "json_schema")
        self.assertEqual(
            openai_req.response_format.json_schema.schema_, {"type": "object"}
        )

        # Verify DeepSeek V4.1 recipe compatibility for enabled/disabled/adaptive
        v41_enabled, v41_budget = resolve_thinking(openai_req.model_dump(), False)
        self.assertTrue(v41_enabled)
        self.assertEqual(v41_budget, 75)

        # Messages: system, user(mm), assistant(tool_calls), tool, user(text collapsed)
        self.assertEqual(len(openai_req.messages), 5)
        self.assertEqual(openai_req.messages[0].role, RoleEnum.system)
        self.assertEqual(
            openai_req.messages[0].content, "System part 1\n\nSystem part 2"
        )
        self.assertEqual(openai_req.messages[1].role, RoleEnum.user)
        self.assertIsInstance(openai_req.messages[1].content, list)
        self.assertEqual(
            openai_req.messages[1].content[1].image_url.url,
            "data:image/png;base64,QUJD",
        )
        self.assertEqual(
            openai_req.messages[1].content[2].image_url.url, "https://example.com/a.jpg"
        )
        self.assertEqual(openai_req.messages[2].role, RoleEnum.assistant)
        self.assertEqual(openai_req.messages[2].reasoning_content, "Checking tools")
        self.assertEqual(openai_req.messages[2].content, "Let me look that up.")
        self.assertEqual(openai_req.messages[2].tool_calls[0].id, "toolu_1")
        self.assertEqual(
            json.loads(openai_req.messages[2].tool_calls[0].function.arguments),
            {"city": "Hangzhou"},
        )
        self.assertEqual(openai_req.messages[3].role, RoleEnum.tool)
        self.assertEqual(openai_req.messages[3].tool_call_id, "toolu_1")
        self.assertEqual(openai_req.messages[3].content, '{"temp": 22}')
        self.assertEqual(openai_req.messages[4].role, RoleEnum.user)
        self.assertEqual(openai_req.messages[4].content, "Summarize now.")

        # Thinking disabled + tool_choice none omits tools so renderers never inject tools
        req_disabled = AnthropicMessagesRequest(
            model="m",
            max_tokens=32,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "only text"}]}
            ],
            tools=[{"name": "t", "input_schema": {"type": "object"}}],
            thinking={"type": "disabled"},
            tool_choice={"type": "none"},
            output_config={"format": {"type": "text"}},
        )
        oai_disabled = translate_anthropic_request(req_disabled)
        self.assertFalse(oai_disabled.enable_thinking)
        self.assertEqual(oai_disabled.thinking_budget, 0)
        self.assertEqual(oai_disabled.chat_template_kwargs, {"enable_thinking": False})
        self.assertEqual(
            oai_disabled.get_explicit_thinking_mode(), ThinkingMode.DISABLED
        )
        self.assertEqual(resolve_thinking(oai_disabled.model_dump(), True), (False, 75))
        self.assertEqual(oai_disabled.messages[0].content, "only text")
        self.assertEqual(oai_disabled.tool_choice, "none")
        self.assertIsNone(oai_disabled.tools)
        self.assertEqual(oai_disabled.response_format.type, "text")

        # Thinking adaptive: must NOT put "thinking_mode": "adaptive" into chat_template_kwargs
        req_adaptive = AnthropicMessagesRequest(
            model="m",
            max_tokens=128,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "budget_tokens": 32},
            tools=[{"name": "t", "input_schema": {"type": "object"}}],
            tool_choice={"type": "any", "disable_parallel_tool_use": False},
        )
        oai_adaptive = translate_anthropic_request(req_adaptive)
        self.assertIsNone(oai_adaptive.thinking)
        self.assertIsNone(oai_adaptive.enable_thinking)
        self.assertEqual(oai_adaptive.thinking_budget, 32)
        self.assertIsNone(oai_adaptive.chat_template_kwargs)
        self.assertEqual(
            oai_adaptive.extra_configs.thinking_mode, ThinkingMode.ADAPTIVE
        )
        self.assertEqual(
            oai_adaptive.get_explicit_thinking_mode(), ThinkingMode.ADAPTIVE
        )
        self.assertEqual(
            resolve_thinking(oai_adaptive.model_dump(), False), (False, 75)
        )
        self.assertEqual(oai_adaptive.tool_choice, "required")
        self.assertTrue(oai_adaptive.parallel_tool_calls)

    def test_translate_user_assistant_and_count_tokens_edge_cases(self):
        # Empty system collapses away; empty text after tool_result does not emit empty user turn;
        # is_error prefixes [Tool Error]; multimodal tool_result translates images;
        # assistant with only thinking/redacted_thinking has content="".
        count_req = AnthropicCountTokensRequest(
            model="m",
            system=[{"type": "text", "text": ""}],
            messages=[
                {
                    "role": "user",
                    "content": "go",
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "redacted_thinking", "data": "secret"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_err",
                            "is_error": True,
                            "content": "timeout",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_empty_err",
                            "is_error": True,
                            "content": None,
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_img_err",
                            "is_error": True,
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://example.com/err.png",
                                    },
                                }
                            ],
                        },
                        {"type": "text", "text": ""},
                    ],
                },
            ],
            tools=[{"name": "f", "input_schema": {"type": "object"}}],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        oai = translate_anthropic_request(count_req)
        self.assertIsNone(oai.max_tokens)
        self.assertFalse(oai.parallel_tool_calls)
        # No system, no trailing empty user after tool messages -> user, assistant, 3x tool
        self.assertEqual(
            [m.role for m in oai.messages],
            [
                RoleEnum.user,
                RoleEnum.assistant,
                RoleEnum.tool,
                RoleEnum.tool,
                RoleEnum.tool,
            ],
        )
        self.assertEqual(oai.messages[1].content, "")
        self.assertEqual(oai.messages[1].reasoning_content, "hmm")
        self.assertEqual(oai.messages[2].content, "[Tool Error]: timeout")
        self.assertEqual(oai.messages[3].content, "[Tool Error]")
        self.assertEqual(oai.messages[4].content[0].type, ContentPartTypeEnum.text)
        self.assertEqual(oai.messages[4].content[0].text, "[Tool Error]: ")
        self.assertEqual(oai.messages[4].content[1].type, ContentPartTypeEnum.image_url)

    def test_translate_openai_response_variants_and_stop_sequences(self):
        resp = ChatCompletionResponse(
            model="backend-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role=RoleEnum.assistant,
                        reasoning_content="step 1",
                        content="Done.",
                        tool_calls=[
                            ToolCall(
                                index=0,
                                id="call_abc",
                                type="function",
                                function=FunctionCall(
                                    name="calc", arguments='{"x": 1}'
                                ),
                            )
                        ],
                    ),
                    finish_reason=FinisheReason.tool_calls,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=100,
                completion_tokens=25,
                total_tokens=125,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=40),
            ),
        )
        msg = translate_openai_response(resp, model="claude-req")
        self.assertTrue(msg.id.startswith("msg_"))
        self.assertEqual(msg.model, "claude-req")
        self.assertEqual(msg.stop_reason, "tool_use")
        self.assertIsNone(msg.stop_sequence)
        # to_wire_dict must explicitly preserve "stop_sequence": None
        wire = msg.to_wire_dict()
        self.assertIn("stop_sequence", wire)
        self.assertIsNone(wire["stop_sequence"])

        self.assertEqual(msg.usage.input_tokens, 60)
        self.assertEqual(msg.usage.cache_read_input_tokens, 40)
        self.assertEqual(msg.usage.output_tokens, 25)
        self.assertEqual(
            [b.type for b in msg.content], ["thinking", "text", "tool_use"]
        )

        # Length takes precedence even if matched_stop_sequence is also populated
        resp_length = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role=RoleEnum.assistant, content="truncated"),
                    finish_reason=FinisheReason.length,
                    matched_stop_sequence="<STOP>",
                )
            ],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=10),
        )
        msg_length = translate_openai_response(
            resp_length, model="m", stop_sequences=["<STOP>"]
        )
        self.assertEqual(msg_length.stop_reason, "max_tokens")
        self.assertIsNone(msg_length.stop_sequence)

        # Choice matched_stop_sequence works WITHOUT DebugInfo
        resp_choice_stop = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role=RoleEnum.assistant, content="hello"),
                    finish_reason=FinisheReason.stop,
                    matched_stop_sequence="<STOP>",
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=2),
            debug_info=None,
        )
        msg_choice_stop = translate_openai_response(
            resp_choice_stop, model="m", stop_sequences=["<STOP>"]
        )
        self.assertEqual(msg_choice_stop.stop_reason, "stop_sequence")
        self.assertEqual(msg_choice_stop.stop_sequence, "<STOP>")

        # DebugInfo fallback still works when matched_stop_sequence is unset
        resp_debug_stop = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role=RoleEnum.assistant, content="hello"),
                    finish_reason=FinisheReason.stop,
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=2),
            debug_info=DebugInfo(raw_output=["hello<STOP>"]),
        )
        msg_debug_stop = translate_openai_response(
            resp_debug_stop, model="m", stop_sequences=["<STOP>"]
        )
        self.assertEqual(msg_debug_stop.stop_reason, "stop_sequence")
        self.assertEqual(msg_debug_stop.stop_sequence, "<STOP>")

    def test_v41_stream_parser_tracks_matched_stop_sequence(self):
        # DeepSeek V4.1 parser tracks matched user stop sequence
        parser = V41StreamParser(thinking=False, stop=("DONE",))
        deltas = parser.feed("AnswerDONEignored")
        self.assertEqual([d.text for d in deltas], ["Answer"])
        self.assertTrue(parser.stopped)
        self.assertEqual(parser.matched_stop, "DONE")

    def test_streaming_state_machine_and_tool_continuation_and_stop_sequences(self):
        async def chunk_gen():
            yield ChatCompletionStreamResponse(
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role=RoleEnum.assistant, content=""),
                    )
                ],
                usage=None,
            )
            # Single chunk with both reasoning_content and text content
            yield ChatCompletionStreamResponse(
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            reasoning_content="Let me think",
                            content="Calling tools:",
                        ),
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=30,
                    completion_tokens=8,
                    total_tokens=38,
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=10),
                ),
            )
            yield ChatCompletionStreamResponse(
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            tool_calls=[
                                ToolCall(
                                    index=0,
                                    id="toolu_a",
                                    type="function",
                                    function=FunctionCall(
                                        name="fn_a", arguments='{"city": '
                                    ),
                                )
                            ]
                        ),
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=30, completion_tokens=16, total_tokens=46
                ),
            )
            # Continuation with id=None AND repeated name="fn_a" must NOT split the block
            yield ChatCompletionStreamResponse(
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(
                            tool_calls=[
                                ToolCall(
                                    index=0,
                                    id=None,
                                    type="function",
                                    function=FunctionCall(
                                        name="fn_a", arguments='"NYC"}'
                                    ),
                                ),
                                ToolCall(
                                    index=1,
                                    id="toolu_b",
                                    type="function",
                                    function=FunctionCall(
                                        name="fn_b", arguments='{"n": 2}'
                                    ),
                                ),
                            ]
                        ),
                        finish_reason=FinisheReason.tool_calls,
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=30, completion_tokens=24, total_tokens=54
                ),
            )

        async def collect():
            translator = AnthropicStreamTranslator(model="claude-stream")
            return [ev async for ev in translator.stream(chunk_gen())]

        events = _parse_sse_events(asyncio.run(collect()))
        starts = [payload for name, payload in events if name == "content_block_start"]
        stops = [payload for name, payload in events if name == "content_block_stop"]
        self.assertEqual([s["index"] for s in starts], [0, 1, 2, 3])
        self.assertEqual([s["index"] for s in stops], [0, 1, 2, 3])
        self.assertEqual(
            [s["content_block"]["type"] for s in starts],
            ["thinking", "text", "tool_use", "tool_use"],
        )

        # Streaming stop_sequences without DebugInfo
        async def stop_gen():
            yield ChatCompletionStreamResponse(
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content="Hi"),
                        finish_reason=FinisheReason.stop,
                        matched_stop_sequence="STOP",
                    )
                ],
                usage=UsageInfo(prompt_tokens=4, completion_tokens=2),
            )

        async def collect_stop():
            translator = AnthropicStreamTranslator(model="m", stop_sequences=["STOP"])
            return [ev async for ev in translator.stream(stop_gen())]

        stop_events = _parse_sse_events(asyncio.run(collect_stop()))
        self.assertEqual(
            stop_events[-2][1]["delta"],
            {"stop_reason": "stop_sequence", "stop_sequence": "STOP"},
        )

    def test_streaming_empty_zero_chunk_and_length_and_cancel(self):
        async def zero_gen():
            if False:
                yield None  # pragma: no cover

        events = _parse_sse_events(
            asyncio.run(
                self._collect_stream(AnthropicStreamTranslator(model="m"), zero_gen())
            )
        )
        self.assertEqual(
            [name for name, _ in events],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )

        async def cancel_gen():
            yield ChatCompletionStreamResponse(
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0, delta=DeltaMessage(content="x")
                    )
                ],
                usage=UsageInfo(prompt_tokens=2, completion_tokens=1),
            )
            raise asyncio.CancelledError("cancelled")

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(
                self._collect_stream(AnthropicStreamTranslator(model="m"), cancel_gen())
            )

    @staticmethod
    async def _collect_stream(translator, gen):
        return [ev async for ev in translator.stream(gen)]

    def test_usage_and_tool_arg_helpers(self):
        none_usage = build_anthropic_usage(None, output_tokens_override=3)
        self.assertEqual(
            none_usage,
            AnthropicUsage(input_tokens=0, output_tokens=3, cache_read_input_tokens=0),
        )
        clamped = build_anthropic_usage(
            UsageInfo(
                prompt_tokens=5,
                completion_tokens=2,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=20),
            )
        )
        self.assertEqual(clamped.input_tokens, 0)
        self.assertEqual(clamped.cache_read_input_tokens, 20)

        self.assertEqual(_parse_tool_arguments(None), {})
        self.assertEqual(_parse_tool_arguments("  "), {})
        self.assertEqual(_parse_tool_arguments("{bad"), {})
        self.assertEqual(_parse_tool_arguments("[1, 2]"), {"value": [1, 2]})
        self.assertEqual(_parse_tool_arguments({"a": 1}), {"a": 1})

    def test_exception_mapping_sanitizes_internal_details(self):
        code, body = exception_to_anthropic_http(
            ConcurrencyException("secret internal limit detail")
        )
        self.assertEqual(code, 429)
        self.assertEqual(body["error"]["type"], "overloaded_error")
        self.assertNotIn("secret", body["error"]["message"])

        code, body = exception_to_anthropic_http(
            AnthropicTranslationError("bad translated field")
        )
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["message"], "bad translated field")

        code, body = exception_to_anthropic_http(
            FtRuntimeException(ExceptionType.INVALID_PARAMS, "bad param")
        )
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")

        code, body = exception_to_anthropic_http(
            FtRuntimeException(ExceptionType.UNSUPPORTED_OPERATION, "unsupported")
        )
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")

        code, body = exception_to_anthropic_http(
            FtRuntimeException(
                ExceptionType.RESOURCE_EXHAUSTED, "secret host 10.0.0.1 full"
            )
        )
        self.assertEqual(code, 529)
        self.assertEqual(body["error"]["type"], "overloaded_error")
        self.assertNotIn("10.0.0.1", body["error"]["message"])

        code, body = exception_to_anthropic_http(
            FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT, "secret backend 1.2.3.4 timed out"
            )
        )
        self.assertEqual(code, 504)
        self.assertEqual(body["error"]["type"], "api_error")
        self.assertNotIn("1.2.3.4", body["error"]["message"])

        code, body = exception_to_anthropic_http(
            FtRuntimeException(ExceptionType.CANCELLED_ERROR, "secret internal cancel")
        )
        self.assertEqual(code, 499)
        self.assertEqual(body["error"]["type"], "api_error")
        self.assertNotIn("secret", body["error"]["message"])

        # Internal ValueError / RuntimeError / ExecutionException must NEVER leak messages
        for exc in (
            ValueError("secret internal parse error /mnt/model"),
            RuntimeError("boom /internal/traceback"),
            FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION, "CUDA error at secret path"
            ),
        ):
            with self.subTest(exc=exc):
                code, body = exception_to_anthropic_http(exc)
                self.assertEqual(code, 500)
                self.assertEqual(body["error"]["type"], "api_error")
                self.assertEqual(body["error"]["message"], "Internal server error")

        # AnthropicMessage wire dict contract
        msg = AnthropicMessage(
            id="msg_1",
            model="m",
            content=[AnthropicOutputTextBlock(text="ok")],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=AnthropicUsage(input_tokens=1, output_tokens=1),
        )
        wire = msg.to_wire_dict()
        self.assertIsNone(wire["stop_sequence"])
        self.assertEqual(wire["stop_reason"], "end_turn")


if __name__ == "__main__":
    main()
