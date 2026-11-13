import asyncio
import json
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import (
    OutputDelta,
    RendererParams,
    ThinkStatus,
)
from rtp_llm.openai.renderers.minimax_m31_vl_renderer import (
    MiniMaxM31Renderer,
    MiniMaxM31VLRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.minimax_m31_detector import (
    NS_TOKEN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    MiniMaxM31Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import (
    MiniMaxM31ReasoningDetector,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, MMUrlType


class _Tokenizer:
    eos_token_id = 0
    special_tokens_map = {}
    additional_special_tokens = []
    chat_template = (
        "{% for message in messages %}{{ message.role }}:"
        "{% if message.content is string %}{{ message.content }}"
        "{% else %}{% for part in message.content %}"
        "{% if part.type == 'text' %}{{ part.text }}"
        "{% elif part.type == 'image' %}]<]image[>["
        "{% elif part.type == 'video' %}]<]video[>[{% endif %}"
        "{% endfor %}{% endif %};{% endfor %}"
    )

    def encode(self, text, **kwargs):
        return list(text.encode())

    def __len__(self):
        return 256

    def tokenize(self, text):
        return list(text)

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8")


def _renderer(renderer_type):
    tokenizer = _Tokenizer()
    params = RendererParams(
        model_type="minimax_m31_vl",
        max_seq_len=8192,
        eos_token_id=0,
        stop_word_ids_list=[],
    )
    return renderer_type(tokenizer, params, GenerateEnvConfig(), RenderConfig())


class MiniMaxM31VLRendererTest(unittest.TestCase):
    def test_raw_thinking_uses_model_defaults_before_renderer_creation(self):
        from rtp_llm.server.server_args import server_args

        for model_type in ("minimax_m31", "minimax_m31_vl"):
            with self.subTest(model_type=model_type), patch.dict(
                os.environ, {}, clear=True
            ):
                env = server_args.setup_args(
                    ["--model_type", model_type]
                ).generate_env_config
                generated = GenerateConfig(
                    thinking_mode=ThinkingMode.ENABLED, max_thinking_tokens=32
                )
                generated.add_thinking_params(_Tokenizer(), env)
                # Fixed ENABLED starts inside reasoning; main only requires
                # the end marker, unlike ADAPTIVE which detects a begin tag.
                self.assertEqual(generated.begin_think_token_ids, [])
                self.assertTrue(generated.in_think_mode)
                self.assertEqual(generated.end_think_token_ids, list(b"</mm:think>"))

    def test_empty_think_markers_do_not_loop(self):
        detector = MiniMaxM31ReasoningDetector()
        detector.think_start_token = ""
        detector.think_end_token = ""
        self.assertEqual(
            detector.parse_streaming_increment("answer").normal_text, "answer"
        )

    def test_renderer_does_not_change_shared_thinking_config(self):
        config = GenerateEnvConfig()
        tokenizer = _Tokenizer()
        params = RendererParams(
            model_type="minimax_m31",
            max_seq_len=8192,
            eos_token_id=0,
            stop_word_ids_list=[],
        )
        renderer = MiniMaxM31Renderer(tokenizer, params, config, RenderConfig())
        self.assertEqual(renderer.default_thinking_mode, ThinkingMode.ADAPTIVE)
        self.assertEqual(config.think_mode, "disabled")
        self.assertEqual(config.think_start_tag, "<think>\n")

    def test_runtime_escaped_tags_drive_adaptive_request_config(self):
        config = GenerateEnvConfig()
        config.think_start_tag = r"<think>\n"
        config.think_end_tag = r"</think>\n\n"
        renderer = MiniMaxM31Renderer(
            _Tokenizer(),
            RendererParams(
                model_type="minimax_m31",
                max_seq_len=8192,
                eos_token_id=0,
                stop_word_ids_list=[],
            ),
            config,
            RenderConfig(),
        )
        endpoint = object.__new__(OpenaiEndpoint)
        endpoint.tokenizer = renderer.tokenizer
        endpoint.chat_renderer = renderer
        endpoint.generate_env_config = config
        endpoint.stop_words_id_list = []
        endpoint.stop_words_str_list = []
        request = ChatCompletionRequest.model_validate(
            {"model": "minimax_m31", "messages": [{"role": "user", "content": "hello"}]}
        )
        generated = endpoint._extract_generation_config(request, renderer=renderer)
        self.assertEqual(renderer.think_start_tag, "<mm:think>")
        self.assertEqual(renderer.think_end_tag, "</mm:think>")
        self.assertEqual(generated.thinking_mode, ThinkingMode.ADAPTIVE)
        self.assertEqual(generated.begin_think_token_ids, list(b"<mm:think>"))
        self.assertEqual(generated.end_think_token_ids, list(b"</mm:think>"))
        self.assertEqual(config.think_start_tag, r"<think>\n")

    def test_explicit_deployment_thinking_config_is_preserved(self):
        config = GenerateEnvConfig()
        config.think_start_tag = "<custom_think>"
        config.think_end_tag = "</custom_think>"
        config.think_mode_explicit = True
        renderer = MiniMaxM31Renderer(
            _Tokenizer(),
            RendererParams(
                model_type="minimax_m31",
                max_seq_len=8192,
                eos_token_id=0,
                stop_word_ids_list=[],
            ),
            config,
            RenderConfig(),
        )
        self.assertEqual(renderer.default_thinking_mode, ThinkingMode.DISABLED)
        self.assertEqual(config.think_start_tag, "<custom_think>")
        request = ChatCompletionRequest.model_validate(
            {
                "model": "minimax_m31",
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"thinking_mode": "enabled"},
            }
        )
        parser = renderer._create_reasoning_parser(request)
        self.assertEqual(
            parser.parse_non_stream("<custom_think>reason</custom_think>answer"),
            ("reason", "answer"),
        )

    def test_adaptive_thinking_with_tools_separates_reasoning(self):
        renderer = _renderer(MiniMaxM31Renderer)
        request = ChatCompletionRequest.model_validate(
            {
                "model": "minimax_m31",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a value",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )

        async def parse_output():
            status = (await renderer._create_status_list(1, request))[0]
            text = "<mm:think>Reasoning.</mm:think>Answer"
            output = GenerateOutput(
                output_ids=torch.tensor([list(text.encode())], dtype=torch.int32),
                finished=True,
                aux_info=AuxInfo(output_len=len(text)),
            )
            result = await renderer._update_single_status(
                status, output, 256, renderer.extra_stop_words, [], False
            )
            return result, status

        result, status = asyncio.run(parse_output())
        self.assertEqual(result.output_str.reasoning_content, "Reasoning.")
        self.assertNotIn("Reasoning.", result.output_str.content or "")
        self.assertEqual(status.delta_output_string, "Answer")

    def test_streamed_thinking_end_tag_split_across_chunks(self):
        renderer = _renderer(MiniMaxM31Renderer)
        request = ChatCompletionRequest.model_validate(
            {
                "model": "minimax_m31",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a value",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
        parser = renderer._create_reasoning_parser(request)
        first = parser.parse_stream_chunk("<mm:think>abc</mm:")
        second = parser.parse_stream_chunk("think>Answer")
        self.assertEqual(first, ("abc", ""))
        self.assertEqual(second, ("", "Answer"))
        buffered = renderer._create_reasoning_parser(request)
        buffered.detector.stream_reasoning = False
        self.assertEqual(buffered.parse_stream_chunk("<mm:think>abc</mm:"), ("", ""))
        self.assertEqual(buffered.parse_stream_chunk("think>Answer"), ("abc", "Answer"))

    def test_main_adaptive_reasoning_parser(self):
        renderer = _renderer(MiniMaxM31Renderer)
        self.assertEqual(renderer.default_thinking_mode, ThinkingMode.ADAPTIVE)
        request = ChatCompletionRequest.model_validate(
            {"model": "minimax_m31", "messages": [{"role": "user", "content": "hello"}]}
        )
        self.assertEqual(renderer.resolve_thinking_mode(request), ThinkingMode.ADAPTIVE)
        disabled = request.model_copy(
            update={"chat_template_kwargs": {"thinking_mode": "disabled"}}
        )
        self.assertEqual(
            renderer.resolve_thinking_mode(disabled), ThinkingMode.DISABLED
        )
        budget_zero = request.model_copy(update={"thinking_budget": 0})
        self.assertEqual(
            renderer.resolve_thinking_mode(budget_zero), ThinkingMode.DISABLED
        )
        enabled = request.model_copy(
            update={"chat_template_kwargs": {"thinking_mode": "enabled"}}
        )
        self.assertEqual(renderer.resolve_thinking_mode(enabled), ThinkingMode.ENABLED)
        self.assertFalse(renderer.should_process_think(enabled))
        self.assertFalse(renderer.should_process_think(disabled))
        parser = renderer._create_reasoning_parser(enabled)
        self.assertEqual(
            parser.parse_non_stream("Reasoning.</mm:think>Answer"),
            ("Reasoning.", "Answer"),
        )
        status = ThinkStatus(thinking_mode=ThinkingMode.ADAPTIVE, decision_made=False)

        def packet(text):
            return OutputDelta(text, None, 1, 1, 0)

        first = renderer._split_reasoning_text_and_content(
            packet("<mm:think>Reasoning."), status
        )
        second = renderer._split_reasoning_text_and_content(
            packet("</mm:think>Answer"), status
        )
        self.assertEqual(first.reasoning_content, "Reasoning.")
        self.assertEqual(first.content, "")
        self.assertEqual(second.content, "Answer")

    def test_explicit_thinking_modes_set_generation_prefix(self):
        for renderer_type in (MiniMaxM31Renderer, MiniMaxM31VLRenderer):
            renderer = _renderer(renderer_type)
            base = {
                "model": "minimax_m31",
                "messages": [{"role": "user", "content": "hello"}],
            }
            adaptive = renderer.render_chat(ChatCompletionRequest.model_validate(base))
            self.assertEqual(adaptive.rendered_prompt, "user:hello;")
            for mode, suffix in (
                ("disabled", "</mm:think>"),
                ("enabled", "<mm:think>"),
            ):
                request = ChatCompletionRequest.model_validate(
                    {**base, "chat_template_kwargs": {"thinking_mode": mode}}
                )
                rendered = renderer.render_chat(request)
                self.assertEqual(
                    rendered.rendered_prompt, adaptive.rendered_prompt + suffix
                )
                self.assertEqual(
                    rendered.input_ids, list(rendered.rendered_prompt.encode())
                )

            without_generation_prompt = renderer.render_chat(
                ChatCompletionRequest.model_validate(
                    {
                        **base,
                        "chat_template_kwargs": {
                            "thinking_mode": "disabled",
                            "add_generation_prompt": False,
                        },
                    }
                )
            )
            self.assertEqual(
                without_generation_prompt.rendered_prompt, adaptive.rendered_prompt
            )
            budget_zero = renderer.render_chat(
                ChatCompletionRequest.model_validate({**base, "thinking_budget": 0})
            )
            self.assertEqual(
                budget_zero.rendered_prompt, adaptive.rendered_prompt + "</mm:think>"
            )

    def test_protocol_special_tokens_are_not_engine_stop_words(self):
        protocol = [
            NS_TOKEN,
            TOOL_CALL_BEGIN,
            TOOL_CALL_END,
            "<mm:think>",
            "</mm:think>",
        ]
        tokenizer = _Tokenizer()
        tokenizer.special_tokens_map = {
            "additional_special_tokens": protocol,
            "eos_token": "<eos>",
        }
        tokenizer.additional_special_tokens = protocol
        tokenizer.convert_tokens_to_ids = lambda word: tokenizer.encode(word)
        params = RendererParams(
            model_type="minimax_m31",
            max_seq_len=8192,
            eos_token_id=0,
            stop_word_ids_list=[],
        )
        renderer = MiniMaxM31Renderer(
            tokenizer, params, GenerateEnvConfig(), RenderConfig()
        )
        stops = renderer.get_all_extra_stop_word_ids_list()
        self.assertIn(tokenizer.encode("<eos>"), stops)
        for marker in protocol:
            self.assertNotIn(tokenizer.encode(marker), stops)

    def test_tool_named_array_elements_and_invoke_parameter(self):
        tool = Tool(
            type="function",
            function=Function(
                name="lookup",
                description="Lookup",
                parameters={
                    "type": "object",
                    "properties": {
                        "cities": {"type": "array", "items": {"type": "string"}},
                        "invoke": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            ),
        )
        xml = (
            f'{TOOL_CALL_BEGIN}{NS_TOKEN}<invoke name="lookup">'
            f"{NS_TOKEN}<cities>{NS_TOKEN}<city>Paris{NS_TOKEN}</city>"
            f"{NS_TOKEN}<city>London{NS_TOKEN}</city>{NS_TOKEN}</cities>"
            f"{NS_TOKEN}<invoke>literal{NS_TOKEN}</invoke>"
            f"{NS_TOKEN}<count>2{NS_TOKEN}</count>"
            f"{NS_TOKEN}</invoke>{TOOL_CALL_END}"
        )
        for streaming, extra_attributes in (
            (False, ""),
            (True, ""),
            (False, 'id="call0" '),
            (True, 'id="call0" '),
        ):
            text = xml.replace("<invoke name=", f"<invoke {extra_attributes}name=")
            detector = MiniMaxM31Detector()
            if streaming:
                calls = []
                for start in range(0, len(text), 3):
                    calls.extend(
                        detector.parse_streaming_increment(
                            text[start : start + 3], [tool]
                        ).calls
                    )
            else:
                calls = detector.detect_and_parse(text, [tool]).calls
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                json.loads(calls[0].parameters),
                {"cities": ["Paris", "London"], "invoke": "literal", "count": 2},
            )

    def test_tool_output_schema_and_streaming(self):
        tool = Tool(
            type="function",
            function=Function(
                name="lookup",
                description="Look up a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            ),
        )
        xml = (
            f"{NS_TOKEN}<tool_call>"
            f'{NS_TOKEN}<invoke name="lookup">'
            f"{NS_TOKEN}<city>Paris{NS_TOKEN}</city>"
            f"{NS_TOKEN}<count>2{NS_TOKEN}</count>"
            f"{NS_TOKEN}</invoke>{NS_TOKEN}</tool_call>"
        )
        parsed = MiniMaxM31Detector().detect_and_parse("intro" + xml, [tool])
        self.assertEqual(parsed.normal_text, "intro")
        self.assertEqual(parsed.calls[0].name, "lookup")
        self.assertEqual(
            json.loads(parsed.calls[0].parameters), {"city": "Paris", "count": 2}
        )
        detector = MiniMaxM31Detector()
        calls = []
        for start in range(0, len(xml), 7):
            calls.extend(
                detector.parse_streaming_increment(xml[start : start + 7], [tool]).calls
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris", "count": 2})

    def test_tool_parameter_starting_with_invoke_is_not_a_new_call(self):
        tool = Tool(
            type="function",
            function=Function(
                name="lookup",
                description="Look up a value",
                parameters={
                    "type": "object",
                    "properties": {"invoke_count": {"type": "integer"}},
                },
            ),
        )
        xml = (
            f"{NS_TOKEN}<tool_call>"
            f'{NS_TOKEN}<invoke name="lookup">'
            f"{NS_TOKEN}<invoke_count>3{NS_TOKEN}</invoke_count>"
            f"{NS_TOKEN}</invoke>{NS_TOKEN}</tool_call>"
        )
        parsed = MiniMaxM31Detector().detect_and_parse(xml, [tool])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(json.loads(parsed.calls[0].parameters), {"invoke_count": 3})

        detector = MiniMaxM31Detector()
        calls = []
        for start in range(0, len(xml), 7):
            calls.extend(
                detector.parse_streaming_increment(xml[start : start + 7], [tool]).calls
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"invoke_count": 3})

    def test_tool_output_resolves_local_schema_references(self):
        tool = Tool(
            type="function",
            function=Function(
                name="lookup",
                description="Look up a value",
                parameters={
                    "type": "object",
                    "$defs": {
                        "Count": {"type": "integer"},
                        "Flag": {"type": "boolean"},
                    },
                    "properties": {
                        "count": {"$ref": "#/$defs/Count"},
                        "flag": {"$ref": "#/$defs/Flag"},
                    },
                },
            ),
        )
        xml = (
            f"{NS_TOKEN}<tool_call>"
            f'{NS_TOKEN}<invoke name="lookup">'
            f"{NS_TOKEN}<count>2{NS_TOKEN}</count>"
            f"{NS_TOKEN}<flag>false{NS_TOKEN}</flag>"
            f"{NS_TOKEN}</invoke>{NS_TOKEN}</tool_call>"
        )
        parsed = MiniMaxM31Detector().detect_and_parse(xml, [tool])
        self.assertEqual(
            json.loads(parsed.calls[0].parameters), {"count": 2, "flag": False}
        )

    def test_unclosed_streamed_invoke_recovers_after_eot(self):
        detector = MiniMaxM31Detector()
        malformed = (
            TOOL_CALL_BEGIN
            + f'{NS_TOKEN}<invoke name="lookup">'
            + "unfinished"
            + TOOL_CALL_END
        )
        first = detector.parse_streaming_increment(malformed, [])
        self.assertEqual(first.calls, [])
        self.assertEqual(first.normal_text, "")
        after = detector.parse_streaming_increment("plain text", [])
        self.assertEqual(after.normal_text, "plain text")

    def test_unclosed_invoke_does_not_consume_following_call(self):
        tool = Tool(
            type="function",
            function=Function(
                name="lookup",
                description="Look up a city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
        )
        xml = (
            TOOL_CALL_BEGIN
            + f'{NS_TOKEN}<invoke name="lookup">broken'
            + f'{NS_TOKEN}<invoke name="lookup">'
            + f"{NS_TOKEN}<city>Paris{NS_TOKEN}</city>"
            + f"{NS_TOKEN}</invoke>"
            + TOOL_CALL_END
        )
        parsed = MiniMaxM31Detector().detect_and_parse(xml, [tool])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(json.loads(parsed.calls[0].parameters), {"city": "Paris"})
        streaming = MiniMaxM31Detector()
        calls = []
        for offset in range(0, len(xml), 5):
            calls.extend(
                streaming.parse_streaming_increment(
                    xml[offset : offset + 5], [tool]
                ).calls
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris"})

    def test_incomplete_tool_prefix_is_flushed_as_normal_text(self):
        detector = MiniMaxM31Detector()
        self.assertEqual(
            detector.parse_streaming_increment("answer]", []).normal_text, "answer"
        )
        self.assertEqual(detector.drain_pending_normal_text(), "]")
        self.assertEqual(detector.drain_pending_normal_text(), "")

        async def flush():
            renderer = _renderer(MiniMaxM31Renderer)
            request = ChatCompletionRequest.model_validate(
                {
                    "model": "minimax_m31",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "Look up a value",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                }
            )
            status = (await renderer._create_status_list(1, request))[0]
            status.detector.parse_streaming_increment("answer]", [])
            status.reasoning_parser.parse_stream_chunk("<")
            status.output = GenerateOutput(
                output_ids=torch.tensor([[93]], dtype=torch.int32),
                finished=True,
                aux_info=AuxInfo(output_len=1),
            )
            response = await renderer._flush_buffer(
                [status], [], True, [ThinkStatus(decision_made=True)]
            )
            return response.choices[0].delta.content

        self.assertEqual(asyncio.run(flush()), "]<")

    def test_truncated_thinking_end_tag_stays_in_reasoning(self):
        async def flush():
            renderer = _renderer(MiniMaxM31Renderer)
            request = ChatCompletionRequest.model_validate(
                {
                    "model": "minimax_m31",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            )
            # Adaptive mode without tools uses plain StreamStatus, so exercise
            # the enabled mode that allocates a reasoning parser.
            request = request.model_copy(
                update={"chat_template_kwargs": {"thinking_mode": "enabled"}}
            )
            status = (await renderer._create_status_list(1, request))[0]
            status.reasoning_parser.parse_stream_chunk("<mm:think>abc</mm:")
            status.output = GenerateOutput(
                output_ids=torch.tensor([[60]], dtype=torch.int32),
                finished=True,
                aux_info=AuxInfo(output_len=1),
            )
            response = await renderer._flush_buffer(
                [status], [], True, [ThinkStatus(decision_made=True)]
            )
            return response.choices[0].delta

        delta = asyncio.run(flush())
        self.assertEqual(delta.reasoning_content, "</mm:")
        self.assertEqual(delta.content, "")

    def test_main_output_pipeline_returns_tool_call(self):
        renderer = _renderer(MiniMaxM31Renderer)
        request = ChatCompletionRequest.model_validate(
            {
                "model": "minimax_m31",
                "messages": [{"role": "user", "content": "Look up Paris"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
            }
        )
        xml = (
            f"{NS_TOKEN}<tool_call>"
            f'{NS_TOKEN}<invoke name="lookup">'
            f"{NS_TOKEN}<city>Paris{NS_TOKEN}</city>"
            f"{NS_TOKEN}</invoke>{NS_TOKEN}</tool_call>"
        )

        async def parse_output():
            status = (await renderer._create_status_list(1, request))[0]
            output = GenerateOutput(
                output_ids=torch.tensor([list(xml.encode())], dtype=torch.int32),
                finished=True,
                aux_info=AuxInfo(output_len=len(xml)),
            )
            return await renderer._update_single_status(
                status, output, 256, renderer.extra_stop_words, [], False
            )

        result = asyncio.run(parse_output())
        call = result.output_str.tool_calls[0]
        self.assertEqual(call.function.name, "lookup")
        self.assertEqual(json.loads(call.function.arguments), {"city": "Paris"})

    def test_tool_schema_and_history_arguments_reach_template(self):
        renderer = _renderer(MiniMaxM31Renderer)
        renderer.compiled_template_map[renderer.default_template_key] = (
            renderer._compile_jinja_template(
                "{{ tools[0].function.name }}:"
                "{{ messages[0].tool_calls[0].function.arguments.city }}:"
                "{{ tools[0].function.parameters | tojson(ensure_ascii=False) }}"
            )
        )
        request = ChatCompletionRequest.model_validate(
            {
                "model": "minimax_m31",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
            }
        )
        self.assertIsInstance(renderer._create_detector(request), MiniMaxM31Detector)
        self.assertEqual(
            renderer.render_chat(request).rendered_prompt,
            'lookup:Paris:{"type": "object", "properties": {"city": {"type": "string"}}}',
        )

    def test_malformed_historical_tool_arguments_report_invalid_params(self):
        renderer = _renderer(MiniMaxM31Renderer)
        renderer.compiled_template_map[renderer.default_template_key] = (
            renderer._compile_jinja_template(
                "{{ messages[0].tool_calls[0].function.arguments | tojson }}"
            )
        )
        request = ChatCompletionRequest.model_validate(
            {
                "model": "minimax_m31",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "call1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{broken",
                                },
                            }
                        ],
                    }
                ],
            }
        )
        with self.assertRaises(FtRuntimeException) as caught:
            renderer.render_chat(request)
        self.assertEqual(caught.exception.exception_type, ExceptionType.INVALID_PARAMS)

    def test_text_matches_main_basic_renderer(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "minimax_m31_vl",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
        actual = _renderer(MiniMaxM31VLRenderer).render_chat(request)
        expected = _renderer(BasicRenderer).render_chat(request)
        self.assertEqual(actual.input_ids, expected.input_ids)
        self.assertEqual(actual.rendered_prompt, expected.rendered_prompt)
        self.assertEqual(actual.multimodal_inputs, [])

    def test_vl_rejects_unsupported_or_missing_media_as_invalid_params(self):
        renderer = _renderer(MiniMaxM31VLRenderer)
        for part in (
            {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}},
            {"type": "image_url"},
        ):
            request = ChatCompletionRequest.model_validate(
                {
                    "model": "minimax_m31_vl",
                    "messages": [{"role": "user", "content": [part]}],
                }
            )
            with self.assertRaises(FtRuntimeException) as caught:
                renderer.render_chat(request)
            self.assertEqual(
                caught.exception.exception_type, ExceptionType.INVALID_PARAMS
            )

    def test_image_video_order_and_preprocess(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "minimax_m31_vl",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"},
                                "preprocess_config": {"max_pixels": 4096},
                            },
                            {
                                "type": "video_url",
                                "video_url": {"url": "https://example.com/video.mp4"},
                                "preprocess_config": {"fps": 2},
                            },
                        ],
                    }
                ],
            }
        )
        result = _renderer(MiniMaxM31VLRenderer).render_chat(request)
        self.assertIn("look]<]image[>[]<]video[>[", result.rendered_prompt)
        self.assertEqual(
            [item.url for item in result.multimodal_inputs],
            ["https://example.com/image.png", "https://example.com/video.mp4"],
        )
        self.assertEqual(
            [item.mm_type for item in result.multimodal_inputs],
            [MMUrlType.IMAGE, MMUrlType.VIDEO],
        )
        self.assertEqual(
            result.multimodal_inputs[0].mm_preprocess_config.max_pixels, 4096
        )
        self.assertEqual(result.multimodal_inputs[1].mm_preprocess_config.fps, 2)


if __name__ == "__main__":
    unittest.main()
