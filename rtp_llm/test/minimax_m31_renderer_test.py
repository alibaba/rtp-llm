import asyncio
import json
import unittest

import torch

from rtp_llm.config.generate_config import ThinkingMode
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest
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

    def encode(self, text):
        return list(text.encode())

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
        self.assertTrue(renderer.should_process_think(enabled))
        self.assertFalse(renderer.should_process_think(disabled))
        enabled_status = ThinkStatus(
            enable_think_mode=True,
            in_think_mode=renderer.should_process_think(enabled),
            thinking_mode=ThinkingMode.ENABLED,
        )
        enabled_delta = renderer._split_reasoning_text_and_content(
            OutputDelta("Reasoning.</mm:think>Answer", None, 1, 2, 0), enabled_status
        )
        self.assertEqual(enabled_delta.reasoning_content, "Reasoning.")
        self.assertEqual(enabled_delta.content, "Answer")
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
