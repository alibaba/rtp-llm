import copy
import importlib.util
import json
import os
from pathlib import Path
from unittest import TestCase, main, skipUnless

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    GPTFunctionDefinition,
    GPTToolDefinition,
)
from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv4_detector import (
    DeepSeekV4Detector,
)


DSV4_ENCODING_PATH = Path(
    os.environ.get(
        "DSV4_ENCODING_PATH",
        "/mnt/nas1/hf/DeepSeek-V4-Flash/encoding/encoding_dsv4.py",
    )
)
VLLM_DSV4_ENCODING_PATH = Path(
    os.environ.get(
        "VLLM_DSV4_ENCODING_PATH",
        "/data0/baowending.bwd/vllm/vllm/tokenizers/deepseek_v4_encoding.py",
    )
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    eos_token_id = 1

    def __init__(self):
        self.encode_calls = []

    def encode(self, prompt: str, **kwargs):
        self.encode_calls.append((prompt, kwargs))
        return [ord(ch) for ch in prompt]

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(token_id) for token_id in token_ids)


def _make_renderer(encoding_module):
    renderer = DeepseekV4Renderer.__new__(DeepseekV4Renderer)
    renderer.encoding_module = encoding_module
    renderer.tokenizer = FakeTokenizer()
    renderer.think_mode = False
    return renderer


def _tool_dicts():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def _rtp_tools():
    return [
        GPTToolDefinition(
            type="function",
            function=GPTFunctionDefinition(**tool["function"]),
        )
        for tool in _tool_dicts()
    ]


def _rtp_messages_with_tools(messages, tools):
    expected_messages = copy.deepcopy(messages)
    if not tools:
        return expected_messages

    has_system = False
    for message in expected_messages:
        if message["role"] == "system":
            message["tools"] = copy.deepcopy(tools)
            has_system = True
            break
    if not has_system:
        expected_messages.insert(
            0, {"role": "system", "content": "", "tools": copy.deepcopy(tools)}
        )
    return expected_messages


def _normalize_effort(effort):
    if not isinstance(effort, str) or effort == "none":
        return None
    if effort in ("max", "xhigh"):
        return "max"
    return "high"


def _rtp_expected_prompt(
    encoding_module,
    messages,
    tools=None,
    thinking_mode="chat",
    reasoning_effort=None,
    drop_thinking=True,
    add_default_bos_token=True,
):
    return encoding_module.encode_messages(
        _rtp_messages_with_tools(messages, tools),
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=_normalize_effort(reasoning_effort),
    )


@skipUnless(
    DSV4_ENCODING_PATH.exists(),
    "DeepSeek V4 encoding reference file is not available",
)
class DeepseekV4RendererTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding = _load_module("rtp_dsv4_encoding_for_test", DSV4_ENCODING_PATH)

    def setUp(self):
        self.renderer = _make_renderer(self.encoding)

    def test_default_renderer_behavior_is_unchanged(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(messages=messages)

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(self.encoding, messages)

        self.assertEqual(rendered.rendered_prompt, expected)
        self.assertEqual(self.renderer.tokenizer.encode_calls[-1], (expected, {}))
        self.assertEqual(self.renderer.tokenizer.decode(rendered.input_ids), expected)

    def test_existing_think_mode_env_path_is_unchanged(self):
        self.renderer.think_mode = True
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(messages=messages)

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            thinking_mode="thinking",
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_enable_thinking_behavior_is_unchanged(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            messages=messages,
            chat_template_kwargs={"enable_thinking": True},
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            thinking_mode="thinking",
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_thinking_mode_kwarg_still_overrides_encoding_config(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            messages=messages,
            chat_template_kwargs={"thinking_mode": "thinking"},
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            thinking_mode="thinking",
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_reasoning_effort_is_applied_without_changing_thinking_mode(self):
        messages = [{"role": "user", "content": "Hello"}]
        for effort, mapped in (
            ("max", "max"),
            ("xhigh", "max"),
            ("high", "high"),
            ("minimal", "high"),
            ("none", None),
            (None, None),
        ):
            with self.subTest(effort=effort):
                request = ChatCompletionRequest(
                    messages=messages,
                    reasoning_effort=effort,
                    chat_template_kwargs={"thinking_mode": "thinking"},
                )
                rendered = self.renderer.render_chat(request)
                expected = _rtp_expected_prompt(
                    self.encoding,
                    messages,
                    thinking_mode="thinking",
                    reasoning_effort=mapped,
                )
                self.assertEqual(rendered.rendered_prompt, expected)

    def test_reasoning_effort_can_come_from_chat_template_kwargs(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            messages=messages,
            chat_template_kwargs={
                "thinking_mode": "thinking",
                "reasoning_effort": "xhigh",
            },
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            thinking_mode="thinking",
            reasoning_effort="max",
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_extra_config_still_overrides_request_chat_template_kwargs(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            messages=messages,
            reasoning_effort="high",
            chat_template_kwargs={"thinking_mode": "thinking"},
            extra_configs=GenerateConfig(
                chat_template_kwargs={
                    "thinking_mode": "chat",
                    "reasoning_effort": "max",
                }
            ),
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            thinking_mode="chat",
            reasoning_effort="max",
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_request_tools_keep_original_rtp_system_message_placement(self):
        messages = [{"role": "user", "content": "Weather?"}]
        request = ChatCompletionRequest(messages=messages, tools=_rtp_tools())

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            tools=_tool_dicts(),
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_existing_system_message_receives_tools(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Weather?"},
        ]
        request = ChatCompletionRequest(messages=messages, tools=_rtp_tools())

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            tools=_tool_dicts(),
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_history_tool_call_arguments_dict_is_encoded_as_json(self):
        request_messages = [
            {"role": "user", "content": "Call the tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "杭州"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        expected_messages = copy.deepcopy(request_messages)
        expected_messages[1]["tool_calls"][0]["function"]["arguments"] = json.dumps(
            {"city": "杭州"}, ensure_ascii=False
        )
        request = ChatCompletionRequest(
            messages=request_messages,
            tools=_rtp_tools(),
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            expected_messages,
            tools=_tool_dicts(),
        )

        self.assertEqual(rendered.rendered_prompt, expected)


class DeepseekV4DetectorTest(TestCase):
    def _tools(self):
        return [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "count": {"type": "integer"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="calculate_area",
                    parameters={
                        "type": "object",
                        "properties": {
                            "dimensions": {"type": "object"},
                            "precision": {"type": "integer"},
                        },
                    },
                ),
            ),
        ]

    def test_v4_tool_calls_block_is_parsed(self):
        detector = DeepSeekV4Detector()
        text = (
            "Let me check. "
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        result = detector.detect_and_parse(text, self._tools())

        self.assertEqual(result.normal_text, "Let me check. ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "杭州"})

    def test_v4_tool_parser_matches_vllm_schema_conversion_and_wrapper_repair(self):
        detector = DeepSeekV4Detector()
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="enabled" string="false">true</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="count" string="false">42</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="calculate_area">\n'
            '<｜DSML｜parameter name="arguments" string="false">'
            '{"dimensions":{"w":3,"h":4},"precision":2}'
            "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        result = detector.detect_and_parse(text, self._tools())

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"enabled": True, "count": 42}
        )
        self.assertEqual(
            json.loads(result.calls[1].parameters),
            {"dimensions": {"w": 3, "h": 4}, "precision": 2},
        )

    @skipUnless(
        VLLM_DSV4_ENCODING_PATH.exists(),
        "vLLM DeepSeek V4 encoding reference file is not available",
    )
    def test_official_parser_matches_vllm_reasoning_and_multiple_tool_calls(self):
        vllm_encoding = _load_module(
            "vllm_dsv4_encoding_for_detector_test", VLLM_DSV4_ENCODING_PATH
        )
        detector = DeepSeekV4Detector(
            encoding_module=vllm_encoding, thinking_mode="thinking"
        )
        text = (
            "think step</think>"
            "Final answer"
            "\n\n<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="count" string="false">2</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="calculate_area">\n'
            '<｜DSML｜parameter name="dimensions" string="false">{"w":3,"h":4}</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
            "<｜end▁of▁sentence｜>"
        )

        expected = vllm_encoding.parse_message_from_completion_text(text, "thinking")
        result = detector.detect_and_parse(text, self._tools())

        self.assertEqual(
            result.normal_text,
            "\n\n".join([expected["reasoning"], expected["content"]]),
        )
        self.assertEqual(len(result.calls), len(expected["tool_calls"]))
        self.assertEqual(
            [json.loads(call.parameters) for call in result.calls],
            [
                json.loads(tool_call["function"]["arguments"])
                for tool_call in expected["tool_calls"]
            ],
        )

    def test_v32_function_calls_block_is_not_accepted(self):
        detector = DeepSeekV4Detector()
        text = (
            "<｜DSML｜function_calls>"
            '<｜DSML｜invoke name="search">'
            '<｜DSML｜parameter name="query" string="true">vllm</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
            "</｜DSML｜function_calls>"
        )

        result = detector.detect_and_parse(text, [])

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])


if __name__ == "__main__":
    main()
