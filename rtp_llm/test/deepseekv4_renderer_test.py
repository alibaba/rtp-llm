import copy
import importlib.util
import json
import os
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main, skipUnless
from unittest.mock import AsyncMock, Mock

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import (
    ContentPart,
    ContentPartTypeEnum,
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    GPTFunctionDefinition,
    GPTToolDefinition,
    ImageURL,
    RoleEnum,
    ToolCall,
)
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RendererParams,
    StreamResponseObject,
)
from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolStreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv4_detector import (
    DeepSeekV4Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs

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
    renderer.default_thinking_mode = ThinkingMode.DISABLED
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


def _rtp_two_tools():
    tools = _tool_dicts() + [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search documents",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    return [
        GPTToolDefinition(
            type="function",
            function=GPTFunctionDefinition(**tool["function"]),
        )
        for tool in tools
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
    if effort in ("low", "medium"):
        return None
    return "max" if effort == "xhigh" else effort


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


class DeepseekV4ToolChoiceConstraintTest(TestCase):
    def setUp(self):
        self.renderer = DeepseekV4Renderer.__new__(DeepseekV4Renderer)

    def test_named_tool_choice_filters_structural_tag_tools(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Search docs"}],
            tools=_rtp_two_tools(),
            tool_choice={"type": "function", "function": {"name": "search"}},
        )
        config = GenerateConfig()

        self.renderer.apply_chat_completion_constraints(request, config)

        tag = json.loads(config.structural_tag)
        invoke_info = DeepSeekV4Detector().structure_info()("search")
        self.assertEqual(tag["type"], "structural_tag")
        self.assertEqual(
            [item["begin"] for item in tag["format"]["content"]["tags"]],
            [invoke_info.begin],
        )
        self.assertIs(tag["format"]["content"]["stop_after_first"], True)

    def test_required_tool_choice_keeps_all_tools_and_allows_multiple_calls(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Search weather"}],
            tools=_rtp_two_tools(),
            tool_choice="required",
        )
        config = GenerateConfig()

        self.renderer.apply_chat_completion_constraints(request, config)

        tag = json.loads(config.structural_tag)
        get_info = DeepSeekV4Detector().structure_info()
        self.assertEqual(tag["type"], "structural_tag")
        self.assertEqual(
            [item["begin"] for item in tag["format"]["content"]["tags"]],
            [get_info("get_weather").begin, get_info("search").begin],
        )
        self.assertIs(tag["format"]["content"]["stop_after_first"], False)

    def test_forced_tool_choice_rejects_existing_grammar_constraints(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=_rtp_tools(),
            tool_choice="required",
        )
        cases = [
            GenerateConfig(json_format=True),
            GenerateConfig(json_schema={"type": "object"}),
            GenerateConfig(regex=".*"),
            GenerateConfig(ebnf='root ::= "x"'),
            GenerateConfig(structural_tag={"format": {"type": "tag"}}),
            GenerateConfig(response_format={"type": "json_object"}),
        ]

        for config in cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(
                    FtRuntimeException,
                    "tool_choice forced tool-call decoding conflicts",
                ):
                    self.renderer.apply_chat_completion_constraints(request, config)


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
        # The real renderer constructor derives both fields from THINK_MODE.
        self.renderer.think_mode = True
        self.renderer.default_thinking_mode = ThinkingMode.ENABLED
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

    def test_zero_max_thinking_tokens_disables_thinking_render(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            messages=messages,
            extra_configs=GenerateConfig(max_thinking_tokens=0),
            enable_thinking=True,
            chat_template_kwargs={"thinking_mode": "enabled"},
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(self.encoding, messages, thinking_mode="chat")

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_enabled_thinking_mode_maps_to_thinking_encoding_prompt(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            messages=messages,
            chat_template_kwargs={"thinking_mode": "enabled"},
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            thinking_mode="thinking",
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_adaptive_thinking_uses_chat_encoding_prompt(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            messages=messages,
            chat_template_kwargs={"thinking_mode": "adaptive"},
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            messages,
            thinking_mode="chat",
        )

        self.assertEqual(rendered.rendered_prompt, expected)

    def test_reasoning_effort_is_applied_without_changing_thinking_mode(self):
        messages = [{"role": "user", "content": "Hello"}]
        for effort, mapped in (
            ("max", "max"),
            ("xhigh", "max"),
            ("high", "high"),
            ("medium", None),
            ("low", None),
            ("none", None),
            (None, None),
        ):
            with self.subTest(effort=effort):
                request = ChatCompletionRequest(
                    messages=messages,
                    reasoning_effort=effort,
                    chat_template_kwargs={"thinking_mode": "enabled"},
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
                "thinking_mode": "enabled",
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

    def test_invalid_reasoning_effort_uses_dashscope_enum_message(self):
        for effort in ("minimum", "turbo"):
            with self.subTest(effort=effort):
                request = ChatCompletionRequest(
                    messages=[{"role": "user", "content": "Hello"}],
                    reasoning_effort=effort,
                    chat_template_kwargs={"thinking_mode": "enabled"},
                )

                with self.assertRaises(FtRuntimeException) as ctx:
                    self.renderer.render_chat(request)
                self.assertEqual(
                    ctx.exception.exception_type, ExceptionType.INVALID_PARAMS
                )
                self.assertIn(
                    "'reasoning_effort' must be one of: "
                    "'low', 'medium', 'high', 'xhigh', 'max'",
                    str(ctx.exception),
                )

    def test_clear_thinking_controls_drop_thinking_in_real_encoding(self):
        messages = [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": "First answer",
                "reasoning_content": "HISTORICAL_REASONING",
            },
            {"role": "user", "content": "Second question"},
        ]

        def _make_req(clear_thinking):
            if "clear_thinking" in ChatCompletionRequest.model_fields:
                return ChatCompletionRequest(
                    messages=messages,
                    enable_thinking=True,
                    clear_thinking=clear_thinking,
                )
            req = ChatCompletionRequest(messages=messages, enable_thinking=True)
            object.__setattr__(req, "clear_thinking", clear_thinking)
            return req

        rendered_default = self.renderer.render_chat(_make_req(None))
        rendered_true = self.renderer.render_chat(_make_req(True))
        rendered_false = self.renderer.render_chat(_make_req(False))

        self.assertNotIn("HISTORICAL_REASONING", rendered_default.rendered_prompt)
        self.assertNotIn("HISTORICAL_REASONING", rendered_true.rendered_prompt)
        self.assertIn("HISTORICAL_REASONING", rendered_false.rendered_prompt)
        self.assertEqual(
            rendered_false.rendered_prompt,
            _rtp_expected_prompt(
                self.encoding,
                messages,
                thinking_mode="thinking",
                drop_thinking=False,
            ),
        )

    def test_extra_config_still_overrides_request_chat_template_kwargs(self):
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            messages=messages,
            reasoning_effort="high",
            chat_template_kwargs={"thinking_mode": "enabled"},
            extra_configs=GenerateConfig(
                chat_template_kwargs={
                    "thinking_mode": "disabled",
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

    def test_tool_choice_none_does_not_expose_tools_to_dsv4_prompt(self):
        messages = [{"role": "user", "content": "Weather?"}]
        request = ChatCompletionRequest(
            messages=messages,
            tools=_rtp_tools(),
            tool_choice="none",
        )

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(self.encoding, messages)

        self.assertEqual(rendered.rendered_prompt, expected)
        self.assertIsNone(self.renderer._create_detector(request))

    def test_named_tool_choice_filters_prompt_and_structural_tag(self):
        messages = [{"role": "user", "content": "Search docs"}]
        request = ChatCompletionRequest(
            messages=messages,
            tools=_rtp_two_tools(),
            tool_choice={"type": "function", "function": {"name": "search"}},
        )
        config = GenerateConfig()

        rendered = self.renderer.render_chat(request)
        self.renderer.apply_chat_completion_constraints(request, config)
        tag = json.loads(config.structural_tag)

        self.assertIn("search", rendered.rendered_prompt)
        self.assertNotIn("get_weather", rendered.rendered_prompt)
        invoke_info = DeepSeekV4Detector().structure_info()("search")
        self.assertEqual(
            [item["begin"] for item in tag["format"]["content"]["tags"]],
            [invoke_info.begin],
        )

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

    def test_openai_tool_call_history_messages_are_rendered(self):
        request_messages = [
            {
                "role": "system",
                "content": "你是一个严谨的助手。在回答用户之前，你可以调用工具获取外部数据。",
            },
            {
                "role": "user",
                "content": "帮我查一下上海和北京的天气。",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_sh_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "上海"}',
                        },
                    },
                    {
                        "id": "call_bj_002",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "北京"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_sh_001",
                "name": "get_weather",
                "content": '{"temperature": "26C", "status": "晴"}',
            },
            {
                "role": "tool",
                "tool_call_id": "call_bj_002",
                "name": "get_weather",
                "content": '{"temperature": "18C", "status": "多云"}',
            },
        ]
        request = ChatCompletionRequest(
            messages=request_messages,
            tools=_rtp_tools(),
        )

        self.assertEqual(request.messages[3].name, "get_weather")
        self.assertEqual(request.messages[3].tool_call_id, "call_sh_001")
        self.assertEqual(request.messages[4].name, "get_weather")
        self.assertEqual(request.messages[4].tool_call_id, "call_bj_002")

        rendered = self.renderer.render_chat(request)
        expected = _rtp_expected_prompt(
            self.encoding,
            request_messages,
            tools=_tool_dicts(),
        )

        self.assertEqual(rendered.rendered_prompt, expected)
        self.assertEqual(
            rendered.rendered_prompt.count('<｜DSML｜invoke name="get_weather">'),
            2,
        )
        self.assertIn(
            '<tool_result>{"temperature": "26C", "status": "晴"}</tool_result>',
            rendered.rendered_prompt,
        )
        self.assertIn(
            '<tool_result>{"temperature": "18C", "status": "多云"}</tool_result>',
            rendered.rendered_prompt,
        )


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

    def test_detector_does_not_use_full_completion_parser_for_dsml_block(self):
        class FailingEncoding:
            def parse_message_from_completion_text(self, text, thinking_mode):
                raise AssertionError("full parser should not be used for DSML blocks")

        detector = DeepSeekV4Detector(
            encoding_module=FailingEncoding(), thinking_mode="thinking"
        )
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="count" string="false">2</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        result = detector.detect_and_parse(text, self._tools())

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "杭州", "count": 2}
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


class DeepseekV4ReasoningToolPipelineTest(IsolatedAsyncioTestCase):
    def _output(self):
        aux_info = AuxInfo()
        aux_info.input_len = 10
        aux_info.output_len = 20
        aux_info.reuse_len = 0
        output = Mock(spec=GenerateOutput)
        output.aux_info = aux_info
        return output

    async def test_tool_calls_after_unclosed_thinking_are_not_swallowed(self):
        renderer = _make_renderer(None)
        renderer._generate_log_probs = AsyncMock(return_value=None)

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=_rtp_tools(),
            chat_template_kwargs={"enable_thinking": True},
        )
        status = ReasoningToolStreamStatus(
            request,
            DeepSeekV4Detector(),
            ReasoningParser(model_type="deepseek-v3", force_reasoning=True),
        )
        status.delta_output_string = (
            "Let me inspect the request.\n\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), is_streaming=False
        )

        self.assertIsNotNone(delta)
        self.assertEqual(
            delta.output_str.reasoning_content, "Let me inspect the request."
        )
        self.assertIsNone(delta.output_str.content)
        self.assertEqual(len(delta.output_str.tool_calls), 1)
        self.assertEqual(delta.output_str.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(delta.output_str.tool_calls[0].function.arguments),
            {"city": "杭州"},
        )
        self.assertEqual(status.delta_output_string, "")

    async def test_non_streaming_full_completion_parser_handles_reasoning_and_tools(
        self,
    ):
        class FullCompletionEncoding:
            def __init__(self):
                self.calls = []

            def parse_message_from_completion_text(self, text, thinking_mode):
                self.calls.append((text, thinking_mode))
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Let me inspect the request.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "杭州"}',
                            },
                        }
                    ],
                }

        encoding = FullCompletionEncoding()
        renderer = _make_renderer(encoding)
        renderer._generate_log_probs = AsyncMock(return_value=None)

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=_rtp_tools(),
            chat_template_kwargs={"enable_thinking": True},
        )
        status = ReasoningToolStreamStatus(
            request,
            DeepSeekV4Detector(),
            ReasoningParser(model_type="deepseek-v3", force_reasoning=True),
        )
        raw_text = (
            "Let me inspect the request.</think>\n\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
        )
        status.delta_output_string = raw_text

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), is_streaming=False
        )

        self.assertEqual(encoding.calls, [(raw_text, "thinking")])
        self.assertIsNotNone(delta)
        self.assertEqual(
            delta.output_str.reasoning_content, "Let me inspect the request."
        )
        self.assertEqual(len(delta.output_str.tool_calls), 1)
        self.assertEqual(delta.output_str.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(delta.output_str.tool_calls[0].function.arguments),
            {"city": "杭州"},
        )
        self.assertEqual(status.delta_output_string, "")

    async def test_full_completion_parser_splits_content_left_in_reasoning(
        self,
    ):
        class FullCompletionEncoding:
            def parse_message_from_completion_text(self, text, thinking_mode):
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Let me compare.</think>9.9更大。",
                }

        renderer = _make_renderer(FullCompletionEncoding())
        renderer._generate_log_probs = AsyncMock(return_value=None)
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "9.9和9.11哪个大"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        status = ReasoningToolStreamStatus(
            request,
            None,
            ReasoningParser(model_type="deepseek-v3", force_reasoning=True),
        )
        status.delta_output_string = "Let me compare.</think>9.9更大。"

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), is_streaming=False
        )

        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "Let me compare.")
        self.assertEqual(delta.output_str.content, "9.9更大。")

    async def test_full_completion_parser_drops_repeated_leading_think_end(
        self,
    ):
        class FullCompletionEncoding:
            def parse_message_from_completion_text(self, text, thinking_mode):
                return {
                    "role": "assistant",
                    "content": "</think></think>春天来了。",
                    "reasoning_content": "短思考",
                }

        renderer = _make_renderer(FullCompletionEncoding())
        renderer._generate_log_probs = AsyncMock(return_value=None)
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "写春天"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        status = ReasoningToolStreamStatus(
            request,
            None,
            ReasoningParser(model_type="deepseek-v3", force_reasoning=True),
        )
        status.delta_output_string = "短思考</think></think>春天来了。"

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), is_streaming=False
        )

        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "短思考")
        self.assertEqual(delta.output_str.content, "春天来了。")

    async def test_full_completion_parser_parses_dsml_left_in_reasoning(
        self,
    ):
        class FullCompletionEncoding:
            def parse_message_from_completion_text(self, text, thinking_mode):
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": text,
                }

        renderer = _make_renderer(FullCompletionEncoding())
        renderer._generate_log_probs = AsyncMock(return_value=None)
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=_rtp_tools(),
            chat_template_kwargs={"enable_thinking": True},
        )
        status = ReasoningToolStreamStatus(
            request,
            DeepSeekV4Detector(),
            ReasoningParser(model_type="deepseek-v3", force_reasoning=True),
        )
        status.delta_output_string = (
            "Need weather.\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), is_streaming=False
        )

        self.assertIsNotNone(delta)
        self.assertEqual(delta.output_str.reasoning_content, "Need weather.")
        self.assertIsNone(delta.output_str.content)
        self.assertEqual(len(delta.output_str.tool_calls), 1)
        self.assertEqual(delta.output_str.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(delta.output_str.tool_calls[0].function.arguments),
            {"city": "杭州"},
        )
        self.assertEqual(status.delta_output_string, "")

    async def test_streaming_split_think_and_dsml_tags_do_not_leak(self):
        renderer = _make_renderer(None)
        renderer._generate_log_probs = AsyncMock(return_value=None)

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=_rtp_tools(),
            chat_template_kwargs={"enable_thinking": True},
        )
        status = ReasoningToolStreamStatus(
            request,
            DeepSeekV4Detector(),
            ReasoningParser(model_type="deepseek-v3", force_reasoning=True),
        )
        output = self._output()

        chunks = [
            "Let me inspect</thi",
            "nk>",
            "\n\n",
            "<｜DS",
            "ML｜tool_calls>\n",
            '<｜DSML｜invoke name="get_weather">\n',
            '<｜DSML｜parameter name="city" string="true">杭',
            "州</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ]

        reasoning_parts = []
        content_parts = []
        tool_names = {}
        tool_args = {}
        for chunk in chunks:
            delta = await renderer._process_single_token_delta(
                status,
                chunk,
                output,
                stop_words_str=[],
                stop_word_slice_list=[],
                is_streaming=True,
            )
            if delta is None:
                continue
            message = delta.output_str
            if message.reasoning_content:
                reasoning_parts.append(message.reasoning_content)
            if message.content:
                content_parts.append(message.content)
            for tool_call in message.tool_calls or []:
                tool_names.setdefault(tool_call.index, tool_call.function.name)
                tool_args[tool_call.index] = tool_args.get(tool_call.index, "") + (
                    tool_call.function.arguments or ""
                )

        self.assertEqual("".join(reasoning_parts), "Let me inspect")
        self.assertEqual(content_parts, [])
        self.assertEqual(tool_names, {0: "get_weather"})
        self.assertEqual(json.loads(tool_args[0]), {"city": "杭州"})
        self.assertNotIn("</think>", "".join(reasoning_parts + content_parts))
        self.assertNotIn("<｜DSML｜", "".join(reasoning_parts + content_parts))


class DeepseekV4StreamResponseTest(TestCase):
    def _response(self, delta: DeltaMessage, finish_reason=None):
        return StreamResponseObject(
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0, delta=delta, finish_reason=finish_reason
                )
            ]
        )

    def test_empty_intermediate_stream_chunks_are_suppressed(self):
        renderer = _make_renderer(None)

        self.assertFalse(
            renderer._should_yield_stream_response(
                self._response(DeltaMessage(content=""))
            )
        )
        self.assertTrue(
            renderer._should_yield_stream_response(
                self._response(DeltaMessage(role=RoleEnum.assistant, content=""))
            )
        )
        self.assertTrue(
            renderer._should_yield_stream_response(
                self._response(
                    DeltaMessage(
                        tool_calls=[
                            ToolCall(
                                index=0,
                                type="function",
                                function=FunctionCall(name="get_weather", arguments=""),
                            )
                        ]
                    )
                )
            )
        )
        self.assertTrue(
            renderer._should_yield_stream_response(
                self._response(
                    DeltaMessage(content=""), finish_reason=FinisheReason.tool_calls
                )
            )
        )


class DeepSeekV4DetectorStreamingTest(TestCase):
    def _make_tools(self):
        from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
            Function,
            Tool,
        )

        return [
            Tool(
                type="function",
                function=Function(
                    name="Write",
                    parameters={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                            "overwrite": {"type": "boolean"},
                            "line": {"type": "integer"},
                            "ratio": {"type": "number"},
                            "items": {"type": "array"},
                            "obj": {"type": "object"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "extra": {"type": "string"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

    def _stream_collect(self, text: str, chunk_size: int, tools=None):
        detector = DeepSeekV4Detector()
        tools = self._make_tools() if tools is None else tools
        normal_parts = []
        names = {}
        arg_parts = {}
        for i in range(0, len(text), chunk_size):
            res = detector.parse_streaming_increment(text[i : i + chunk_size], tools)
            if res.normal_text:
                normal_parts.append(res.normal_text)
            for call in res.calls:
                idx = call.tool_index
                if call.name:
                    names[idx] = call.name
                if call.parameters:
                    arg_parts.setdefault(idx, []).append(call.parameters)
        parsed_args = {
            idx: json.loads("".join(parts)) for idx, parts in arg_parts.items()
        }
        return "".join(normal_parts), names, parsed_args

    def test_streaming_typed_parameters_never_corrupt_json(self):
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="Write">\n'
            '<｜DSML｜parameter name="file_path" string="true">/tmp/test.py</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="overwrite" string="false">false</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="line" string="false">42</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="ratio" string="false">3.5</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="items" string="false">["alpha", "beta", "a"]</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="obj" string="false">{"a": 1, "b": "hello"}</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        for chunk_size in (1, 2, 4):
            normal, names, args = self._stream_collect(text, chunk_size=chunk_size)
            self.assertEqual(normal, "")
            self.assertEqual(names, {0: "Write"})
            self.assertEqual(
                args[0],
                {
                    "file_path": "/tmp/test.py",
                    "overwrite": False,
                    "line": 42,
                    "ratio": 3.5,
                    "items": ["alpha", "beta", "a"],
                    "obj": {"a": 1, "b": "hello"},
                },
            )

    def test_streaming_strings_preserve_tag_like_suffixes_and_escapes(self):
        content_val = (
            'Line1 "quoted" and \\backslash\\\n'
            "<div>html</div> & 杭州\t"
            "ends_with_parameter\n"
            "ends_with_DSML\n"
            "ends_with_</\n"
            "ends_with_</｜DSML｜"
        )
        text = (
            "Leading text\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="Write">\n'
            f'<｜DSML｜parameter name="content" string="true">{content_val}</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="file_path" string="true">parameter</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>\n"
            "Trailing text"
        )
        for chunk_size in (1, 2, 3, len(text)):
            normal, names, args = self._stream_collect(text, chunk_size=chunk_size)
            self.assertEqual(normal, "Leading text\n\nTrailing text")
            self.assertEqual(names, {0: "Write"})
            self.assertEqual(
                args[0], {"content": content_val, "file_path": "parameter"}
            )

    def test_streaming_bare_invoke_split_across_chunks_without_bot_token(self):
        text = (
            "Intro\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Tokyo</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>"
        )
        for chunk_size in (2, 5):
            normal, names, args = self._stream_collect(text, chunk_size=chunk_size)
            self.assertEqual(normal, "Intro\n")
            self.assertEqual(names, {0: "get_weather"})
            self.assertEqual(args, {0: {"city": "Tokyo"}})

    def test_streaming_direct_json_zero_param_and_multi_invoke(self):
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_time">\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '{"city": "San Francisco"}\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="Write">\n'
            '<｜DSML｜parameter name="overwrite" string="false">true</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        for chunk_size in (1, 3):
            normal, names, args = self._stream_collect(text, chunk_size=chunk_size)
            self.assertEqual(normal, "")
            self.assertEqual(names, {0: "get_time", 1: "get_weather", 2: "Write"})
            self.assertEqual(args[0], {})
            self.assertEqual(args[1], {"city": "San Francisco"})
            self.assertEqual(args[2], {"overwrite": True})

    def test_streaming_wrapper_repair_and_malformed_tag_recovery(self):
        # Single wrapper parameter ("arguments" / "input") unwraps cleanly on invoke close.
        wrapped = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="arguments" string="true">{"city": "NYC"}</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        _, names, args = self._stream_collect(wrapped, chunk_size=2)
        self.assertEqual(names, {0: "get_weather"})
        self.assertEqual(args[0], {"city": "NYC"})

        # Wrapper candidate followed by a second parameter must not unwrap.
        non_wrapped = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="input" string="true">{"city": "NYC"}</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="extra" string="true">ok</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        _, _, args2 = self._stream_collect(non_wrapped, chunk_size=2)
        self.assertEqual(args2[0], {"input": '{"city": "NYC"}', "extra": "ok"})

        # Missing closing parameter tag before </｜DSML｜invoke> still produces valid JSON.
        malformed = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜invoke>\n'
            "</｜DSML｜tool_calls>"
        )
        _, names3, args3 = self._stream_collect(malformed, chunk_size=2)
        self.assertEqual(names3, {0: "get_weather"})
        self.assertEqual(args3[0], {"city": "Paris"})

    def test_streaming_large_43kb_tool_call_is_linear_time(self):
        import time

        payload = ("x = 1234567890 # padding line with 'quotes' and \\slash\n") * 860
        self.assertGreater(len(payload), 43000)
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="Write">\n'
            '<｜DSML｜parameter name="file_path" string="true">/tmp/big.py</｜DSML｜parameter>\n'
            f'<｜DSML｜parameter name="content" string="true">{payload}</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="overwrite" string="false">true</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        start = time.monotonic()
        _, names, args = self._stream_collect(text, chunk_size=4)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0)
        self.assertEqual(names, {0: "Write"})
        self.assertEqual(
            args[0],
            {"file_path": "/tmp/big.py", "content": payload, "overwrite": True},
        )

    def test_renderer_runs_render_chat_in_thread_and_skips_double_render(self):
        from types import SimpleNamespace
        from unittest import mock

        from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer

        self.assertTrue(DeepseekV4Renderer.render_chat_in_thread)
        renderer = _make_renderer(None)
        request = SimpleNamespace()
        with (
            mock.patch.object(
                renderer, "render_chat", side_effect=AssertionError("called")
            ),
            mock.patch.object(renderer, "in_think_mode", return_value=True),
        ):
            parser = renderer._create_reasoning_parser(request)
            self.assertIsInstance(parser, ReasoningParser)
        with (
            mock.patch.object(
                renderer, "render_chat", side_effect=AssertionError("called")
            ),
            mock.patch.object(renderer, "in_think_mode", return_value=False),
        ):
            self.assertIsNone(renderer._create_reasoning_parser(request))


class _WordPieceMockTokenizer:
    """Tokenizer mapping integer IDs to multi-char tokens, including </think>."""

    eos_token_id = 0

    def __init__(self, special_tokens=None):
        self._id_to_tok = {0: "<｜end▁of▁sentence｜>"}
        self._tok_to_id = {"<｜end▁of▁sentence｜>": 0}
        self._next_id = 100
        for tok in special_tokens or []:
            self._register(tok)

    def _register(self, tok: str) -> int:
        if tok not in self._tok_to_id:
            tid = self._next_id
            self._next_id += 1
            self._tok_to_id[tok] = tid
            self._id_to_tok[tid] = tok
        return self._tok_to_id[tok]

    def encode(self, text: str, **kwargs):
        if not text:
            return []
        ids = []
        i = 0
        # Longest-first match against registered multi-char tokens, else per char.
        registered = sorted(self._tok_to_id.keys(), key=len, reverse=True)
        while i < len(text):
            matched = None
            for tok in registered:
                if tok and text.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                ids.append(self._tok_to_id[matched])
                i += len(matched)
            else:
                ids.append(self._register(text[i]))
                i += 1
        return ids

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._id_to_tok[tid] for tid in token_ids)

    def tokenize(self, text: str, **kwargs):
        return [self._id_to_tok[tid] for tid in self.encode(text)]


async def _token_stream(chunks_of_ids, input_len=10):
    total = 0
    for idx, step_ids in enumerate(chunks_of_ids):
        total += len(step_ids)
        aux = AuxInfo()
        aux.input_len = input_len
        aux.output_len = total
        aux.reuse_len = 0
        finished = torch.tensor([idx == len(chunks_of_ids) - 1], dtype=torch.bool)
        yield GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    output_ids=torch.tensor([step_ids], dtype=torch.int),
                    finished=finished,
                    aux_info=aux,
                )
            ]
        )


def _make_full_dsv4_renderer(
    encoding_module=None, tokenizer=None, think_end_tag="</think>"
):
    env_config = GenerateEnvConfig()
    env_config.think_mode = 0
    env_config.think_start_tag = "<think>"
    env_config.think_end_tag = think_end_tag
    renderer = DeepseekV4Renderer.__new__(DeepseekV4Renderer)
    renderer.encoding_module = encoding_module
    renderer.tokenizer = tokenizer or FakeTokenizer()
    renderer.think_mode = 0
    renderer.default_thinking_mode = ThinkingMode.DISABLED
    renderer.generate_env_config = env_config
    renderer.think_start_tag = "<think>"
    renderer.think_end_tag = think_end_tag
    renderer.model_type = "deepseek_v4"
    renderer.max_seq_len = 4096
    renderer.eos_token_id = getattr(renderer.tokenizer, "eos_token_id", 0)
    renderer.stop_words_id_list = []
    renderer.stop_words_str_list = []
    renderer.extra_stop_words = []
    renderer.extra_stop_word_ids_list = []
    renderer.render_config = None
    renderer.ckpt_path = ""
    return renderer


class DeepseekV4PromptControlTest(TestCase):
    def test_invalid_reasoning_effort_turbo_raises_ft_invalid_params(self):
        class CaptureEncoding:
            def encode_messages(self, messages, **kwargs):
                return "ok"

        renderer = _make_renderer(CaptureEncoding())
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="turbo",
            enable_thinking=True,
        )
        with self.assertRaises(FtRuntimeException) as ctx:
            renderer.render_chat(request)
        self.assertEqual(ctx.exception.exception_type, ExceptionType.INVALID_PARAMS)
        self.assertIn(
            "'reasoning_effort' must be one of: "
            "'low', 'medium', 'high', 'xhigh', 'max'",
            str(ctx.exception),
        )

    def test_clear_thinking_wires_drop_thinking_and_respects_overrides(self):
        captured_configs = []

        class CaptureEncoding:
            def encode_messages(self, messages, **kwargs):
                captured_configs.append(kwargs)
                return "prompt"

        renderer = _make_renderer(CaptureEncoding())
        messages = [{"role": "user", "content": "Hello"}]

        def _req(clear_thinking=None, **extra_kw):
            if "clear_thinking" in ChatCompletionRequest.model_fields:
                return ChatCompletionRequest(
                    messages=messages,
                    enable_thinking=True,
                    clear_thinking=clear_thinking,
                    **extra_kw,
                )
            req = ChatCompletionRequest(
                messages=messages, enable_thinking=True, **extra_kw
            )
            if clear_thinking is not None:
                object.__setattr__(req, "clear_thinking", clear_thinking)
            return req

        renderer.render_chat(_req(None))
        renderer.render_chat(_req(True))
        renderer.render_chat(_req(False))
        renderer.render_chat(_req(False, chat_template_kwargs={"drop_thinking": True}))
        renderer.render_chat(
            _req(
                False,
                chat_template_kwargs={"drop_thinking": True},
                extra_configs=GenerateConfig(
                    chat_template_kwargs={"drop_thinking": False}
                ),
            )
        )
        renderer.render_chat(_req(None, chat_template_kwargs={"clear_thinking": False}))

        self.assertEqual(
            [cfg["drop_thinking"] for cfg in captured_configs],
            [True, True, False, True, False, False],
        )


class DeepseekV4ParallelToolCallsTest(IsolatedAsyncioTestCase):
    @staticmethod
    def _multi_dsml_text():
        return (
            "Checking.\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="search">\n'
            '<｜DSML｜parameter name="query" string="true">weather</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">北京</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

    def test_auto_tool_choice_with_parallel_false_does_not_conflict_with_thinking(self):
        renderer = _make_full_dsv4_renderer()
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=_rtp_two_tools(),
            tool_choice="auto",
            parallel_tool_calls=False,
            enable_thinking=True,
        )
        config = GenerateConfig()
        renderer.apply_chat_completion_constraints(request, config)
        self.assertIsNone(config.structural_tag)

    async def test_non_streaming_official_parser_enforces_parallel_tool_calls_false(
        self,
    ):
        class OfficialMultiToolEncoding:
            def parse_message_from_completion_text(self, text, thinking_mode):
                return {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Let me check.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "杭州"}',
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "weather"}',
                            },
                        },
                    ],
                }

        renderer = _make_full_dsv4_renderer(encoding_module=OfficialMultiToolEncoding())
        renderer._generate_log_probs = AsyncMock(return_value=None)
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=_rtp_two_tools(),
            tool_choice="auto",
            parallel_tool_calls=False,
            enable_thinking=True,
        )
        status = ReasoningToolStreamStatus(
            request,
            renderer._create_detector(request),
            renderer._create_reasoning_parser(request),
        )
        status.delta_output_string = "dummy"
        aux = AuxInfo()
        aux.input_len = 10
        aux.output_len = 20
        output = Mock(spec=GenerateOutput)
        output.aux_info = aux

        delta = await renderer._process_reasoning_and_tool_calls(
            status, output, is_streaming=False
        )
        self.assertIsNotNone(delta)
        self.assertEqual(len(delta.output_str.tool_calls), 1)
        self.assertEqual(delta.output_str.tool_calls[0].index, 0)
        self.assertEqual(delta.output_str.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(delta.output_str.tool_calls[0].function.arguments),
            {"city": "杭州"},
        )
        self.assertTrue(status.generating_tool_call)

    async def test_streaming_and_non_streaming_enforce_parallel_tool_calls_false(self):
        dsml_text = self._multi_dsml_text()
        for tool_choice in ("auto", None):
            for is_streaming in (True, False):
                for parallel_flag in (False, True, None):
                    with self.subTest(
                        tool_choice=tool_choice,
                        is_streaming=is_streaming,
                        parallel_tool_calls=parallel_flag,
                    ):
                        tokenizer = FakeTokenizer()
                        renderer = _make_full_dsv4_renderer(tokenizer=tokenizer)
                        request = ChatCompletionRequest(
                            messages=[{"role": "user", "content": "Weather?"}],
                            tools=_rtp_two_tools(),
                            tool_choice=tool_choice,
                            parallel_tool_calls=parallel_flag,
                            stream=is_streaming,
                        )
                        all_ids = tokenizer.encode(dsml_text)
                        if is_streaming:
                            # Split chunks directly across invoke boundaries too.
                            step_size = 7
                            chunks = [
                                all_ids[i : i + step_size]
                                for i in range(0, len(all_ids), step_size)
                            ]
                        else:
                            chunks = [all_ids]

                        gen_cfg = GenerateConfig(
                            is_streaming=is_streaming,
                            thinking_mode=ThinkingMode.DISABLED,
                        )
                        responses = [
                            resp
                            async for resp in renderer.render_response_stream(
                                _token_stream(chunks), request, gen_cfg
                            )
                        ]

                        content_parts = []
                        tool_names = {}
                        tool_args = {}
                        finish_reasons = []
                        for resp in responses:
                            choice = resp.choices[0]
                            if choice.finish_reason is not None:
                                finish_reasons.append(choice.finish_reason)
                            if choice.delta.content:
                                content_parts.append(choice.delta.content)
                            for tc in choice.delta.tool_calls or []:
                                self.assertIsNotNone(tc.index)
                                if parallel_flag is False:
                                    self.assertEqual(tc.index, 0)
                                if tc.function.name:
                                    tool_names[tc.index] = tc.function.name
                                if tc.function.arguments:
                                    tool_args[tc.index] = (
                                        tool_args.get(tc.index, "")
                                        + tc.function.arguments
                                    )

                        combined_content = "".join(content_parts)
                        self.assertEqual(combined_content, "Checking.\n")
                        self.assertNotIn("<｜DSML｜", combined_content)
                        self.assertEqual(finish_reasons[-1], FinisheReason.tool_calls)
                        if parallel_flag is False:
                            self.assertEqual(tool_names, {0: "get_weather"})
                            self.assertEqual(list(tool_args.keys()), [0])
                            self.assertEqual(json.loads(tool_args[0]), {"city": "杭州"})
                        else:
                            self.assertEqual(
                                tool_names,
                                {0: "get_weather", 1: "search", 2: "get_weather"},
                            )
                            self.assertEqual(json.loads(tool_args[0]), {"city": "杭州"})
                            self.assertEqual(
                                json.loads(tool_args[1]), {"query": "weather"}
                            )
                            self.assertEqual(json.loads(tool_args[2]), {"city": "北京"})


class ThinkTokensAccuracyTest(IsolatedAsyncioTestCase):
    @staticmethod
    def _make_custom_renderer(tokenizer, think_end_tag="</think>"):
        env_config = GenerateEnvConfig()
        env_config.think_mode = 1
        env_config.think_start_tag = "<think>"
        env_config.think_end_tag = think_end_tag
        return CustomChatRenderer(
            tokenizer=tokenizer,
            renderer_params=RendererParams(
                model_type="qwen",
                max_seq_len=4096,
                eos_token_id=tokenizer.eos_token_id,
                stop_word_ids_list=[],
            ),
            generate_env_config=env_config,
        )

    async def test_custom_renderer_streaming_think_tokens_excludes_end_tag_and_content(
        self,
    ):
        for think_end_tag, end_pieces in (
            ("</think>", ["</think>"]),
            ("</think>\n\n", ["</think>", "\n\n"]),
        ):
            with self.subTest(think_end_tag=think_end_tag):
                tokenizer = _WordPieceMockTokenizer(
                    special_tokens=["</think>", "\n\n", "answer"]
                )
                renderer = self._make_custom_renderer(
                    tokenizer, think_end_tag=think_end_tag
                )
                think_tokens = [f"t{i:02d}" for i in range(40)]
                for tok in think_tokens:
                    tokenizer._register(tok)
                think_ids = [tokenizer.encode(t)[0] for t in think_tokens]
                end_ids = [tokenizer.encode(p)[0] for p in end_pieces]
                content_ids = tokenizer.encode("answer")

                # Single-token streaming: 40 think tokens, then end tag, then content
                step_chunks = [[tid] for tid in think_ids + end_ids + content_ids]
                request = ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hi"}],
                    enable_thinking=True,
                    stream=True,
                )
                cfg = GenerateConfig(
                    is_streaming=True, thinking_mode=ThinkingMode.ENABLED
                )
                responses = [
                    r
                    async for r in renderer.render_response_stream(
                        _token_stream(step_chunks), request, cfg
                    )
                ]
                final = responses[-1]
                self.assertEqual(
                    final.usage.completion_tokens_details.reasoning_tokens, 40
                )
                self.assertEqual(
                    final.usage.completion_tokens, 40 + len(end_ids) + len(content_ids)
                )

                # MTP batch crossing </think> + post-think content in one step
                mtp_chunks = [[tid] for tid in think_ids[:-1]] + [
                    [think_ids[-1]] + end_ids + content_ids
                ]
                mtp_responses = [
                    r
                    async for r in renderer.render_response_stream(
                        _token_stream(mtp_chunks), request, cfg
                    )
                ]
                self.assertEqual(
                    mtp_responses[-1].usage.completion_tokens_details.reasoning_tokens,
                    40,
                )

                # Non-streaming path
                non_stream_cfg = GenerateConfig(
                    is_streaming=False, thinking_mode=ThinkingMode.ENABLED
                )
                non_stream_resps = [
                    r
                    async for r in renderer.render_response_stream(
                        _token_stream([think_ids + end_ids + content_ids]),
                        request,
                        non_stream_cfg,
                    )
                ]
                self.assertEqual(
                    non_stream_resps[
                        -1
                    ].usage.completion_tokens_details.reasoning_tokens,
                    40,
                )

    async def test_deepseekv4_streaming_and_mtp_think_tokens_excludes_end_tag(self):
        tokenizer = _WordPieceMockTokenizer(special_tokens=["</think>", "answer"])
        renderer = _make_full_dsv4_renderer(
            tokenizer=tokenizer, think_end_tag="</think>"
        )
        think_tokens = [f"r{i:02d}" for i in range(40)]
        for tok in think_tokens:
            tokenizer._register(tok)
        think_ids = [tokenizer.encode(t)[0] for t in think_tokens]
        end_id = tokenizer.encode("</think>")[0]
        content_id = tokenizer.encode("answer")[0]

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=True,
            stream=True,
        )
        cfg = GenerateConfig(is_streaming=True, thinking_mode=ThinkingMode.ENABLED)

        for label, chunks in (
            ("single_token", [[tid] for tid in think_ids + [end_id, content_id]]),
            (
                "mtp_last_think_with_end",
                [[tid] for tid in think_ids[:-1]]
                + [[think_ids[-1], end_id], [content_id]],
            ),
            (
                "mtp_cross_end_and_content",
                [[tid] for tid in think_ids[:-1]]
                + [[think_ids[-1], end_id, content_id]],
            ),
            (
                "non_streaming",
                [think_ids + [end_id, content_id]],
            ),
        ):
            with self.subTest(label=label):
                is_stream = label != "non_streaming"
                run_cfg = GenerateConfig(
                    is_streaming=is_stream, thinking_mode=ThinkingMode.ENABLED
                )
                responses = [
                    r
                    async for r in renderer.render_response_stream(
                        _token_stream(chunks), request, run_cfg
                    )
                ]
                final = responses[-1]
                self.assertEqual(
                    final.usage.completion_tokens_details.reasoning_tokens, 40
                )
                reasoning_out = "".join(
                    r.choices[0].delta.reasoning_content or "" for r in responses
                )
                content_out = "".join(
                    r.choices[0].delta.content or "" for r in responses
                )
                self.assertEqual(reasoning_out, "".join(think_tokens))
                self.assertEqual(content_out, "answer")


@skipUnless(
    DSV4_ENCODING_PATH.exists(),
    f"DSV4 encoding module not found at {DSV4_ENCODING_PATH}",
)
class DeepseekV4ContentPartNormalizationTest(TestCase):
    def test_tool_and_chat_messages_with_content_part_lists(self):
        encoding_mod = _load_module("encoding_dsv4_content_test", DSV4_ENCODING_PATH)
        renderer = _make_renderer(encoding_mod)
        renderer.think_mode = True
        renderer.default_thinking_mode = ThinkingMode.ENABLED

        req = ChatCompletionRequest(
            messages=[
                {
                    "role": "system",
                    "content": [
                        ContentPart(type=ContentPartTypeEnum.text, text="sys prompt"),
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        ContentPart(
                            type=ContentPartTypeEnum.text, text="inspect image"
                        ),
                        ContentPart(
                            type=ContentPartTypeEnum.image_url,
                            image_url=ImageURL(url="data:image/png;base64,AAAA"),
                        ),
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        ContentPart(type=ContentPartTypeEnum.text, text="reading"),
                    ],
                    "tool_calls": [
                        ToolCall(
                            id="call_1",
                            type="function",
                            function=FunctionCall(
                                name="Read", arguments='{"file":"x.png"}'
                            ),
                        )
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        ContentPart(
                            type=ContentPartTypeEnum.text, text="file metadata"
                        ),
                        ContentPart(
                            type=ContentPartTypeEnum.image_url,
                            image_url=ImageURL(url="data:image/png;base64,BBBB"),
                        ),
                    ],
                },
            ]
        )
        prompt = renderer._build_prompt(req)
        self.assertIn("sys prompt", prompt)
        self.assertIn("inspect image\n\n[Unsupported image_url]", prompt)
        self.assertIn(
            "<tool_result>file metadata\n\n[Unsupported image_url]</tool_result>",
            prompt,
        )

    def test_normalize_message_content_preserves_string_and_none(self):
        self.assertEqual(
            DeepseekV4Renderer._normalize_message_content(RoleEnum.user, "hello"),
            "hello",
        )
        self.assertIsNone(
            DeepseekV4Renderer._normalize_message_content(RoleEnum.tool, None)
        )


if __name__ == "__main__":
    main()
