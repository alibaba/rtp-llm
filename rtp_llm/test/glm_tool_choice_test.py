import os
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    FinisheReason,
    FunctionCall,
    GPTFunctionDefinition,
    GPTToolDefinition,
    RoleEnum,
    ToolCall,
)
from rtp_llm.openai.renderers.chatglm47_renderer import ChatGlm47Renderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RendererParams,
    StreamStatus,
)
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
    ReasoningToolStreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm47_moe_detector import (
    Glm47MoeDetector,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput


def make_tool(name: str) -> GPTToolDefinition:
    return GPTToolDefinition(
        function=GPTFunctionDefinition(
            name=name,
            description=f"call {name}",
            parameters={"type": "object", "properties": {}},
        )
    )


def make_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role=RoleEnum.user, content="test")],
        tools=[make_tool("alpha"), make_tool("beta")],
        **kwargs,
    )


class CountingAsyncIterator:
    def __init__(self, values):
        self._values = iter(values)
        self.next_calls = 0
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.next_calls += 1
        return next(self._values)

    async def aclose(self):
        self.close_calls += 1


class ToolRenderLoopHarness(ReasoningToolBaseRenderer):
    render_response_stream = CustomChatRenderer.render_response_stream

    def __init__(self, request):
        self.status = ReasoningToolStreamStatus(
            request, Glm47MoeDetector(), None
        )

    async def _create_status_list(self, n, request):
        return [self.status]

    def in_think_mode(self, request):
        return False

    def should_process_think(self, request):
        return False

    def _extract_reasoning_content(self, parser, text, is_streaming):
        return "", text

    async def _generate_log_probs(self, status, output):
        return None

    async def _generate_first(self, n):
        return "first"

    async def _update_single_status(self, status, output, *args):
        status.output = output
        status.delta_output_string = output.tool_text
        return await self._process_reasoning_and_tool_calls(status, output, True)

    async def _generate_stream_response(self, delta_list, think_status_list):
        return delta_list

    async def _generate_extra_outputs(self, output, generate_config):
        return object()

    async def _flush_buffer(self, *args):
        self._finalize_tool_detector(self.status)
        return "flush"

    async def _generate_final(self, *args):
        return "final"


class ToolChoiceRequestTest(TestCase):
    def test_standard_tool_policy_fields_are_preserved(self):
        request = make_request(tool_choice="required", parallel_tool_calls=False)

        self.assertEqual(request.tool_choice, "required")
        self.assertFalse(request.parallel_tool_calls)
        self.assertEqual([tool.function.name for tool in request.effective_tools()], [
            "alpha",
            "beta",
        ])
        self.assertTrue(request.requires_tool_call())

    def test_named_tool_choice_filters_to_selected_function(self):
        request = make_request(
            tool_choice={"type": "function", "function": {"name": "beta"}}
        )

        self.assertEqual(
            [tool.function.name for tool in request.effective_tools()], ["beta"]
        )
        self.assertTrue(request.requires_tool_call())

    def test_none_hides_all_tools(self):
        request = make_request(tool_choice="none")

        self.assertEqual(request.effective_tools(), [])
        self.assertFalse(request.requires_tool_call())

    def test_required_without_tools_is_rejected(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role=RoleEnum.user, content="test")],
                tool_choice="required",
            )

    def test_unknown_named_tool_is_rejected(self):
        with self.assertRaises(ValidationError):
            make_request(
                tool_choice={"type": "function", "function": {"name": "missing"}}
            )

    def test_duplicate_tool_names_are_rejected(self):
        duplicate = make_tool("alpha")
        duplicate.function.parameters = {
            "type": "object",
            "properties": {"other": {"type": "integer"}},
        }

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role=RoleEnum.user, content="test")],
                tools=[make_tool("alpha"), duplicate],
            )

    def test_named_choice_cannot_select_duplicate_tool_names(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role=RoleEnum.user, content="test")],
                tools=[make_tool("alpha"), make_tool("alpha")],
                tool_choice={"type": "function", "function": {"name": "alpha"}},
            )

    def test_none_does_not_bypass_duplicate_tool_validation(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role=RoleEnum.user, content="test")],
                tools=[make_tool("alpha"), make_tool("alpha")],
                tool_choice="none",
            )

    def test_tool_names_remain_case_sensitive(self):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            tools=[make_tool("alpha"), make_tool("Alpha")],
        )

        self.assertEqual(
            [tool.function.name for tool in request.tools or []], ["alpha", "Alpha"]
        )

    def test_user_template_with_enabled_tools_is_rejected(self):
        with self.assertRaises(ValidationError):
            make_request(user_template="custom", tool_choice="auto")

    def test_user_template_with_tool_choice_none_is_allowed(self):
        request = make_request(user_template="custom", tool_choice="none")

        self.assertEqual(request.effective_tools(), [])


class ToolChoiceRendererTest(IsolatedAsyncioTestCase):
    @staticmethod
    def _renderer_for_prompt():
        renderer = object.__new__(ReasoningToolBaseRenderer)
        renderer.chat_template = (
            "{% for message in messages %}[{{ message.role }}]{{ message.content }}"
            "{% endfor %}|{% for tool in tools %}{{ tool.function.name }} {% endfor %}"
        )
        renderer._preprocess_messages = (
            ReasoningToolBaseRenderer._preprocess_messages.__get__(renderer)
        )
        renderer._customize_jinja_env = (
            ReasoningToolBaseRenderer._customize_jinja_env.__get__(renderer)
        )
        return renderer

    def test_named_choice_restricts_prompt_and_adds_required_instruction(self):
        renderer = self._renderer_for_prompt()
        request = make_request(
            tool_choice={"type": "function", "function": {"name": "beta"}},
            parallel_tool_calls=False,
        )

        prompt = renderer._build_prompt(request)

        self.assertNotIn("alpha", prompt)
        self.assertIn("beta", prompt)
        self.assertIn("must call", prompt.lower())
        self.assertIn("at most one tool", prompt.lower())

    def test_template_kwargs_cannot_override_policy_tools_or_messages(self):
        renderer = self._renderer_for_prompt()
        request = make_request(
            tool_choice={"type": "function", "function": {"name": "beta"}},
            chat_template_kwargs={
                "tools": [make_tool("alpha").model_dump(mode="json")],
                "messages": [{"role": "user", "content": "injected"}],
            },
        )

        prompt = renderer._build_prompt(request)

        self.assertNotIn("alpha", prompt)
        self.assertNotIn("injected", prompt)
        self.assertIn("test", prompt)
        self.assertIn("beta", prompt)

    def test_default_tool_policy_preserves_tools_without_instruction(self):
        renderer = self._renderer_for_prompt()

        prompt = renderer._build_prompt(make_request())

        self.assertIn("alpha", prompt)
        self.assertIn("beta", prompt)
        self.assertNotIn("must call", prompt.lower())
        self.assertNotIn("at most one tool", prompt.lower())

    def test_real_glm_template_only_exposes_named_tool_and_policy(self):
        tokenizer_path = os.path.join(
            os.path.dirname(__file__), "model_test/fake_test/testdata/glm45/tokenizer"
        )
        tokenizer = BaseTokenizer(tokenizer_path)
        renderer = ChatGlm47Renderer(
            tokenizer,
            RendererParams(
                model_type="glm_5",
                max_seq_len=1024,
                eos_token_id=tokenizer.eos_token_id or 0,
                stop_word_ids_list=[],
            ),
            GenerateEnvConfig(),
            RenderConfig(),
        )

        prompt = renderer.render_chat(
            make_request(
                tool_choice={"type": "function", "function": {"name": "beta"}},
                parallel_tool_calls=False,
            )
        ).rendered_prompt

        self.assertNotIn('"name": "alpha"', prompt)
        self.assertIn('"name": "beta"', prompt)
        self.assertIn("must call", prompt.lower())
        self.assertIn("at most one tool", prompt.lower())

    async def test_none_uses_plain_status_without_detector(self):
        renderer = Mock(spec=ReasoningToolBaseRenderer)
        renderer._create_status_list = (
            ReasoningToolBaseRenderer._create_status_list.__get__(renderer)
        )
        renderer.in_think_mode = Mock(return_value=False)
        renderer._create_detector = Mock()
        renderer._create_reasoning_parser = Mock(return_value=None)

        (status,) = await renderer._create_status_list(
            1, make_request(tool_choice="none")
        )

        self.assertIsInstance(status, StreamStatus)
        self.assertNotIsInstance(status, ReasoningToolStreamStatus)
        renderer._create_detector.assert_not_called()

    async def test_none_with_thinking_keeps_reasoning_status_without_detector(self):
        reasoning_parser = Mock()
        renderer = Mock(spec=ReasoningToolBaseRenderer)
        renderer._create_status_list = (
            ReasoningToolBaseRenderer._create_status_list.__get__(renderer)
        )
        renderer.in_think_mode = Mock(return_value=True)
        renderer._create_detector = Mock()
        renderer._create_reasoning_parser = Mock(return_value=reasoning_parser)

        (status,) = await renderer._create_status_list(
            1, make_request(tool_choice="none")
        )

        self.assertIsInstance(status, ReasoningToolStreamStatus)
        self.assertIsNone(status.detector)
        self.assertIs(status.reasoning_parser, reasoning_parser)
        renderer._create_detector.assert_not_called()

    async def test_named_choice_keeps_declared_tools_for_response_validation(self):
        detector = Mock(spec=BaseFormatDetector)
        renderer = Mock(spec=ReasoningToolBaseRenderer)
        renderer._create_status_list = (
            ReasoningToolBaseRenderer._create_status_list.__get__(renderer)
        )
        renderer.in_think_mode = Mock(return_value=False)
        renderer._create_detector = Mock(return_value=detector)
        renderer._create_reasoning_parser = Mock(return_value=None)
        selected = Mock(name="selected")
        request = make_request(
            tool_choice={"type": "function", "function": {"name": "beta"}}
        )

        with patch(
            "rtp_llm.openai.renderers.reasoning_tool_base_renderer."
            "rtp_tools_to_sglang_tools",
            return_value=[selected],
        ) as convert:
            (status,) = await renderer._create_status_list(1, request)

        convert.assert_called_once()
        converted_input = convert.call_args.args[0]
        self.assertEqual(
            [tool.function.name for tool in converted_input], ["alpha", "beta"]
        )
        self.assertEqual(status.sglang_tools, (selected,))
        renderer._create_detector.assert_called_once_with(request)

    def test_strict_tool_validation_is_isolated_per_request_status(self):
        strict_detector = Glm47MoeDetector()
        auto_detector = Glm47MoeDetector()

        ReasoningToolStreamStatus(
            make_request(tool_choice="required"), strict_detector, None
        )
        ReasoningToolStreamStatus(make_request(), auto_detector, None)

        self.assertTrue(strict_detector.strict_tool_validation)
        self.assertFalse(auto_detector.strict_tool_validation)

    def test_required_choice_fails_if_generation_ends_without_tool(self):
        renderer = Mock(spec=ReasoningToolBaseRenderer)
        renderer._finalize_tool_detector = (
            ReasoningToolBaseRenderer._finalize_tool_detector.__get__(renderer)
        )
        detector = Mock(spec=BaseFormatDetector)
        detector.finalize_streaming.return_value = StreamingParseResult()
        status = ReasoningToolStreamStatus(
            make_request(tool_choice="required"), detector, None, ()
        )
        status.finish_reason = FinisheReason.stop

        with self.assertRaises(FtRuntimeException) as context:
            renderer._finalize_tool_detector(status)

        self.assertEqual(
            context.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
        )
        self.assertIn("required tool call", context.exception.message)

    async def test_parallel_false_rejects_multiple_tool_calls(self):
        renderer = object.__new__(ReasoningToolBaseRenderer)
        renderer._extract_reasoning_content = Mock(return_value=("", "tool text"))
        renderer._extract_tool_calls_content = AsyncMock(
            return_value=(
                [
                    ToolCall(
                        index=0,
                        id="call-0",
                        type="function",
                        function=FunctionCall(name="alpha", arguments="{}"),
                    ),
                    ToolCall(
                        index=1,
                        id="call-1",
                        type="function",
                        function=FunctionCall(name="beta", arguments="{}"),
                    ),
                ],
                "",
            )
        )
        renderer._generate_log_probs = AsyncMock(return_value=None)
        status = ReasoningToolStreamStatus(
            make_request(parallel_tool_calls=False), Mock(spec=BaseFormatDetector), None
        )
        output = Mock(spec=GenerateOutput)
        output.aux_info = AuxInfo(input_len=1, output_len=2, reuse_len=0)

        with self.assertRaises(FtRuntimeException) as context:
            await renderer._process_reasoning_and_tool_calls(status, output, True)

        self.assertEqual(
            context.exception.exception_type, ExceptionType.EXECUTION_EXCEPTION
        )
        self.assertIn("parallel tool calls", context.exception.message)

    @staticmethod
    def _real_glm_renderer():
        renderer = object.__new__(ReasoningToolBaseRenderer)
        renderer._extract_reasoning_content = Mock(return_value=("", ""))
        renderer._extract_tool_calls_content = (
            ReasoningToolBaseRenderer._extract_tool_calls_content.__get__(renderer)
        )
        renderer._process_reasoning_and_tool_calls = (
            ReasoningToolBaseRenderer._process_reasoning_and_tool_calls.__get__(
                renderer
            )
        )
        renderer._finalize_tool_detector = (
            ReasoningToolBaseRenderer._finalize_tool_detector.__get__(renderer)
        )
        renderer._generate_log_probs = AsyncMock(return_value=None)
        return renderer

    @staticmethod
    def _output():
        output = Mock(spec=GenerateOutput)
        output.aux_info = AuxInfo(input_len=1, output_len=2, reuse_len=0)
        return output

    async def test_parallel_false_rejects_second_tool_in_later_chunk(self):
        renderer = self._real_glm_renderer()
        status = ReasoningToolStreamStatus(
            make_request(parallel_tool_calls=False), Glm47MoeDetector(), None
        )
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)

        status.delta_output_string = "<tool_call>alpha</tool_call>"
        first = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), True
        )
        self.assertEqual(first.output_str.tool_calls[0].function.name, "alpha")
        self.assertEqual(status.finish_reason, FinisheReason.tool_calls)

        status.delta_output_string = "<tool_call>beta</tool_call>"
        with self.assertRaises(FtRuntimeException) as context:
            await renderer._process_reasoning_and_tool_calls(
                status, self._output(), True
            )

        self.assertIn("parallel tool calls", context.exception.message)

    async def test_parallel_false_rejects_complete_then_partial_in_same_chunk(self):
        for partial_second in ("<tool_call>beta", "<tool_"):
            with self.subTest(partial_second=partial_second):
                renderer = self._real_glm_renderer()
                status = ReasoningToolStreamStatus(
                    make_request(parallel_tool_calls=False),
                    Glm47MoeDetector(),
                    None,
                )
                renderer._extract_reasoning_content.side_effect = (
                    lambda _, text, __: ("", text)
                )
                status.delta_output_string = (
                    "<tool_call>alpha</tool_call>" + partial_second
                )

                with self.assertRaises(FtRuntimeException) as context:
                    await renderer._process_reasoning_and_tool_calls(
                        status, self._output(), True
                    )

                self.assertIn("parallel tool calls", context.exception.message)
                self.assertEqual(status.tool_call_keys, set())

    async def test_parallel_false_rejects_normalized_complete_then_partial(self):
        for partial_second in ("<tool_call>beta", "<tool_"):
            with self.subTest(partial_second=partial_second):
                renderer = self._real_glm_renderer()
                renderer._process_normalized_tokens = (
                    ReasoningToolBaseRenderer._process_normalized_tokens.__get__(
                        renderer
                    )
                )
                status = ReasoningToolStreamStatus(
                    make_request(parallel_tool_calls=False),
                    Glm47MoeDetector(),
                    None,
                )
                renderer._extract_reasoning_content.side_effect = (
                    lambda _, text, __: ("", text)
                )

                async def process_delta(status, text, output, *args, **kwargs):
                    status.delta_output_string = text
                    return await renderer._process_reasoning_and_tool_calls(
                        status, output, True
                    )

                renderer._process_single_token_delta = AsyncMock(
                    side_effect=process_delta
                )
                normalizer = Mock()
                normalizer.normalize_tokens.return_value = iter(
                    ["<tool_call>alpha</tool_call>", partial_second]
                )

                with self.assertRaises(FtRuntimeException) as context:
                    await renderer._process_normalized_tokens(
                        normalizer,
                        status,
                        [1, 2],
                        self._output(),
                        [],
                        [],
                        True,
                    )

                self.assertIn("parallel tool calls", context.exception.message)
                self.assertEqual(status.tool_call_keys, set())
                self.assertFalse(status.generating_tool_call)
                self.assertIsNone(status.finish_reason)

    async def test_parallel_false_stops_source_after_first_complete_tool(self):
        request = make_request(parallel_tool_calls=False)
        output = lambda text: SimpleNamespace(
            tool_text=text,
            aux_info=AuxInfo(input_len=1, output_len=1, reuse_len=0),
        )
        source = CountingAsyncIterator(
            [
                SimpleNamespace(
                    generate_outputs=[output("<tool_call>alpha</tool_call>")]
                ),
                SimpleNamespace(
                    generate_outputs=[output("<tool_call>beta</tool_call>")]
                ),
            ]
        )
        renderer = ToolRenderLoopHarness(request)

        responses = [
            response
            async for response in renderer.render_response_stream(
                source, request, GenerateConfig(is_streaming=True)
            )
        ]

        self.assertEqual(source.next_calls, 1)
        self.assertEqual(responses[0], "first")
        self.assertEqual(
            responses[1][0].output_str.tool_calls[0].function.name, "alpha"
        )
        self.assertEqual(responses[2:], ["flush", "final"])

    async def test_named_choice_rejects_different_generated_tool_before_emit(self):
        renderer = self._real_glm_renderer()
        request = make_request(
            tool_choice={"type": "function", "function": {"name": "beta"}}
        )
        status = ReasoningToolStreamStatus(request, Glm47MoeDetector(), None)
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.delta_output_string = "<tool_call>alpha</tool_call>"

        with self.assertRaises(FtRuntimeException) as context:
            await renderer._process_reasoning_and_tool_calls(
                status, self._output(), True
            )

        self.assertIn("outside tool_choice", context.exception.message)
        self.assertEqual(status.tool_call_keys, set())

    async def test_named_choice_rejects_undeclared_tool_before_stream_emit(self):
        renderer = self._real_glm_renderer()
        request = make_request(
            tool_choice={"type": "function", "function": {"name": "beta"}}
        )
        status = ReasoningToolStreamStatus(request, Glm47MoeDetector(), None)
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.delta_output_string = "<tool_call>hallucinated</tool_call>"

        with self.assertRaises(FtRuntimeException) as context:
            await renderer._process_reasoning_and_tool_calls(
                status, self._output(), True
            )

        self.assertIn("undefined function", context.exception.message)
        self.assertEqual(status.tool_call_keys, set())

    async def test_named_choice_rejects_undeclared_tool_non_streaming(self):
        renderer = self._real_glm_renderer()
        request = make_request(
            tool_choice={"type": "function", "function": {"name": "beta"}}
        )
        status = ReasoningToolStreamStatus(request, Glm47MoeDetector(), None)
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.delta_output_string = "<tool_call>hallucinated</tool_call>"

        with self.assertRaises(FtRuntimeException) as context:
            await renderer._process_reasoning_and_tool_calls(
                status, self._output(), False
            )

        self.assertIn("undefined function", context.exception.message)
        self.assertEqual(status.tool_call_keys, set())

    async def test_parallel_false_rejects_valid_then_undeclared_tool(self):
        renderer = self._real_glm_renderer()
        status = ReasoningToolStreamStatus(
            make_request(parallel_tool_calls=False), Glm47MoeDetector(), None
        )
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.delta_output_string = (
            "<tool_call>alpha</tool_call>"
            "<tool_call>hallucinated</tool_call>"
        )

        with self.assertRaises(FtRuntimeException) as context:
            await renderer._process_reasoning_and_tool_calls(
                status, self._output(), True
            )

        self.assertIn("undefined function", context.exception.message)
        self.assertEqual(status.tool_call_keys, set())

    async def test_parallel_false_preserves_length_when_tool_closes_at_limit(self):
        renderer = self._real_glm_renderer()
        status = ReasoningToolStreamStatus(
            make_request(parallel_tool_calls=False), Glm47MoeDetector(), None
        )
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.finish_reason = FinisheReason.length
        status.delta_output_string = "<tool_call>alpha</tool_call>"

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), True
        )

        self.assertEqual(delta.output_str.tool_calls[0].function.name, "alpha")
        self.assertEqual(status.finish_reason, FinisheReason.length)

    async def test_non_streaming_length_discards_incomplete_tool_transaction(self):
        for parallel_tool_calls in (True, False):
            with self.subTest(parallel_tool_calls=parallel_tool_calls):
                renderer = self._real_glm_renderer()
                status = ReasoningToolStreamStatus(
                    make_request(parallel_tool_calls=parallel_tool_calls),
                    Glm47MoeDetector(),
                    None,
                )
                renderer._extract_reasoning_content.side_effect = (
                    lambda _, text, __: ("", text)
                )
                status.finish_reason = FinisheReason.length
                status.delta_output_string = "<tool_call>alpha"

                delta = await renderer._process_reasoning_and_tool_calls(
                    status, self._output(), False
                )

                self.assertIsNone(delta)
                self.assertEqual(status.delta_output_string, "")
                renderer._finalize_tool_detector(status)
                self.assertEqual(status.finish_reason, FinisheReason.length)

    async def test_streaming_length_discards_partial_second_tool(self):
        renderer = self._real_glm_renderer()
        status = ReasoningToolStreamStatus(
            make_request(parallel_tool_calls=False), Glm47MoeDetector(), None
        )
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.finish_reason = FinisheReason.length
        status.delta_output_string = (
            "<tool_call>alpha</tool_call><tool_call>beta"
        )

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), True
        )

        self.assertEqual(delta.output_str.tool_calls[0].function.name, "alpha")
        self.assertEqual(status.finish_reason, FinisheReason.length)
        renderer._finalize_tool_detector(status)

    async def test_non_streaming_incomplete_tool_without_length_is_terminal(self):
        renderer = self._real_glm_renderer()
        status = ReasoningToolStreamStatus(
            make_request(), Glm47MoeDetector(), None
        )
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.finish_reason = FinisheReason.stop
        status.delta_output_string = "<tool_call>alpha"

        with self.assertRaises(FtRuntimeException) as context:
            await renderer._process_reasoning_and_tool_calls(
                status, self._output(), False
            )

        self.assertIn("incomplete tool call", context.exception.message)

    async def test_required_non_streaming_length_still_requires_committed_tool(self):
        renderer = self._real_glm_renderer()
        status = ReasoningToolStreamStatus(
            make_request(tool_choice="required"), Glm47MoeDetector(), None
        )
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.finish_reason = FinisheReason.length
        status.delta_output_string = "<tool_call>alpha"

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), False
        )
        self.assertIsNone(delta)

        with self.assertRaises(FtRuntimeException) as context:
            renderer._finalize_tool_detector(status)

        self.assertIn("required tool call", context.exception.message)

    async def test_required_rejects_empty_tool_name_before_emit(self):
        renderer = self._real_glm_renderer()
        status = ReasoningToolStreamStatus(
            make_request(tool_choice="required"), Glm47MoeDetector(), None
        )
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.delta_output_string = "<tool_call></tool_call>"

        with self.assertRaises(FtRuntimeException) as context:
            await renderer._process_reasoning_and_tool_calls(
                status, self._output(), True
            )

        self.assertIn("empty function name", context.exception.message)
        self.assertEqual(status.tool_call_keys, set())

    async def test_required_text_then_eof_is_terminal_not_success(self):
        renderer = self._real_glm_renderer()
        status = ReasoningToolStreamStatus(
            make_request(tool_choice="required"), Glm47MoeDetector(), None
        )
        renderer._extract_reasoning_content.side_effect = lambda _, text, __: ("", text)
        status.delta_output_string = "ordinary answer"

        delta = await renderer._process_reasoning_and_tool_calls(
            status, self._output(), True
        )
        self.assertIsNone(delta)
        self.assertEqual(status.delta_output_string, "ordinary answer")
        status.finish_reason = FinisheReason.stop

        with self.assertRaises(FtRuntimeException) as context:
            renderer._finalize_tool_detector(status)

        self.assertIn("required tool call", context.exception.message)


if __name__ == "__main__":
    import unittest

    unittest.main()
