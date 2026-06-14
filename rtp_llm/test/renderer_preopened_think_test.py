"""Unit tests for the renderer-side decisions touched by the grammar PR cleanup:

* CustomChatRenderer.prompt_preopens_think — does the rendered chat prompt
  leave the assistant turn inside an open `<think>` block? (Qwen3.5 with
  enable_thinking=true emits `<think>\\n`; legacy templates / disabled
  thinking emit `<think>\\n\\n</think>\\n\\n` and must NOT trigger this.)
* Qwen3CoderRenderer._create_reasoning_parser — picks
  ReasoningParser(model_type="qwen3-thinking") iff the prompt pre-opens
  `<think>`, else ReasoningParser(model_type="qwen3"); returns None when
  not in think mode.
* ReasoningToolBaseRenderer._create_status_list — returns
  ReasoningToolStreamStatus when tools or think_mode are active and
  logprobs is off; otherwise plain StreamStatus.
"""

from typing import List, Optional
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock

from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
    StreamStatus,
)
from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
    ReasoningToolStreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser


def _make_tokenizer() -> MagicMock:
    """A minimal tokenizer stub: only decode/encode/chat_template are touched
    by the renderer code under test. decode returns '' so stop_words_str_list
    construction stays safe; chat_template is empty (not used — render_chat
    is overridden in tests)."""
    tok = MagicMock()
    tok.decode = MagicMock(return_value="")
    tok.encode = MagicMock(return_value=[])
    tok.chat_template = ""
    tok.path = ""
    return tok


def _make_renderer_params() -> RendererParams:
    return RendererParams(
        model_type="qwen3",
        max_seq_len=2048,
        eos_token_id=0,
        stop_word_ids_list=[],
    )


def _think_env_config(enabled: bool) -> GenerateEnvConfig:
    cfg = GenerateEnvConfig()
    cfg.think_mode = bool(enabled)
    cfg.think_start_tag = "<think>"
    cfg.think_end_tag = "</think>"
    return cfg


class _StubRenderer(CustomChatRenderer):
    """CustomChatRenderer with render_chat short-circuited to a fixed prompt
    so prompt_preopens_think can be exercised without driving Jinja2."""

    def __init__(self, rendered_prompt: str, *, think_enabled: bool):
        super().__init__(
            tokenizer=_make_tokenizer(),
            renderer_params=_make_renderer_params(),
            generate_env_config=_think_env_config(think_enabled),
        )
        self._rendered_prompt = rendered_prompt

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        return RenderedInputs(input_ids=[], rendered_prompt=self._rendered_prompt)


class _StubQwen3CoderRenderer(Qwen3CoderRenderer):
    """Qwen3CoderRenderer with chat-template setup bypassed and the
    pre-opened-think decision injectable per test."""

    def __init__(self, *, think_enabled: bool, preopens: bool):
        self._preopens = preopens
        super().__init__(
            tokenizer=_make_tokenizer(),
            renderer_params=_make_renderer_params(),
            generate_env_config=_think_env_config(think_enabled),
        )

    def _setup_chat_template(self) -> None:
        self.chat_template = ""

    def prompt_preopens_think(self, request: ChatCompletionRequest) -> bool:
        return self._preopens


class _StubReasoningToolRenderer(ReasoningToolBaseRenderer):
    """Concrete ReasoningToolBaseRenderer subclass for status-list tests.
    Lets us drive in_think_mode independently of generate_env_config so
    the (tools, think, logprobs) matrix is testable in isolation."""

    def __init__(self, *, think_enabled: bool):
        self._think_enabled = think_enabled
        super().__init__(
            tokenizer=_make_tokenizer(),
            renderer_params=_make_renderer_params(),
            generate_env_config=_think_env_config(think_enabled),
        )

    def _setup_chat_template(self) -> None:
        self.chat_template = ""

    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        return self._think_enabled


def _make_tools() -> list:
    """Minimal-but-valid GPTToolDefinition list. function.description and
    function.parameters are required by the pydantic schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "stub",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_request(
    *, tools: Optional[List[dict]] = None, logprobs: bool = False
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role=RoleEnum.user, content="test")],
        tools=tools,
        logprobs=logprobs,
    )


class PromptPreopensThinkTest(TestCase):
    """CustomChatRenderer.prompt_preopens_think — boundary cases for the
    Qwen3.5 vs legacy assistant-prefix template distinction."""

    def test_qwen35_enable_thinking_open_block_is_preopened(self):
        # Qwen3.5 with enable_thinking=true ends the assistant prefix with
        # `<think>\n` — prompt rstrip() == "...<think>".
        prompt = "<|im_start|>assistant\n<think>\n"
        renderer = _StubRenderer(prompt, think_enabled=True)
        self.assertTrue(renderer.prompt_preopens_think(_make_request()))

    def test_qwen35_enable_thinking_false_closed_block_not_preopened(self):
        # Qwen3.5 with enable_thinking=false emits a closed block; the
        # rendered prompt ends with `</think>\n\n`, which must NOT match.
        prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        renderer = _StubRenderer(prompt, think_enabled=True)
        self.assertFalse(renderer.prompt_preopens_think(_make_request()))

    def test_no_think_tag_at_end_not_preopened(self):
        prompt = "<|im_start|>assistant\nHello"
        renderer = _StubRenderer(prompt, think_enabled=True)
        self.assertFalse(renderer.prompt_preopens_think(_make_request()))

    def test_think_mode_disabled_short_circuits(self):
        # think_mode=False must short-circuit to False even if the prompt
        # happens to look pre-opened.
        prompt = "<|im_start|>assistant\n<think>\n"
        renderer = _StubRenderer(prompt, think_enabled=False)
        self.assertFalse(renderer.prompt_preopens_think(_make_request()))


class Qwen3CoderReasoningParserTest(TestCase):
    """Qwen3CoderRenderer._create_reasoning_parser — must pick the right
    ReasoningParser flavor based on (in_think_mode, prompt_preopens_think)."""

    def test_in_think_and_preopened_picks_qwen3_thinking(self):
        # ReasoningParser(model_type="qwen3-thinking") forces force_reasoning=True
        # in the underlying Qwen3Detector, so the detector enters reasoning mode
        # immediately — matching a prompt that already opened `<think>`.
        renderer = _StubQwen3CoderRenderer(think_enabled=True, preopens=True)
        parser = renderer._create_reasoning_parser(_make_request())
        self.assertIsInstance(parser, ReasoningParser)
        assert parser is not None
        self.assertTrue(parser.detector._in_reasoning)

    def test_in_think_and_not_preopened_picks_qwen3(self):
        # Plain "qwen3" → force_reasoning=False; detector waits for `<think>`
        # in the model output before flipping into reasoning mode.
        renderer = _StubQwen3CoderRenderer(think_enabled=True, preopens=False)
        parser = renderer._create_reasoning_parser(_make_request())
        self.assertIsInstance(parser, ReasoningParser)
        assert parser is not None
        self.assertFalse(parser.detector._in_reasoning)

    def test_not_in_think_returns_none(self):
        renderer = _StubQwen3CoderRenderer(think_enabled=False, preopens=True)
        self.assertIsNone(renderer._create_reasoning_parser(_make_request()))


class CreateStatusListTest(IsolatedAsyncioTestCase):
    """ReasoningToolBaseRenderer._create_status_list — the (tools, think,
    logprobs) matrix collapses to: ReasoningToolStreamStatus when (tools
    or think) AND not logprobs; plain StreamStatus otherwise."""

    async def test_tools_only_yields_reasoning_tool_status(self):
        renderer = _StubReasoningToolRenderer(think_enabled=False)
        request = _make_request(tools=_make_tools())
        statuses = await renderer._create_status_list(2, request)
        self.assertEqual(len(statuses), 2)
        for s in statuses:
            self.assertIsInstance(s, ReasoningToolStreamStatus)

    async def test_think_only_yields_reasoning_tool_status(self):
        renderer = _StubReasoningToolRenderer(think_enabled=True)
        request = _make_request()
        statuses = await renderer._create_status_list(1, request)
        self.assertEqual(len(statuses), 1)
        self.assertIsInstance(statuses[0], ReasoningToolStreamStatus)

    async def test_tools_and_think_yields_reasoning_tool_status(self):
        renderer = _StubReasoningToolRenderer(think_enabled=True)
        request = _make_request(tools=_make_tools())
        statuses = await renderer._create_status_list(1, request)
        self.assertIsInstance(statuses[0], ReasoningToolStreamStatus)

    async def test_logprobs_overrides_to_plain_stream_status(self):
        # logprobs=True forces plain StreamStatus regardless of tools/think:
        # tool-call / reasoning extraction would interfere with token-level
        # logprob accounting.
        renderer = _StubReasoningToolRenderer(think_enabled=True)
        request = _make_request(
            tools=_make_tools(),
            logprobs=True,
        )
        statuses = await renderer._create_status_list(1, request)
        self.assertIs(type(statuses[0]), StreamStatus)


if __name__ == "__main__":
    main()
