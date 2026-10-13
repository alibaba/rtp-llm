"""gpt-oss (harmony format) chat renderer.

Input side: the harmony chat template loads from the checkpoint's
``chat_template.jinja`` (the base class reads it from the tokenizer path),
with ``tools`` / ``reasoning_effort`` kwargs forwarded from the request.

Output side: assistant turns emit an optional ``analysis`` channel
(reasoning_content) then a ``final`` channel (content) — split by the
HarmonyChannelDetector. ``<|return|>`` (tokenizer eos) ends the turn.

Tool-call OUTPUT parsing (harmony ``<|call|>`` protocol) is not wired yet:
calls currently surface as raw text in the content channel; input-side
tool definitions render correctly through the template.
"""

from typing import Optional

from typing_extensions import override

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser


class GptOssRenderer(ReasoningToolBaseRenderer):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
        # Safety stop markers for harmony turns; <|return|> arrives via the
        # tokenizer eos, <|end|> closes intermediate channels.
        self.add_extra_stop_words(["<|end|>", "<|return|>"])

    @override
    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        """gpt-oss always opens with the analysis channel by default."""
        return True

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        # harmony <|call|> tool-call output parser: not wired yet.
        return None

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        return ReasoningParser(model_type="gpt-oss", force_reasoning=True)

    @override
    def _customize_jinja_env(self, env) -> None:
        super()._customize_jinja_env(env)
        # harmony template calls strftime_now("%Y-%m-%d") in the system
        # header; the base renderer only provides raise_exception.
        import datetime

        env.globals.setdefault(
            "strftime_now", lambda fmt: datetime.datetime.now().strftime(fmt)
        )


register_renderer("gpt_oss", GptOssRenderer)
