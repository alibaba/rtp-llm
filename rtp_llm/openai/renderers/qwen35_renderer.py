import json
from typing import Any

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ThinkingMode
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
from rtp_llm.openai.renderers.qwen_vl_renderer import Qwen2VLRenderer


class Qwen35Renderer(Qwen3CoderRenderer, Qwen2VLRenderer):
    def _format_tool_call_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return {}
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed_arguments if isinstance(parsed_arguments, dict) else {}

    def _render_messages(
        self, request: ChatCompletionRequest, add_vision_id: bool
    ) -> PromptWithMMInput:
        mode = request.get_explicit_thinking_mode()
        enable_thinking = None if mode is None else mode != ThinkingMode.DISABLED
        return Qwen2VLRenderer._render_messages(
            self, request, add_vision_id, enable_thinking
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        prompt_and_mm_input = self._render_messages(
            request,
            request.extra_configs.add_vision_id if request.extra_configs else True,
        )
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


class Qwen4Renderer(Qwen35Renderer):
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        # QSA side caches currently retain text positions only. Reject before
        # image preprocessing/RPC so a request cannot stop the backend loop.
        for message in request.messages:
            if isinstance(message.content, list) and any(
                part.type != "text" for part in message.content
            ):
                raise FtRuntimeException(
                    ExceptionType.UNSUPPORTED_OPERATION,
                    "Qwen3.8-Flash-Next QSA currently supports text inputs only",
                )
        return super().render_chat(request)


register_renderer("qwen35_moe", Qwen35Renderer)
register_renderer("qwen35_dense", Qwen35Renderer)
register_renderer("qwen35_dense_mtp", Qwen35Renderer)
register_renderer("qwen35_moe_mtp", Qwen35Renderer)
register_renderer("qwen4_exp", Qwen4Renderer)
register_renderer("qwen4_exp_mtp", Qwen4Renderer)
