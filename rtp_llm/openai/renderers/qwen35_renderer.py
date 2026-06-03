import json
from typing import Any

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
        return Qwen2VLRenderer._render_messages(self, request, add_vision_id)

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


register_renderer("qwen35_moe", Qwen35Renderer)
register_renderer("qwen35_dense", Qwen35Renderer)
