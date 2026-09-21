import json
from typing import Any

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs, RendererParams
from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
from rtp_llm.openai.renderers.qwen_vl_renderer import Qwen2VLRenderer


class Qwen35Renderer(Qwen3CoderRenderer, Qwen2VLRenderer):
    # QwenRenderer adds these for its legacy tokenizer.  In Qwen3.5 they decode
    # as ordinary text, so they must not become generation stop sequences.
    _LEGACY_EXTRA_STOP_WORD_IDS = ((37763, 367, 25), (151643,))

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
        self.extra_stop_word_ids_list = [
            stop_word_ids
            for stop_word_ids in self.extra_stop_word_ids_list
            if tuple(stop_word_ids) not in self._LEGACY_EXTRA_STOP_WORD_IDS
        ]

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
register_renderer("qwen35_dense_mtp", Qwen35Renderer)
register_renderer("qwen35_moe_mtp", Qwen35Renderer)
