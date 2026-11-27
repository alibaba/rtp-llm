import json
import os
from typing import Any

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ThinkingMode
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
        self._prompt_backend = None
        if (
            renderer_params.model_type == "atom_qwen4_exp"
            and os.getenv("ROLE_TYPE", "PDFUSION").upper() in ("PREFILL", "PDFUSION")
        ):
            import torch
            from transformers import PreTrainedTokenizerFast

            hf = getattr(tokenizer, "tokenizer", None)
            if (
                torch.version.hip
                and type(hf) is PreTrainedTokenizerFast
                and getattr(tokenizer.encode, "__func__", None) is BaseTokenizer.encode
                and hasattr(hf.backend_tokenizer, "encode_batch_fast")
            ):
                self._prompt_backend = hf.backend_tokenizer
                self.render_chat_in_thread = True
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
        backend = self._prompt_backend
        # The renderer consumes IDs only. Keep HF's default state handling
        # when another caller configured padding, truncation or special tokens.
        if (
            backend is not None
            and backend.padding is None
            and backend.truncation is None
            and backend.encode_special_tokens
            == self.tokenizer.tokenizer.split_special_tokens
        ):
            input_ids = backend.encode_batch_fast([prompt_and_mm_input.prompt])[0].ids
        else:
            input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


class Qwen4Renderer(Qwen35Renderer):
    def _render_messages(
        self, request: ChatCompletionRequest, add_vision_id: bool
    ) -> PromptWithMMInput:
        return Qwen2VLRenderer._render_messages(
            self,
            request,
            add_vision_id,
            self.resolve_thinking_mode(request) != ThinkingMode.DISABLED,
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        for message in request.messages:
            if isinstance(message.content, list) and any(
                part.type not in ("text", "image_url", "video_url")
                for part in message.content
            ):
                raise FtRuntimeException(
                    ExceptionType.UNSUPPORTED_OPERATION,
                    "Qwen3.8-Flash-Next supports text, image_url and video_url inputs",
                )
        return super().render_chat(request)


register_renderer("qwen35_moe", Qwen35Renderer)
register_renderer("qwen35_dense", Qwen35Renderer)
register_renderer("qwen35_dense_mtp", Qwen35Renderer)
register_renderer("qwen35_moe_mtp", Qwen35Renderer)
register_renderer("qwen4_exp", Qwen4Renderer)
register_renderer("qwen4_exp_mtp", Qwen4Renderer)
