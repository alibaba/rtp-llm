import os
import logging
from typing import Optional, List, Dict, Any, Union, Callable

from transformers import PreTrainedTokenizer

from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from maga_transformer.openai.renderers.qwen_renderer import QwenRenderer
from maga_transformer.openai.renderers.qwen_vl_renderer import QwenVLRenderer
from maga_transformer.openai.renderers.llava_renderer import LlavaRenderer
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer
from maga_transformer.openai.renderers.llama_template_renderer import LlamaTemplateRenderer
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from maga_transformer.models.base_model import BaseTokenizer

class ChatRendererFactory():
    def __init__(self):
        pass

    @staticmethod
    def _maybe_change_model_type(params: RendererParams):
        if params.model_type.startswith("chat_glm_"):
            params.model_type.replace("chat_glm_", "chatglm")

        # Online service params does not distinguish baichuan and baichuan2 model.
        # However, they use different llama template.
        # We assume baichuan2 model is always used.
        if params.model_type == "baichuan":
            logging.info(f"model_type {params.model_type} is changed to baichuan2.")
            params.model_type = "baichuan2"

        return params

    @staticmethod
    def get_renderer(
        tokenizer: Union[PreTrainedTokenizer, BaseTokenizer],
        params: RendererParams,
    ) -> CustomChatRenderer:
        # renderer priority: special cases > tokenizer.chat_template
        #                    > model customized renderer (e.g. Qwen, which implemented function call)
        #                    > LlamaTemplateRenderer > default chat template
        # tokenizer.chat_template has the highest priority because it might be user customized.
        #
        # The special cases are:
        # 1. Multimodal models (e.g. QwenVL): need to deal with images.

        params = ChatRendererFactory._maybe_change_model_type(params)
        model_type = params.model_type

        if model_type == "qwen_vl":
            assert (isinstance(tokenizer, PreTrainedTokenizer))
            return QwenVLRenderer(tokenizer, params)
        elif model_type == "llava":
            return LlavaRenderer(tokenizer, params)

        try:
            if tokenizer.chat_template != None:
                logging.info(f"tokenizer has chat_template [{tokenizer.chat_template}], use it.")
                return BasicRenderer(tokenizer, params)
        except AttributeError:
            # tokenizer may has no chat_template property
            pass

        if isinstance(tokenizer, QWenTokenizer):
            return QwenRenderer(tokenizer, params)

        try:
            assert(isinstance(tokenizer, PreTrainedTokenizer))
            return LlamaTemplateRenderer(tokenizer, params)
        except AssertionError as e:
            # model_type is not supported by LlamaTemplateRenderer
            logging.info(f"llama template is not applicable to model_type {model_type}: {e}")
            pass

        logging.info(f"tokenizer {tokenizer} falls back to basic renderer.")
        return BasicRenderer(tokenizer, params)

