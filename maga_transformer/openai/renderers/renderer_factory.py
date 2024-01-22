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
from maga_transformer.openai.renderers.fast_chat_renderer import FastChatRenderer
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from maga_transformer.models.base_model import BaseTokenizer

class ChatRendererFactory():
    def __init__(self):
        pass

    @staticmethod
    def try_get_imported_renderer(
        tokenizer: Union[PreTrainedTokenizer, BaseTokenizer],
        params: RendererParams,
    ) -> Optional[CustomChatRenderer]:
        assert (isinstance(tokenizer, PreTrainedTokenizer))
        model_type = params.model_type
        try:
            logging.info(f"try fast chat conversation with [{model_type}]")
            return FastChatRenderer(tokenizer, params)
        except KeyError:
            logging.info(f"[{model_type}] not found in fast chat conversation, try llama template")
            pass
        try:
            return LlamaTemplateRenderer(tokenizer, params)
        except AssertionError as e: # assertion at llama_template.py:229
            logging.info(f"[{model_type}] not found in llama template.")
            pass
        return None

    @staticmethod
    def get_renderer(
        tokenizer: Union[PreTrainedTokenizer, BaseTokenizer],
        params: RendererParams,
    ) -> CustomChatRenderer:
        # renderer priority: multi modal renderers
        #                    > `MODEL_TEMPLATE_TYPE` env for llama template or fastchat conversation
        #                    > tokenizer.chat_template
        #                    > model customized renderer (e.g. Qwen, which implemented function call)
        #                    > try get template from `MODEL_TYPE`
        #                    > transformers default chat template

        if params.model_type == "qwen_vl":
            assert (isinstance(tokenizer, PreTrainedTokenizer))
            return QwenVLRenderer(tokenizer, params)
        elif params.model_type == "llava":
            assert (isinstance(tokenizer, BaseTokenizer))
            return LlavaRenderer(tokenizer, params)

        model_template_type = os.environ.get("MODEL_TEMPLATE_TYPE", None)
        if model_template_type:
            params.model_type = model_template_type
            logging.info(f"try get renderer from MODEL_TEMPLATE_TYPE: {model_template_type}")
            renderer = ChatRendererFactory.try_get_imported_renderer(tokenizer, params)
            if renderer:
                return renderer
            else:
                raise AttributeError(f"specified MODEL_TEMPLATE_TYPE {model_template_type} not supported.")

        try:
            if tokenizer.chat_template != None:
                logging.info(f"tokenizer has chat_template [{tokenizer.chat_template}], use it.")
                return BasicRenderer(tokenizer, params)
        except AttributeError:
            # tokenizer may has no chat_template property
            pass

        if isinstance(tokenizer, QWenTokenizer):
            return QwenRenderer(tokenizer, params)

        imported_template_renderer = ChatRendererFactory.try_get_imported_renderer(tokenizer, params)
        if imported_template_renderer:
            logging.info(f"found renderer from imported template for [{params.model_type}]")
            return imported_template_renderer

        logging.warn(f"model [{params.model_type}] falls back to basic renderer, this is typically unwanted.")
        return BasicRenderer(tokenizer, params)

