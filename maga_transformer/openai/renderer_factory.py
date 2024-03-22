import os
import logging
import copy
from typing import Optional, List, Dict, Any, Union, Callable

from transformers import PreTrainedTokenizerBase

from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer
from maga_transformer.openai.renderers.llama_template_renderer import LlamaTemplateRenderer
from maga_transformer.openai.renderers.fast_chat_renderer import FastChatRenderer
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from maga_transformer.tokenizer.tokenization_qwen2 import Qwen2Tokenizer
from maga_transformer.openai.renderer_factory_register import _renderer_factory

class ChatRendererFactory():
    def __init__(self):
        pass

    @staticmethod
    def try_get_imported_renderer(
        tokenizer: PreTrainedTokenizerBase,
        params: RendererParams,
    ) -> Optional[CustomChatRenderer]:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            return None

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
        tokenizer: PreTrainedTokenizerBase,
        params: RendererParams,
    ) -> CustomChatRenderer:
        # renderer priority:  `MODEL_TEMPLATE_TYPE` env for llama template or fastchat conversation
        #                    > tokenizer.chat_template
        #                    > model customized renderer (e.g. Qwen, which implemented function call)
        #                    > try get template from `MODEL_TYPE`
        #                    > transformers default chat template

        model_template_type = os.environ.get("MODEL_TEMPLATE_TYPE", None)
        if model_template_type:
            new_params = copy.deepcopy(params)
            new_params.model_type = model_template_type
            logging.info(f"try get renderer from MODEL_TEMPLATE_TYPE: {model_template_type}")
            renderer = ChatRendererFactory.try_get_imported_renderer(tokenizer, new_params)
            if renderer:
                return renderer
            else:
                raise AttributeError(f"specified MODEL_TEMPLATE_TYPE {model_template_type} not supported.")

        # qwen needs to deal with function call, so it has higher priority than simple template
        global _renderer_factory
        if ('qwen' in params.model_type) and ('qwen_vl' not in params.model_type):
            return _renderer_factory[params.model_type](tokenizer, params)

        try:
            if tokenizer.chat_template != None:
                logging.info(f"tokenizer has chat_template [{tokenizer.chat_template}], use it.")
                return BasicRenderer(tokenizer, params)
        except AttributeError:
            # tokenizer may has no chat_template property
            pass

        if params.model_type in _renderer_factory:
            return _renderer_factory[params.model_type](tokenizer, params)

        imported_template_renderer = ChatRendererFactory.try_get_imported_renderer(tokenizer, params)
        if imported_template_renderer:
            logging.info(f"found renderer from imported template for [{params.model_type}]")
            return imported_template_renderer

        logging.warn(f"model [{params.model_type}] falls back to basic renderer, this is typically unwanted.")
        return BasicRenderer(tokenizer, params)

