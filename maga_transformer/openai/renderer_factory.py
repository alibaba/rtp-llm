import os
import logging
import copy
from typing import Optional, List, Dict, Any, Union, Callable

from transformers import PreTrainedTokenizerBase

from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer
from maga_transformer.openai.renderers.llama_template_renderer import LlamaTemplateRenderer
from maga_transformer.openai.renderers.fast_chat_renderer import FastChatRenderer
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

        try:
            return FastChatRenderer(tokenizer, params)
        except KeyError:
            pass

        try:
            return LlamaTemplateRenderer(tokenizer, params)
        except AssertionError as e: # assertion at llama_template.py:229
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
            logging.info(f"Renderer factory try found MODEL_TEMPLATE_TYPE: {model_template_type}, try get predefined renderer.")
            renderer = ChatRendererFactory.try_get_imported_renderer(tokenizer, new_params)
            if renderer:
                return renderer
            else:
                raise AttributeError(f"specified MODEL_TEMPLATE_TYPE {model_template_type} not supported.")

        # renderer in _renderer_factory all have higher priority:
        # qwen needs to deal with function call, multimodal models need to add image token
        global _renderer_factory
        if params.model_type in _renderer_factory:
            logging.info(f"Renderer factory found model type [{params.model_type}] has dedicated renderer, use this.")
            return _renderer_factory[params.model_type](tokenizer, params)

        try:
            if tokenizer.chat_template != None:
                logging.info(f"Renderer factory found tokenizer has chat_template [{tokenizer.chat_template}], use it.")
                return BasicRenderer(tokenizer, params)
            else:
                pass
        except AttributeError:
            pass

        logging.info(f"Renderer factory try get predefined renderer via model type [{params.model_type}]")
        imported_template_renderer = ChatRendererFactory.try_get_imported_renderer(tokenizer, params)
        if imported_template_renderer:
            logging.info(f"found renderer from imported template for [{params.model_type}]")
            return imported_template_renderer

        logging.warn(f"Renderer factory found model [{params.model_type}] falls back to basic renderer, this is typically unwanted.")
        return BasicRenderer(tokenizer, params)

