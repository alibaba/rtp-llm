import copy
import logging
from typing import Optional

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.renderer_factory_register import _renderer_factory
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from rtp_llm.openai.renderers.custom_modal_proxy_renderer import CustomModalProxyRenderer # Import the proxy
from rtp_llm.openai.renderers.fast_chat_renderer import FastChatRenderer
from rtp_llm.openai.renderers.llama_template_renderer import LlamaTemplateRenderer


class ChatRendererFactory:
    def __init__(self):
        pass

    @staticmethod
    def try_get_imported_renderer(
        tokenizer: BaseTokenizer,
        params: RendererParams,
    ) -> Optional[CustomChatRenderer]:
        try:
            return FastChatRenderer(tokenizer, params)
        except KeyError:
            pass

        try:
            return LlamaTemplateRenderer(tokenizer, params)
        except AssertionError as e:  # assertion at llama_template.py:229
            pass
        return None

    @staticmethod
    def get_renderer(
        tokenizer: BaseTokenizer,
        params: RendererParams,
    ) -> CustomChatRenderer:
        # renderer priority:  `MODEL_TEMPLATE_TYPE` env for llama template or fastchat conversation
        #                    > tokenizer.chat_template
        #                    > model customized renderer (e.g. Qwen, which implemented function call)
        #                    > try get template from `MODEL_TYPE`
        #                    > transformers default chat template

        renderer_instance: Optional[CustomChatRenderer] = None

        model_template_type = StaticConfig.render_config.model_template_type
        if model_template_type:
            new_params = copy.deepcopy(params)
            new_params.model_type = model_template_type
            logging.info(
                f"Renderer factory try found MODEL_TEMPLATE_TYPE: {model_template_type}, try get predefined renderer."
            )
            renderer_instance = ChatRendererFactory.try_get_imported_renderer(
                tokenizer, new_params
            )
            if renderer_instance:
                # Fall through to the end for potential wrapping
                pass
            else:
                raise AttributeError(
                    f"specified MODEL_TEMPLATE_TYPE {model_template_type} not supported."
                )

        if renderer_instance is None: # Only proceed if not already found by model_template_type
            # renderer in _renderer_factory all have higher priority:
            # qwen needs to deal with function call, multimodal models need to add image token
            global _renderer_factory
            if params.model_type in _renderer_factory:
                logging.info(
                    f"Renderer factory found model type [{params.model_type}] has dedicated renderer, use this."
                )
                renderer_instance = _renderer_factory[params.model_type](tokenizer, params)
            
            if renderer_instance is None: # Only proceed if not already found by registered factory
                try:
                    if tokenizer.chat_template != None:
                        logging.info(
                            f"Renderer factory found tokenizer has chat_template [{tokenizer.chat_template}], use it."
                        )
                        renderer_instance = BasicRenderer(tokenizer, params)
                    else:
                        pass
                except AttributeError:
                    pass

            if renderer_instance is None: # Only proceed if not already found by tokenizer.chat_template
                logging.info(
                    f"Renderer factory try get predefined renderer via model type [{params.model_type}]"
                )
                imported_template_renderer = ChatRendererFactory.try_get_imported_renderer(
                    tokenizer, params
                )
                if imported_template_renderer:
                    logging.info(
                        f"found renderer from imported template for [{params.model_type}]"
                    )
                    renderer_instance = imported_template_renderer

        if renderer_instance is None: # Final fallback if nothing else found
            logging.warn(
                f"Renderer factory found model [{params.model_type}] falls back to basic renderer, this is typically unwanted."
            )
            renderer_instance = BasicRenderer(tokenizer, params)
        
        # --- Custom Modal Proxy Wrapping Logic ---
        if params.custom_modal_config:
            logging.info(f"Wrapping renderer {type(renderer_instance).__name__} with CustomModalProxyRenderer for custom modal support.")
            return CustomModalProxyRenderer(tokenizer, params, renderer_instance)

        return renderer_instance
