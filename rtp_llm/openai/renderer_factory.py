import copy
import logging
from typing import Any, Optional

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.renderer_factory_register import _renderer_factory
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from rtp_llm.openai.renderers.fast_chat_renderer import FastChatRenderer
from rtp_llm.openai.renderers.llama_template_renderer import LlamaTemplateRenderer


class ChatRendererFactory:
    def __init__(self):
        pass

    @staticmethod
    def try_get_imported_renderer(
        tokenizer: BaseTokenizer,
        params: RendererParams,
        generate_env_config: GenerateEnvConfig,
        render_config: Optional[RenderConfig] = None,
        ckpt_path: Optional[str] = None,
        misc_config: Optional[Any] = None,
        vit_config: Optional[Any] = None,
    ) -> Optional[CustomChatRenderer]:
        try:
            return FastChatRenderer(
                tokenizer,
                params,
                generate_env_config,
                render_config,
                ckpt_path,
                misc_config,
                vit_config,
            )
        except KeyError:
            pass

        try:
            return LlamaTemplateRenderer(
                tokenizer,
                params,
                generate_env_config,
                render_config,
                ckpt_path,
                misc_config,
                vit_config,
            )
        except AssertionError as e:  # assertion at llama_template.py:229
            pass
        return None

    @staticmethod
    def get_renderer(
        tokenizer: BaseTokenizer,
        params: RendererParams,
        generate_env_config: GenerateEnvConfig,
        render_config: Optional[RenderConfig] = None,
        ckpt_path: Optional[str] = None,
        misc_config: Optional[Any] = None,
        vit_config: Optional[Any] = None,
    ) -> CustomChatRenderer:
        """Get renderer for tokenizer and params.

        Args:
            tokenizer: BaseTokenizer instance.
            params: RendererParams object.
            generate_env_config: GenerateEnvConfig object.
            render_config: RenderConfig object.
            ckpt_path: Checkpoint path string.
            misc_config: MiscellaneousConfig object.
            vit_config: VitConfig object.
        """
        # renderer priority:  `MODEL_TEMPLATE_TYPE` env for llama template or fastchat conversation
        #                    > tokenizer.chat_template
        #                    > model customized renderer (e.g. Qwen, which implemented function call)
        #                    > try get template from `MODEL_TYPE`
        #                    > transformers default chat template

        global _renderer_factory
        model_template_type = render_config.model_template_type
        if model_template_type:
            new_params = copy.deepcopy(params)
            new_params.model_type = model_template_type
            logging.info(
                f"Renderer factory try found MODEL_TEMPLATE_TYPE: {model_template_type}, try get predefined renderer."
            )
            renderer = ChatRendererFactory.try_get_imported_renderer(
                tokenizer,
                new_params,
                generate_env_config,
                render_config,
                ckpt_path,
                misc_config,
                vit_config,
            )
            if renderer:
                return renderer
            # Try to get renderer from dedicated renderer factory
            elif model_template_type in _renderer_factory:
                logging.info(
                    f"Renderer factory found MODEL_TEMPLATE_TYPE [{model_template_type}] in dedicated renderer factory, use this."
                )
                return _renderer_factory[model_template_type](
                    tokenizer,
                    new_params,
                    generate_env_config,
                    render_config,
                    ckpt_path,
                    misc_config,
                    vit_config,
                )
            else:
                # exit only when model_template_type is specified but not found
                raise AttributeError(
                    f"specified MODEL_TEMPLATE_TYPE {model_template_type} not supported."
                )

        # renderer in _renderer_factory all have higher priority:
        # qwen needs to deal with function call, multimodal models need to add image token
        if params.model_type in _renderer_factory:
            logging.info(
                f"Renderer factory found model type [{params.model_type}] has dedicated renderer, use this."
            )
            return _renderer_factory[params.model_type](
                tokenizer,
                params,
                generate_env_config,
                render_config,
                ckpt_path,
                misc_config,
                vit_config,
            )

        try:
            if tokenizer.chat_template != None:
                logging.info(
                    f"Renderer factory found tokenizer has chat_template [{tokenizer.chat_template}], use it."
                )
                return BasicRenderer(
                    tokenizer,
                    params,
                    generate_env_config,
                    render_config,
                    ckpt_path,
                    misc_config,
                    vit_config,
                )
            else:
                pass
        except AttributeError:
            pass

        logging.info(
            f"Renderer factory try get predefined renderer via model type [{params.model_type}]"
        )
        imported_template_renderer = ChatRendererFactory.try_get_imported_renderer(
            tokenizer,
            params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
        if imported_template_renderer:
            logging.info(
                f"found renderer from imported template for [{params.model_type}]"
            )
            return imported_template_renderer

        logging.warn(
            f"Renderer factory found model [{params.model_type}] falls back to basic renderer, this is typically unwanted."
        )
        return BasicRenderer(
            tokenizer,
            params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
