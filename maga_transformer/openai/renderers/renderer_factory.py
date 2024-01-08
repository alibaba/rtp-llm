import logging
from typing import Optional, List, Dict, Any, Union, Callable

from transformers import PreTrainedTokenizer

from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from maga_transformer.openai.renderers.qwen_renderer import QwenRenderer
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from maga_transformer.models.base_model import BaseTokenizer

class ChatRendererFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_renderer(
        tokenizer: Union[PreTrainedTokenizer, BaseTokenizer],
        params: RendererParams,
    ) -> CustomChatRenderer:
        # renderer priority: tokenizer.chat_template > CustomRenderer
        #                    > tokenizer.default_chat_template > DEFAULT_CHAT_API_TEMPLATE
        # tokenizer.chat_template has the highest priority because it might be user customized.
        try:
            if tokenizer.chat_template != None:
                logging.info(f"tokenizer has chat_template [{tokenizer.chat_template}], use it.")
                return BasicRenderer(tokenizer, params)
        except:
            pass

        if isinstance(tokenizer, QWenTokenizer):
            return QwenRenderer(tokenizer, params)
        else:
            logging.info(f"tokenizer {tokenizer} falls back to basic renderer.")
            return BasicRenderer(tokenizer, params)

