from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator
from dataclasses import dataclass

from .conversation import Conversation, get_conv_template

from transformers import PreTrainedTokenizer

from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, \
    ChatCompletionRequest, RoleEnum
from maga_transformer.openai.renderers.llama_template import Template, get_template_and_fix_tokenizer
from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams, \
    StreamResponseObject, RenderedInputs
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice

class FastChatRenderer(CustomChatRenderer):
    def __init__(self, tokenizer: PreTrainedTokenizer, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)
        self.conv_template = get_conv_template(renderer_params.model_type)

        if isinstance(self.conv_template.stop_str, list):
            self.add_extra_stop_words(self.conv_template.stop_str)
        elif isinstance(self.conv_template.stop_str, str):
            self.add_extra_stop_words([self.conv_template.stop_str])
        self.add_extra_stop_word_ids([[id] for id in self.conv_template.stop_token_ids])

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:

        return RenderedInputs(input_ids=[])
