import copy
import json
import re
import logging
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator

from transformers import PreTrainedTokenizer
from dataclasses import dataclass

from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, \
    ChatCompletionRequest, RoleEnum, FunctionCall
from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams, \
    StreamResponseObject, RenderedInputs
<<<<<<< HEAD
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer, PromptWithImages
=======
>>>>>>> feat: add llama_template_renderer
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, UsageInfo, \
    ContentPart, ContentPartTypeEnum

class QwenVLRenderer(CustomChatRenderer):
    def __init__(self, tokenizer: PreTrainedTokenizer, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithImages:
        prompt = ""
        images = []
        if messages[0].role != RoleEnum.system:
            messages = [ChatMessage(role=RoleEnum.system, content="You are a helpful assistant.")] + messages

        for message in messages:
            if isinstance(message.content, str):
                prompt += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
            elif isinstance(message.content, list):
                prompt += f"<|im_start|>{message.role}\n"
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert (isinstance(content_part.text, str))
                        prompt += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert (isinstance(content_part.image_url, str))
                        images.append(content_part.image_url)
                        prompt += f"Picture {len(images)}: <img>{content_part.image_url}</img>\n"
                prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return PromptWithImages(prompt=prompt, image_urls=images)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_images = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_images.prompt)
        return RenderedInputs(input_ids=input_ids, input_images=prompt_and_images.image_urls)
