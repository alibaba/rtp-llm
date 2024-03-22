import copy
import json
import re
import logging
import os
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator
from enum import Enum, auto
from transformers import PreTrainedTokenizerBase

from dataclasses import dataclass

from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, \
    ChatCompletionRequest, RoleEnum, FunctionCall
from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams, \
    StreamResponseObject, RenderedInputs
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer, PromptWithImages
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, UsageInfo, \
    ContentPart, ContentPartTypeEnum
from maga_transformer.openai.renderer_factory_register import register_renderer

class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()

@dataclass
class Conversation:
    system_content: str
    roles: Dict[RoleEnum, str]
    sep_style: SeparatorStyle
    seps: List[str]

    def render_messages(self, messages: List[ChatMessage]) -> PromptWithImages:
        prompt: str = ""
        images: List[str] = []
        if messages[0].role != RoleEnum.system:
            prompt = self.system_content + prompt + self.seps[0]

        for index, message in enumerate(messages):
            if isinstance(message.content, str):
                prompt += f"{self.roles[message.role]}: {message.content}"
            elif isinstance(message.content, list):
                prompt += f"{self.roles[message.role]}: "
                now_prompt = ""
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert (isinstance(content_part.text, str))
                        now_prompt += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert (content_part.image_url != None)
                        images.append(content_part.image_url.url)
                        now_prompt = f"<image>\n" + now_prompt
                if self.sep_style == SeparatorStyle.SINGLE:
                    now_prompt += self.seps[0]
                elif self.sep_style == SeparatorStyle.TWO:
                    now_prompt += self.seps[index % 2]
                prompt += now_prompt
        prompt += self.roles[RoleEnum.assistant] + ":"
        return PromptWithImages(prompt, images)

conv_llava_v0 = Conversation(
    system_content="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles={RoleEnum.user: "Human", RoleEnum.assistant: "Assistant", RoleEnum.system: "System"},
    sep_style=SeparatorStyle.SINGLE,
    seps=["###"]
)

conv_llava_v1 = Conversation(
    system_content="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles={RoleEnum.user: "USER", RoleEnum.assistant: "ASSISTANT", RoleEnum.system: "SYSTEM"},
    sep_style=SeparatorStyle.TWO,
    seps=[" ", "</s>"]
)

conv_templates = {
    "llava_v0": conv_llava_v0,
    "llava_v1": conv_llava_v1
}

class LlavaRenderer(CustomChatRenderer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)

    def _get_conv_template(self, model_name: str) -> Conversation:
        if "v1" in model_name.lower():
            conv_mode = "llava_v1"
        else:
            conv_mode = "llava_v0"

        return conv_templates[conv_mode]

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithImages:
        ckpt_path: str = os.environ['CHECKPOINT_PATH']
        model_name: str = ckpt_path.split('?')[0] # oss style path
        model_name = model_name.strip('/').split('/')[-1]
        conv_template = self._get_conv_template(model_name)

        return conv_template.render_messages(messages)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_images = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_images.prompt)
        return RenderedInputs(input_ids=input_ids, input_images=prompt_and_images.image_urls)

register_renderer('llava', LlavaRenderer)