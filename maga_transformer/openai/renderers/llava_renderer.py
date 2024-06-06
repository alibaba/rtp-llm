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
    LLAMA_3 = auto()

@dataclass
class Conversation:
    system_content: str
    roles: Dict[RoleEnum, str]
    sep_style: SeparatorStyle
    seps: List[str]
    connector: List[str]

    def render_messages(self, messages: List[ChatMessage], tokenizer) -> PromptWithImages:
        prompt: str = ""
        images: List[str] = []

        if self.sep_style == SeparatorStyle.LLAMA_3:
            chat_template_messages = [{"role": "system", "content": self.system_content}]
            for message in messages:
                role = self.roles[message.role]
                content = message.content
                if isinstance(content, str):
                    chat_template_messages.append({"role": role, "content": content})
                else:
                    now_prompt = ""
                    for content_part in content:
                        if content_part.type == ContentPartTypeEnum.text:
                            assert (isinstance(content_part.text, str))
                            now_prompt += content_part.text
                        elif content_part.type == ContentPartTypeEnum.image_url:
                            assert (content_part.image_url != None)
                            images.append(content_part.image_url.url)
                            now_prompt = f"<image>\n" + now_prompt
                    chat_template_messages.append({"role": role, "content": now_prompt})

            return PromptWithImages(tokenizer.apply_chat_template(chat_template_messages, tokenize=False, add_generation_prompt=True),
                                    images)

        if messages[0].role != RoleEnum.system:
            prompt = self.system_content + prompt + self.seps[0]

        for index, message in enumerate(messages):
            if isinstance(message.content, str):
                prompt += f"{self.roles[message.role]}: {message.content}"
            elif isinstance(message.content, list):
                now_prompt = ""
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert (isinstance(content_part.text, str))
                        now_prompt += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert (content_part.image_url != None)
                        images.append(content_part.image_url.url)
                        now_prompt = f"<image>\n" + now_prompt
                prompt += f"{self.roles[message.role]}" + self.connector[0] + now_prompt
            if self.sep_style == SeparatorStyle.TWO:
                prompt += self.seps[index % 2]
            else:
                prompt += self.seps[0]
        prompt += self.roles[RoleEnum.assistant] + self.connector[1]
        return PromptWithImages(prompt, images)

conv_llava_v0 = Conversation(
    system_content="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles={RoleEnum.user: "Human", RoleEnum.assistant: "Assistant", RoleEnum.system: "System"},
    sep_style=SeparatorStyle.SINGLE,
    seps=["###"],
    connector=[": ", ":"]
)

conv_llava_v1 = Conversation(
    system_content="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles={RoleEnum.user: "USER", RoleEnum.assistant: "ASSISTANT", RoleEnum.system: "SYSTEM"},
    sep_style=SeparatorStyle.TWO,
    seps=[" ", "</s>"],
    connector=[": ", ":"]
)

conv_llava_llama = Conversation(
    system_content="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
    roles={RoleEnum.user: "<|start_header_id|>user", RoleEnum.assistant: "<|start_header_id|>assistant", RoleEnum.system: "<|start_header_id|>system"},
    sep_style=SeparatorStyle.LLAMA_3,
    seps=[""],
    connector=[": ", ":"]
)

conv_qwen = Conversation(
    system_content="<|im_start|>system\nYou are a helpful assistant.",
    roles={RoleEnum.user: "<|im_start|>user", RoleEnum.assistant: "<|im_start|>assistant", RoleEnum.system: "<|im_start|>system"},
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|im_end|>\n"],
    connector=["\n", "\n"]
)

conv_templates = {
    "llava_v0": conv_llava_v0,
    "llava_v1": conv_llava_v1,
    "llava_llama3": conv_llava_llama,
    "llava_qwen": conv_qwen
}

class LlavaRenderer(CustomChatRenderer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)

    def _get_conv_template(self, model_name: str) -> Conversation:
        if "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "llama3" in model_name.lower():
            conv_mode = "llava_llama3"
        elif "next" in model_name.lower():
            conv_mode = "llava_qwen"
        else:
            conv_mode = "llava_v0"

        return conv_templates[conv_mode]

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithImages:
        ckpt_path: str = os.environ['CHECKPOINT_PATH']
        model_name: str = ckpt_path.split('?')[0] # oss style path
        model_name = model_name.strip('/').split('/')[-1]
        conv_template = self._get_conv_template(model_name)

        return conv_template.render_messages(messages, self.tokenizer)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_images = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_images.prompt)
        return RenderedInputs(input_ids=input_ids, input_images=prompt_and_images.image_urls, rendered_prompt=prompt_and_images.prompt)

register_renderer('llava', LlavaRenderer)