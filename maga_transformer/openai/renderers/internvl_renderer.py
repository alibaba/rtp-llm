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
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer, PromptWithMMInput
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, UsageInfo, \
    ContentPart, ContentPartTypeEnum
from maga_transformer.openai.renderer_factory_register import register_renderer
from maga_transformer.openai.renderers.llava_renderer import Conversation, SeparatorStyle
from maga_transformer.utils.fuser import fetch_remote_file_to_local
from maga_transformer.utils.multimodal_util import MMUrlType

import dataclasses
from enum import IntEnum, auto
from typing import Any, Dict, List, Tuple, Union

class InternVLConversation(Conversation):
    def render_messages(self, messages: List[ChatMessage], video_frame_num: int = 8) -> PromptWithMMInput:
        prompt: str = ""
        urls: List[str] = []
        types: List[MMUrlType] = []

        if messages[0].role != RoleEnum.system:
            prompt = self.roles[RoleEnum.system] + self.system_content + self.seps[0]

        for index, message in enumerate(messages):
            if isinstance(message.content, str):
                prompt += f"{self.roles[message.role]}{self.connector[0]}{message.content}"
            elif isinstance(message.content, list):
                now_prompt = ""
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert (isinstance(content_part.text, str))
                        now_prompt += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert (content_part.image_url != None)
                        urls.append(content_part.image_url.url)
                        now_prompt += "<image>\n"
                        types.append(MMUrlType.IMAGE)
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert (content_part.video_url != None)
                        urls.append(content_part.video_url.url)
                        now_prompt += "".join([f"Frame{i+1}: <image>\n" for i in range(video_frame_num)])
                        types.append(MMUrlType.VIDEO)

                prompt += f"{self.roles[message.role]}" + self.connector[0] + now_prompt
            if self.sep_style == SeparatorStyle.TWO:
                prompt += self.seps[index % 2]
            else:
                prompt += self.seps[0]
        prompt += self.roles[RoleEnum.assistant] + self.connector[1]
        return PromptWithMMInput(prompt, urls, types)

conv_internlm2 = InternVLConversation(
    system_content="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
    roles={RoleEnum.user: "<|im_start|>user\n", RoleEnum.assistant: "<|im_start|>assistant\n", RoleEnum.system: "<|im_start|>system\n"},
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|im_end|>"],
    connector=["", ""]
)

conv_phi3 = InternVLConversation(
    system_content="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
    roles={RoleEnum.user: "<|user|>\n", RoleEnum.assistant: "<|assistant|>\n", RoleEnum.system: "<|system|>\n"},
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|end|>"],
    connector=["", ""]
)

conv_templates = {
    "Hermes-2": conv_internlm2,
    "internlm2-chat": conv_internlm2,
    "phi3-chat": conv_phi3
}

class InternVLRenderer(CustomChatRenderer):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)
        self.roles = {
            RoleEnum.user: "USER",
            RoleEnum.assistant: "ASSISTANT"
        }
        self.video_frame_num = 8

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        ckpt_path: str = os.environ['CHECKPOINT_PATH']
        config_path = os.path.join(fetch_remote_file_to_local(ckpt_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
                template_name = config_json["template"]
                conv_template = conv_templates[template_name]
                prefix = "<s>" if template_name == "internlm2-chat" else ""
                res = conv_template.render_messages(messages, 8)
                res.prompt = prefix + res.prompt
                return res
        else:
            raise Exception("no config.json found")

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(input_ids=input_ids, input_urls=prompt_and_mm_input.urls, rendered_prompt=prompt_and_mm_input.prompt, input_urls_type=prompt_and_mm_input.mm_types)

register_renderer('internvl', InternVLRenderer)