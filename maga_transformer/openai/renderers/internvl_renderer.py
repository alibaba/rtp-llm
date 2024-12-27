import copy
import functools
import json
import re
import logging
import os
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator
from enum import Enum, auto
from transformers import PreTrainedTokenizerBase

from dataclasses import dataclass

from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, \
    ChatCompletionRequest, RoleEnum, FunctionCall
from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, OutputDelta, RendererParams, \
    StreamResponseObject, RenderedInputs, StreamStatus
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

from maga_transformer.utils.word_util import is_truncated

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

conv_internvl2_5 = InternVLConversation(
    system_content="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
    roles={RoleEnum.user: "<|im_start|>user\n", RoleEnum.assistant: "<|im_start|>assistant\n", RoleEnum.system: "<|im_start|>system\n"},
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|im_end|>\n"],
    connector=["", ""]
)

conv_templates = {
    "Hermes-2": conv_internlm2,
    "internlm2-chat": conv_internlm2,
    "phi3-chat": conv_phi3,
    "internvl2_5": conv_internvl2_5
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
    
    async def _update_single_status(self, status: StreamStatus, output: GenerateOutput, max_new_tokens: int, stop_words_str: List[str], stop_word_slice_list: List[str], is_streaming: bool) -> OutputDelta:
        if status.finish_reason != None:
            return await self._create_empty_delta(status.output.aux_info)
        status.update_output(output, self._clean_output_ids, functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens), self._remove_stop_word_ids)
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if is_streaming:
            if len(decoded_string) > 0 and u'\uFFFD' == decoded_string[-1]:
                return await self._create_empty_delta(output.aux_info)
        else:
            while (len(decoded_string) > 0) and (u'\uFFFD' == decoded_string[-1]):
                decoded_string = decoded_string[:-1]
        if len(decoded_prev_token) == 0 and len(decoded_string) > 0 and decoded_string[0] == ' ':
            status.delta_output_string = decoded_string[1:]
        elif len(decoded_prev_token) > 0 and len(decoded_string) > 0 and decoded_string[0] == ' ' and decoded_prev_token[0] != ' ':
            status.delta_output_string = decoded_string[len(decoded_prev_token) + 1: ]
        else: 
            status.delta_output_string = decoded_string[len(decoded_prev_token):]
        if is_truncated(status.delta_output_string, stop_words_str, is_streaming):
            status.finish_reason = FinisheReason.stop
            return await self._create_empty_delta(output.aux_info)
        if not is_truncated(status.delta_output_string, stop_word_slice_list, is_streaming):
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=await self._generate_log_probs(status, output),
                input_length=output.aux_info.input_len,
                output_length=output.aux_info.output_len,
                reuse_length=output.aux_info.reuse_len)
            status.delta_output_string = ""
            return delta
        else:
            return await self._create_empty_delta(output.aux_info)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(input_ids=input_ids, input_urls=prompt_and_mm_input.urls, rendered_prompt=prompt_and_mm_input.prompt, input_urls_type=prompt_and_mm_input.mm_types)

register_renderer('internvl', InternVLRenderer)