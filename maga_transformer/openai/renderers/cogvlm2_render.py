import copy
import json
import re
import logging
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator

from transformers import PreTrainedTokenizerBase

from maga_transformer.config.gpt_init_model_parameters import TemplateType
from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams, \
    StreamResponseObject, RenderedInputs
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer, PromptWithMMInput
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, UsageInfo, \
    ContentPart, ContentPartTypeEnum, FunctionCall
from maga_transformer.openai.renderer_factory_register import register_renderer

class CogVLM2Renderer(CustomChatRenderer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)
        self.template_type = renderer_params.template_type

    def query_answer_template(self, template_type: TemplateType) -> str:
        if template_type == TemplateType.base:
            return "{}", "{}"
        elif template_type == TemplateType.vqa:
            return "Question: {}", " Short answer: {}\n"
        elif template_type == TemplateType.chat:
            return "Question: {}", " Answer: {}\n"
        else:
            raise Exception("Unknown template type")

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        prompt = ""
        images = []
        template_type = self.template_type

        query_format, answer_format = self.query_answer_template(template_type)
        
        last_message = ""
        last_format_message = ""
        # handle history message and save the latest query in last_message
        # For messages with multiple content parts(containing history message), we assume last message is the lastest query. 
        # cogvlm2 chat template distinguishes between query and answer, and we assume the query's role is user and answer's role is assistant
        # see https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B/blob/main/modeling_cogvlm.py#580
        for message in messages:
            prompt += last_format_message
            if isinstance(message.content, str):
                last_message = message.content
                if message.role == RoleEnum.user:
                    last_format_message = query_format.format(last_message)
                elif message.role == RoleEnum.assistant:
                    last_format_message = answer_format.format(last_message)
                else:
                    raise Exception(f"Unknown role: {message.role}")
            elif isinstance(message.content, list):
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert (isinstance(content_part.text, str))
                        last_message = content_part.text
                        if message.role == RoleEnum.user:
                            last_format_message = query_format.format(last_message)
                        elif message.role == RoleEnum.assistant:
                            last_format_message = answer_format.format(last_message)
                        else:
                            raise Exception(f"Unknown role: {message.role}")
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        images.append(content_part.image_url.url)

        # handle latest query
        if template_type == 'base':
            prompt += last_message
        else:
            # remove tail answer_format for template_type 'vqa' and 'chat'
            prompt += 'Question: {}{}'.format(last_message, answer_format[:-4])

        return PromptWithMMInput(prompt=prompt, urls=images)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(input_ids=input_ids, input_urls=prompt_and_mm_input.urls, rendered_prompt=prompt_and_mm_input.prompt)

register_renderer('cogvlm2', CogVLM2Renderer)