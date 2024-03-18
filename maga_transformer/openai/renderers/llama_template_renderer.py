import copy
import json
import re
import logging
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator

from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass

from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, \
    ChatCompletionRequest, RoleEnum, FunctionCall, RendererInfo
from maga_transformer.openai.renderers.llama_template import Template, get_template_and_fix_tokenizer
from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams, \
    StreamResponseObject, RenderedInputs
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, UsageInfo, \
    ContentPart, ContentPartTypeEnum

@dataclass
class LlamaTemplateArgs:
    query: str
    resp: str = ""
    history: Optional[List[Tuple[str, str]]] = None
    system: Optional[str] = None

class LlamaTemplateRenderer(CustomChatRenderer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)
        model_name = renderer_params.model_type
        self.template = get_template_and_fix_tokenizer(model_name, tokenizer)
        self.add_extra_stop_words(self.template.stop_words)

    def get_renderer_info(self) -> RendererInfo:
        renderer_info = super().get_renderer_info()
        renderer_info.template = str(self.template)
        return renderer_info

    def _extract_history(self, messages: List[ChatMessage]) -> LlamaTemplateArgs:
        # Messages must be formatted in the following way:
        # 1. Messages may start with a system message or not.
        #    If started with a system message, it must be the first and only system message.
        #    If not started with a system message, it must not contain any system message.
        # 2. Messages must be in the order of [user, assistant, user, assistant, ...]
        # 3. The last message must be from the user.

        history = []
        system = None
        query = ""
        if messages[0].role == RoleEnum.system:
            system = messages[0].content
            assert isinstance(system, str)
            messages = messages[1:]

        query_message = messages.pop()
        query = query_message.content
        assert isinstance(query, str)

        assert (len(messages) % 2 == 0)
        for idx in range(0, len(messages), 2):
            user_message = messages[idx]
            assistant_message = messages[idx + 1]
            assert (user_message.role == RoleEnum.user)
            assert (assistant_message.role == RoleEnum.assistant)
            history.append((user_message.content, assistant_message.content))

        return LlamaTemplateArgs(query=query, history=history, system=system)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        template_args = self._extract_history(request.messages)
        assert isinstance(self.tokenizer, PreTrainedTokenizerBase)
        encoded_ids = self.template.encode_oneturn(
            self.tokenizer,
            query=template_args.query,
            resp=template_args.resp,
            history=template_args.history,
            system=template_args.system,
        )[0]
        return RenderedInputs(input_ids=encoded_ids)

