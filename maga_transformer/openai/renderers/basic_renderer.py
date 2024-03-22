from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator
import logging
import torch
from functools import lru_cache
from packaging import version
import json

from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, \
    RendererParams, StreamResponseObject, RenderedInputs, RendererInfo
from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, UsageInfo


DEFAULT_CHAT_API_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

@dataclass
class PromptWithImages:
    prompt: str
    image_urls: List[str]

# This class is designed to replace `PreTrainedTokenizerBase.apply_chat_template` functionality,
# providing more capability to customize the template.
# More specifically, this method allows template to use `functions` field, following openai chat api format.
# Besides that, other template elements is compatible with `PreTrainedTokenizerBase.apply_chat_template`.
class BasicRenderer(CustomChatRenderer):
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 renderer_params: RendererParams,
    ):
        super().__init__(tokenizer, renderer_params)

        if version.parse(jinja2.__version__) <= version.parse("3.0.0"):
            raise ImportError(
                "apply_chat_template requires jinja2>=3.0.0 to be installed. "
                "Your version is " f"{jinja2.__version__}."
            )

        self.add_generation_prompt = True
        self.chat_template = None
        self.special_tokens_map = {}

        try:
            self.chat_template = tokenizer.chat_template
            assert (self.chat_template != None)
        except:
            try:
                self.chat_template = tokenizer.default_chat_template
                assert (self.chat_template != None)
            except:
                logging.info(f"tokenizer {tokenizer} has no chat_template nor "
                              "default_chat_template attribute. Use default template.")
                self.chat_template = DEFAULT_CHAT_API_TEMPLATE
                self.add_extra_stop_words(["<|im_end|>"])

        try:
            if tokenizer.additional_special_tokens != None:
                self.add_extra_stop_words(tokenizer.additional_special_tokens)
        except:
            pass

        logging.info(f"use chat template: [ {self.chat_template} ]  ")
        self.compiled_template = self._compile_jinja_template(self.chat_template)

    def get_renderer_info(self) -> RendererInfo:
        renderer_info = super().get_renderer_info()
        renderer_info.template = self.chat_template
        return renderer_info

    @lru_cache
    def _compile_jinja_template(self, chat_template) -> jinja2.Template:

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        request_dict = json.loads(request.model_dump_json())
        rendered = self.compiled_template.render(
            messages=request_dict['messages'],
            functions=request_dict['functions'],
            json=json,
            add_generation_prompt=self.add_generation_prompt,
            **self.special_tokens_map
        )
        logging.debug(f"request [{request.model_dump_json(indent=4)}] rendered string: [{rendered}]]")
        return RenderedInputs(input_ids=self.tokenizer.encode(rendered))
