import json
import os
import sys

from typing import Optional, List, Dict, Any, Union
from functools import lru_cache
from packaging import version
import logging

from transformers import PreTrainedTokenizer

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from maga_transformer.models.base_model import BaseTokenizer
from maga_transformer.openai.api_datatype import ChatCompletionRequest

# This class is designed to replace `PreTrainedTokenizer.apply_chat_template` method,
# providing more capability to customize the template.
# More specifically, this method allows template to use `functions` field, following openai chat api format.
# Besides that, other template elements is compatible with `PreTrainedTokenizer.apply_chat_template`.

DEFAULT_CHAT_API_TEMPLATE = (
"{% for message in messages %}"
"{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
"{% endfor %}"
"{% if add_generation_prompt %}"
"{{ '<|im_start|>assistant\n' }}"
"{% endif %}"
)

class TemplateRenderer():
    def __init__(self, tokenizer: Union[PreTrainedTokenizer, BaseTokenizer], tokenizer_kwargs: Dict[str, Any] = {}):
        if version.parse(jinja2.__version__) <= version.parse("3.0.0"):
            raise ImportError(
                "apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is " f"{jinja2.__version__}."
            )

        self.tokenizer = tokenizer

        self.chat_template = None
        self.special_tokens_map = {}
        try:
            if isinstance(tokenizer, PreTrainedTokenizer):
                self.chat_template = tokenizer.chat_template
                if self.chat_template == None:
                    self.chat_template = tokenizer.default_chat_template
                self.special_tokens_map = tokenizer.special_tokens_map
        except AttributeError as e:
            logging.warning(f"tokenizer {tokenizer} does not have chat_template or special_tokens_map: {e}")

        if self.chat_template == None:
            logging.info(f"no chat_template found in tokenizer, use default template")
            self.chat_template = DEFAULT_CHAT_API_TEMPLATE

        logging.info(f"render chat_template: {self.chat_template}")
        self.compiled_template = self._compile_jinja_template(self.chat_template)
        # TODO: this tag might should be False, to comply with function call.
        # Need to consider the case of responding function call.
        self.add_generation_prompt = True

    def render(self, request: ChatCompletionRequest) -> List[int]:
        rendered = self.compiled_template.render(
            messages=request.messages,
            functions=request.functions,
            json=json,
            add_generation_prompt=self.add_generation_prompt,
            **self.special_tokens_map
        )
        logging.info(f"openai request [{request.json(indent=4, ensure_ascii=False)}] rendered string: [{rendered}]]")
        return self.tokenizer.encode(rendered)

    @lru_cache
    def _compile_jinja_template(self, chat_template) -> jinja2.Template:

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)
