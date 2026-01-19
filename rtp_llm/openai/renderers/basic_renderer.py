import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from packaging import version

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererInfo,
    RendererParams,
)
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.utils.multimodal_util import MMPreprocessConfig, MMUrlType

DEFAULT_CHAT_API_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


@dataclass
class PromptWithMMInput:
    prompt: str
    urls: List[str]
    mm_types: List[MMUrlType] = field(default_factory=list)
    preprocess_configs: List[MMPreprocessConfig] = field(default_factory=list)


# This class is designed to replace `PreTrainedTokenizerBase.apply_chat_template` functionality,
# providing more capability to customize the template.
# More specifically, this method allows template to use `functions` field, following openai chat api format.
# Besides that, other template elements is compatible with `PreTrainedTokenizerBase.apply_chat_template`.
class BasicRenderer(CustomChatRenderer):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config: GenerateEnvConfig,
        render_config: Optional[RenderConfig] = None,
        ckpt_path: Optional[str] = None,
        misc_config: Optional[Any] = None,
        vit_config: Optional[Any] = None,
    ):
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )

        if version.parse(jinja2.__version__) <= version.parse("3.0.0"):
            raise ImportError(
                "apply_chat_template requires jinja2>=3.0.0 to be installed. "
                "Your version is "
                f"{jinja2.__version__}."
            )

        self.add_generation_prompt = True
        self.chat_template = None
        self.special_tokens_map = {}
        try:
            self._setup_chat_template()
            assert self.chat_template != None
        except:
            try:
                self.chat_template = tokenizer.default_chat_template
                assert self.chat_template != None
            except:
                logging.info(
                    f"tokenizer {tokenizer} has no chat_template nor "
                    "default_chat_template attribute. Use default template."
                )
                self.chat_template = DEFAULT_CHAT_API_TEMPLATE
                self.add_extra_stop_words(["<|im_end|>"])

        try:
            if tokenizer.special_tokens_map != None:
                self.special_tokens_map = tokenizer.special_tokens_map
                for k, v in self.special_tokens_map.items():
                    logging.info(f"special token [{v}]({k}) added as stop words.")
                    if isinstance(v, str):
                        self.add_extra_stop_words([v])
                    elif isinstance(v, list):
                        self.add_extra_stop_words(v)
        except:
            pass

        try:
            if tokenizer.additional_special_tokens != None:
                logging.info(
                    f"additional special tokens {tokenizer.additional_special_tokens}"
                    "added as stop words."
                )
                self.add_extra_stop_words(tokenizer.additional_special_tokens)
        except:
            pass

        # try:
        #     if tokenizer.added_tokens_decoder != None:
        #         for token_id, added_token in tokenizer.added_tokens_decoder.items():
        #             logging.info(f"added token [{token_id}]({added_token}) added as stop words.")
        #             self.add_extra_stop_word_ids([[token_id]])
        # except:
        #     pass

        logging.info(f"found chat template to use: {self.chat_template}")
        self.default_template_key = self.render_config.default_chat_template_key
        self.default_tool_use_template_key = (
            self.render_config.default_tool_use_template_key
        )
        self.compiled_template_map: Dict[str, jinja2.Template] = {}

        if isinstance(self.chat_template, dict):
            for key, template in self.chat_template.items():
                self.compiled_template_map[key] = self._compile_jinja_template(template)
        elif isinstance(self.chat_template, str):
            self.compiled_template_map[self.default_template_key] = (
                self._compile_jinja_template(self.chat_template)
            )
        else:
            raise Exception(
                f"chat template [{self.chat_template}] "
                f"of type [{type(self.chat_template)}] is not supported."
            )

        if self.default_template_key not in self.compiled_template_map:
            raise Exception(
                f"default template key [{self.default_template_key}] not found "
                f"in chat templates: [{self.compiled_template_map.keys()}]"
            )
        if self.default_tool_use_template_key not in self.compiled_template_map:
            self.default_tool_use_template_key = self.default_template_key
        logging.info(
            f"compiled chat templates to use: {self.compiled_template_map.keys()}"
        )

    def get_renderer_info(self) -> RendererInfo:
        renderer_info = super().get_renderer_info()
        renderer_info.template = self.chat_template
        return renderer_info

    @lru_cache
    def _compile_jinja_template(self, chat_template) -> jinja2.Template:

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        )
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    def _get_template(self, request: ChatCompletionRequest) -> jinja2.Template:
        if request.user_template:
            if request.template_key:
                raise ValueError(
                    "template_key and user_template can not be used together."
                )
            return self._compile_jinja_template(request.user_template)
        template_key = (
            self.default_tool_use_template_key
            if request.functions
            else self.default_template_key
        )
        template_key = request.template_key or template_key
        return self.compiled_template_map[template_key]

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        template = self._get_template(request)
        request_dict = json.loads(request.model_dump_json(exclude_none=True))
        render_args = {
            "messages": request_dict["messages"],
            "json": json,
            "add_generation_prompt": self.add_generation_prompt,
        }
        render_args.update(self.special_tokens_map)
        # functions with none value may occur exception in llama3 template
        if request_dict.get("functions", None):
            render_args["functions"] = request_dict["functions"]
        if request.chat_template_kwargs is not None:
            render_args.update(request.chat_template_kwargs)
        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
            and isinstance(request.extra_configs.chat_template_kwargs, dict)
        ):
            render_args.update(request.extra_configs.chat_template_kwargs)
        rendered = template.render(**render_args)
        logging.debug(
            f"request [{request.model_dump_json(indent=4)}] rendered string: [{rendered}]]"
        )
        return RenderedInputs(
            input_ids=self.tokenizer.encode(rendered), rendered_prompt=rendered
        )
