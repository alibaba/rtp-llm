"""
From https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import copy
import dataclasses
import logging
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.multimodal.multimodal_util import MMPreprocessConfig, MMUrlType
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
)


class SeparatorStyle(IntEnum):
    """Separator styles."""

    DeepSeek = auto()
    DeepSeekV2 = auto()
    PLAIN = auto()
    ALIGNMENT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The names of two roles
    roles: Dict[RoleEnum, str]
    # The separator configurations
    seps: List[str]
    connector: List[str]
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.DeepSeek

    def render_messages(
        self, messages: List[ChatMessage], tokenizer
    ) -> PromptWithMMInput:
        prompt: str = ""
        images: List[str] = []
        mm_types: List[MMUrlType] = []
        preprocess_configs: List[MMPreprocessConfig] = []

        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style != SeparatorStyle.DeepSeek:
            raise RuntimeError(
                f"Unsupported sep_style: {self.sep_style} for deepseek_vl_v2"
            )

        def get_preprocess_config(config):
            return MMPreprocessConfig(
                width=config.resized_width or -1,
                height=config.resized_height or -1,
                fps=config.fps or -1,
                min_frames=config.min_frames or -1,
                max_frames=config.max_frames or -1,
            )

        if messages[0].role != RoleEnum.system:
            if system_prompt is not None and system_prompt != "":
                prompt = system_prompt + self.seps[0]
        else:
            if messages[0].content is not None:
                assert isinstance(messages[0].content, str)
                system_prompt = messages[0].content.strip()
                if system_prompt != "":
                    prompt = system_prompt + self.seps[0]
            messages = messages[1:]

        # user message use sep[0], assistant message use sep[1]
        for index, message in enumerate(messages):
            if isinstance(message.content, str):
                prompt += (
                    f"{self.roles[message.role]}{self.connector[0]}{message.content}"
                )
            elif isinstance(message.content, list):
                now_prompt = ""
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        now_prompt += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        images.append(content_part.image_url.url)
                        mm_types.append(MMUrlType.IMAGE)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
                        now_prompt = "<image>\n" + now_prompt
                    else:
                        raise Exception(
                            f"Unsupported content part type: {content_part.type} for deepseek_vl_v2"
                        )
                prompt += f"{self.roles[message.role]}: {now_prompt}"
            prompt += self.seps[index % 2]
        prompt += f"{self.roles[RoleEnum.assistant]}{self.connector[1]}"
        logging.debug(f"deepseek_vl2 prompt: {prompt}")
        return PromptWithMMInput(prompt, images, mm_types)

    def copy(self):
        return Conversation(
            name=self.name,
            roles=self.roles,
            seps=self.seps,
            system_template=self.system_template,
            system_message=self.system_message,
            sep_style=self.sep_style,
            connector=self.connector,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


register_conv_template(
    Conversation(
        name="deepseek",
        system_template="{system_message}",
        # system_message="You are a helpful assistant. Please answer truthfully and write out your "
        # "thinking step by step to be sure you get the right answer.",
        system_message="",
        roles={RoleEnum.user: "<|User|>", RoleEnum.assistant: "<|Assistant|>"},
        sep_style=SeparatorStyle.DeepSeek,
        seps=["\n\n", "<｜end▁of▁sentence｜>"],
        connector=[": ", ":"],
    )
)

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig


class DeepSeekVLV2Renderer(CustomChatRenderer):
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
        self._setup_stop_words()

    def _setup_stop_words(self):
        # 直接使用 token ID 而不是字符串，避免 encode 时添加特殊 token
        user_token_ids = self.tokenizer.tokenizer.encode(
            "User:", add_special_tokens=False
        )
        eos_token_ids = self.tokenizer.tokenizer.encode(
            "<｜end▁of▁sentence｜>", add_special_tokens=False
        )
        self.add_extra_stop_word_ids([user_token_ids, eos_token_ids])

    def _get_conv_template(self, model_name: str) -> Conversation:
        conv_mode = "deepseek"
        return conv_templates[conv_mode]

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        # Use checkpoint path from model_config
        ckpt_path: str = self.model_config.checkpoint_path
        model_name: str = ckpt_path.split("?")[0]  # oss style path
        model_name = model_name.strip("/").split("/")[-1]
        conv_template = self._get_conv_template(model_name)

        return conv_template.render_messages(messages, self.tokenizer)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)

        # must start with bos token
        input_ids = [self.tokenizer.bos_token_id]
        input_ids.extend(
            self.tokenizer.encode(prompt_and_mm_input.prompt, add_special_tokens=False)
        )
        for i in range(len(input_ids)):
            if input_ids[i] < 0:
                input_ids[i] = self.tokenizer.pad_token_id

        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("deepseek_vl_v2", DeepSeekVLV2Renderer)
