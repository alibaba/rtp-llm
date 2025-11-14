import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
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
from rtp_llm.utils.multimodal_util import MMPreprocessConfig, MMUrlType


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
    image_sep: str = "<image>\n"

    def render_messages(
        self, messages: List[ChatMessage], tokenizer
    ) -> PromptWithMMInput:
        prompt: str = ""
        images: List[str] = []
        mm_types: List[MMUrlType] = []
        preprocess_configs: List[MMPreprocessConfig] = []

        if self.sep_style == SeparatorStyle.LLAMA_3:
            chat_template_messages = [
                {"role": "system", "content": self.system_content}
            ]
            for message in messages:
                role = self.roles[message.role]
                content = message.content
                if isinstance(content, str):
                    chat_template_messages.append({"role": role, "content": content})
                else:
                    now_prompt = ""
                    for content_part in content:
                        if content_part.type == ContentPartTypeEnum.text:
                            assert isinstance(content_part.text, str)
                            now_prompt += content_part.text
                        elif content_part.type == ContentPartTypeEnum.image_url:
                            assert content_part.image_url != None
                            images.append(content_part.image_url.url)
                            mm_types.append(MMUrlType.IMAGE)
                            now_prompt = f"<image>\n" + now_prompt
                    chat_template_messages.append({"role": role, "content": now_prompt})

            return PromptWithMMInput(
                tokenizer.apply_chat_template(
                    chat_template_messages, tokenize=False, add_generation_prompt=True
                ),
                images,
                mm_types,
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
            prompt = self.system_content + prompt + self.seps[0]

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
                        now_prompt = now_prompt + self.image_sep
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert content_part.video_url != None
                        images.append(content_part.video_url.url)
                        mm_types.append(MMUrlType.VIDEO)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
                        now_prompt = now_prompt + self.image_sep
                prompt += f"{self.roles[message.role]}" + self.connector[0] + now_prompt
            if self.sep_style == SeparatorStyle.TWO:
                prompt += self.seps[index % 2]
            else:
                prompt += self.seps[0]
        prompt += self.roles[RoleEnum.assistant] + self.connector[1]
        return PromptWithMMInput(prompt, images, mm_types)


conv_llava_v0 = Conversation(
    system_content="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles={
        RoleEnum.user: "Human",
        RoleEnum.assistant: "Assistant",
        RoleEnum.system: "System",
    },
    sep_style=SeparatorStyle.SINGLE,
    seps=["###"],
    connector=[": ", ":"],
)

conv_llava_v1 = Conversation(
    system_content="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles={
        RoleEnum.user: "USER",
        RoleEnum.assistant: "ASSISTANT",
        RoleEnum.system: "SYSTEM",
    },
    sep_style=SeparatorStyle.TWO,
    seps=[" ", "</s>"],
    connector=[": ", ":"],
)

conv_llava_llama = Conversation(
    system_content="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
    roles={
        RoleEnum.user: "<|start_header_id|>user",
        RoleEnum.assistant: "<|start_header_id|>assistant",
        RoleEnum.system: "<|start_header_id|>system",
    },
    sep_style=SeparatorStyle.LLAMA_3,
    seps=[""],
    connector=[": ", ":"],
)

conv_qwen = Conversation(
    system_content="<|im_start|>system\nYou are a helpful assistant.",
    roles={
        RoleEnum.user: "<|im_start|>user",
        RoleEnum.assistant: "<|im_start|>assistant",
        RoleEnum.system: "<|im_start|>system",
    },
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|im_end|>\n"],
    connector=["\n", "\n"],
)

conv_templates = {
    "llava_v0": conv_llava_v0,
    "llava_v1": conv_llava_v1,
    "llava_llama3": conv_llava_llama,
    "llava_qwen": conv_qwen,
}


from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig

class LlavaRenderer(CustomChatRenderer):
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
        super().__init__(tokenizer, renderer_params, generate_env_config, render_config, ckpt_path, misc_config, vit_config)

    def _get_conv_template(self, model_name: str) -> Conversation:
        if "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "llama3" in model_name.lower():
            conv_mode = "llava_llama3"
        elif "next" in model_name.lower() or "onevision" in model_name.lower():
            conv_mode = "llava_qwen"
        else:
            conv_mode = "llava_v0"
        return conv_templates[conv_mode]

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        # Use checkpoint path from model_config
        ckpt_path: str = self.model_config.checkpoint_path
        model_name: str = ckpt_path.split("?")[0]  # oss style path
        model_name = model_name.strip("/").split("/")[-1]
        llava_template_env: str = self.render_config.llava_chat_template
        conv_template = (
            self._get_conv_template(model_name)
            if llava_template_env == ""
            else self._get_conv_template(llava_template_env)
        )

        return conv_template.render_messages(messages, self.tokenizer)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("llava", LlavaRenderer)
