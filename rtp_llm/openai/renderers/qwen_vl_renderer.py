import copy
from typing import List

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
from rtp_llm.utils.base_model_datatypes import MMPreprocessConfig, MMUrlType


class QwenVLRenderer(CustomChatRenderer):
    def __init__(self, tokenizer: BaseTokenizer, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        prompt = ""
        images = []
        if messages[0].role != RoleEnum.system:
            messages = [
                ChatMessage(
                    role=RoleEnum.system, content="You are a helpful assistant."
                )
            ] + messages

        for message in messages:
            if isinstance(message.content, str):
                prompt += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
            elif isinstance(message.content, list):
                prompt += f"<|im_start|>{message.role}\n"
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        prompt += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        url = content_part.image_url.url
                        images.append(url)
                        prompt += f"Picture {len(images)}: <img>{url}</img>\n"
                prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return PromptWithMMInput(prompt=prompt, urls=images)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
        )


class Qwen2VLRenderer(CustomChatRenderer):
    def __init__(self, tokenizer: BaseTokenizer, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)

    def _render_messages(
        self, messages: List[ChatMessage], add_vision_id: bool
    ) -> PromptWithMMInput:
        urls = []
        types = []
        preprocess_configs = []
        final_messages = []

        def get_preprocess_config(config):
            return MMPreprocessConfig(
                width=config.resized_width or -1,
                height=config.resized_height or -1,
                min_pixels=config.min_pixels or -1,
                max_pixels=config.max_pixels or -1,
                fps=config.fps or -1,
                min_frames=config.min_frames or -1,
                max_frames=config.max_frames or -1,
                mm_timeout_ms=config.mm_timeout_ms,
            )

        for message in messages:
            if isinstance(message.content, str):
                final_messages.append(
                    {"role": message.role.value, "content": message.content}
                )
            elif isinstance(message.content, list):
                now_message = {"role": message.role.value}
                now_content = []
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        now_content.append({"type": "text", "text": content_part.text})
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        urls.append(content_part.image_url.url)
                        types.append(MMUrlType.IMAGE)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
                        now_content.append(
                            {"type": "image", "image": content_part.image_url.url}
                        )
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert content_part.video_url != None
                        urls.append(content_part.video_url.url)
                        types.append(MMUrlType.VIDEO)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
                        now_content.append(
                            {"type": "video", "image": content_part.video_url.url}
                        )
                now_message["content"] = now_content
                final_messages.append(now_message)

        prompt = self.tokenizer.apply_chat_template(
            final_messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=add_vision_id,
        )

        return PromptWithMMInput(
            prompt=prompt,
            urls=urls,
            mm_types=types,
            preprocess_configs=preprocess_configs,
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(
            messages,
            request.extra_configs.add_vision_id if request.extra_configs else True,
        )
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("qwen_vl", QwenVLRenderer)
register_renderer("qwen_vl_1b8", QwenVLRenderer)
register_renderer("qwen2_vl", Qwen2VLRenderer)
register_renderer("qwen2_5_vl", Qwen2VLRenderer)
register_renderer("qwen3_vl_moe", Qwen2VLRenderer)
