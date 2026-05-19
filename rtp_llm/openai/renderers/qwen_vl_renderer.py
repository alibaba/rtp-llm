import copy
from typing import Any, Dict, List

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs, RendererParams
from rtp_llm.openai.renderers.llava_renderer import get_preprocess_config
from rtp_llm.openai.renderers.qwen_renderer import QwenRenderer
from rtp_llm.ops import MMPreprocessConfig
from rtp_llm.utils.base_model_datatypes import MMUrlType


class QwenVLRenderer(QwenRenderer):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
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


class Qwen2VLRenderer(QwenRenderer):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
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

    def _format_tool_call_arguments(self, arguments: Any) -> Any:
        return arguments

    def _get_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        chat_template_kwargs: Dict[str, Any] = {
            "enable_thinking": bool(self.think_mode)
        }
        if request.chat_template_kwargs is not None:
            chat_template_kwargs.update(request.chat_template_kwargs)
        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
            and isinstance(request.extra_configs.chat_template_kwargs, dict)
        ):
            chat_template_kwargs.update(request.extra_configs.chat_template_kwargs)
        return chat_template_kwargs

    def _render_messages(
        self, request: ChatCompletionRequest, add_vision_id: bool
    ) -> PromptWithMMInput:
        urls = []
        types = []
        preprocess_configs = []
        final_messages = []
        for message in request.messages:
            msg_dict = {"role": message.role.value}

            if isinstance(message.content, list):
                now_content = []
                has_media = False
                has_non_empty_text = False
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        now_content.append({"type": "text", "text": content_part.text})
                        if content_part.text.strip():
                            has_non_empty_text = True
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        urls.append(content_part.image_url.url)
                        types.append(MMUrlType.IMAGE)
                        has_media = True
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
                        has_media = True
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
                        now_content.append(
                            {"type": "video", "video": content_part.video_url.url}
                        )
                if (
                    message.role == RoleEnum.user
                    and has_media
                    and not has_non_empty_text
                ):
                    now_content.append(
                        {"type": "text", "text": "Describe the visual content."}
                    )
                msg_dict["content"] = now_content
            else:
                msg_dict["content"] = message.content

            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": self._format_tool_call_arguments(
                                tc.function.arguments
                            ),
                        },
                    }
                    for tc in message.tool_calls
                ]
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            final_messages.append(msg_dict)

        final_tools = []
        if request.tools:
            for tool in request.tools:
                final_tools.append(
                    {
                        "type": tool.type,
                        "function": tool.function.model_dump(
                            exclude_none=True, mode="json"
                        ),
                    }
                )

        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "add_vision_id": add_vision_id,
            "tools": final_tools,
        }
        request_chat_template_kwargs = request.get_chat_template_kwargs()
        if request_chat_template_kwargs is not None:
            chat_template_kwargs.update(request_chat_template_kwargs)
        chat_template_kwargs.update(self._get_chat_template_kwargs(request))
        prompt = self.tokenizer.apply_chat_template(
            final_messages, **chat_template_kwargs
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
            request,
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
register_renderer("qwen3_vl", Qwen2VLRenderer)
register_renderer("qwen3_vl_moe", Qwen2VLRenderer)
