import copy
import json
import logging
from typing import Any, Dict, List, Optional

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    MMPreprocessConfigPart,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs, RendererParams
from rtp_llm.openai.renderers.llava_renderer import get_preprocess_config
from rtp_llm.openai.renderers.minimax_m3_renderer import MiniMaxM3Renderer
from rtp_llm.utils.base_model_datatypes import MMUrlType


class MiniMaxM3VLRenderer(MiniMaxM3Renderer):
    """OpenAI chat renderer for MiniMax-M3 VL.

    The MiniMax-M3 Jinja `chat_template` (shipped with the HF tokenizer)
    already understands multimodal content parts of the form
    ``{"type": "image", "image": <url>}`` / ``{"type": "video", "video": <url>}``
    and expands them to the model's ``]<]image[>[`` / ``]<]video[>[`` markers
    via the ``visible_text`` macro. The renderer's job is therefore quite
    thin: rewrite ``ChatMessage.content`` into the dict shape the template
    expects, accumulate the URLs (and their types / preprocess configs) in
    appearance order, and let ``tokenizer.apply_chat_template`` do the rest.
    """

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

    def _render_messages(self, request: ChatCompletionRequest) -> PromptWithMMInput:
        urls: List[str] = []
        types: List[MMUrlType] = []
        preprocess_configs = []
        final_messages: List[Dict[str, Any]] = []

        for message in request.messages:
            msg_dict: Dict[str, Any] = {"role": message.role.value}

            if isinstance(message.content, list):
                rebuilt_content: List[Dict[str, Any]] = []
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        rebuilt_content.append(
                            {"type": "text", "text": content_part.text}
                        )
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url is not None
                        url = content_part.image_url.url
                        urls.append(url)
                        types.append(MMUrlType.IMAGE)
                        preprocess_configs.append(
                            self._get_preprocess_config(
                                content_part, content_part.image_url
                            )
                        )
                        # The template visible_text() macro detects type=="image"
                        # and emits the ]<]image[>[ marker, replacing the URL.
                        rebuilt_content.append({"type": "image", "image": url})
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert content_part.video_url is not None
                        url = content_part.video_url.url
                        urls.append(url)
                        types.append(MMUrlType.VIDEO)
                        preprocess_configs.append(
                            self._get_preprocess_config(
                                content_part, content_part.video_url
                            )
                        )
                        rebuilt_content.append({"type": "video", "video": url})
                msg_dict["content"] = rebuilt_content
            else:
                msg_dict["content"] = message.content

            if message.reasoning_content is not None:
                msg_dict["reasoning_content"] = message.reasoning_content

            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            final_messages.append(msg_dict)

        final_messages = self._preprocess_messages(final_messages)

        final_tools: List[Dict[str, Any]] = []
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

        template_kwargs = request.get_resolved_chat_template_kwargs()

        apply_kwargs: Dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        if final_tools:
            apply_kwargs["tools"] = final_tools
        apply_kwargs.update(template_kwargs)

        prompt = self.tokenizer.apply_chat_template(final_messages, **apply_kwargs)

        return PromptWithMMInput(
            prompt=prompt,
            urls=urls,
            mm_types=types,
            preprocess_configs=preprocess_configs,
        )

    @staticmethod
    def _get_preprocess_config(content_part, media_url):
        config = (
            content_part.preprocess_config.model_copy()
            if content_part.preprocess_config is not None
            else MMPreprocessConfigPart()
        )
        for field in ("max_long_side_pixel", "fps"):
            media_value = getattr(media_url, field, None)
            part_value = getattr(content_part, field, None)
            if media_value is not None:
                setattr(config, field, media_value)
            if part_value is not None:
                setattr(config, field, part_value)
        return get_preprocess_config(config)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        request = copy.deepcopy(request)
        prompt_and_mm_input = self._render_messages(request)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        logging.debug(
            f"minimax_m3_vl rendered prompt: {prompt_and_mm_input.prompt}, "
            f"urls: {prompt_and_mm_input.urls}, "
            f"types: {prompt_and_mm_input.mm_types}"
        )
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("minimax_m3_vl", MiniMaxM3VLRenderer)
