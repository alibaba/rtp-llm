import json
import logging
from typing import List, Tuple

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ContentPartTypeEnum,
)
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs, RendererParams
from rtp_llm.openai.renderers.llava_renderer import get_preprocess_config
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.ops import MMPreprocessConfig
from rtp_llm.utils.base_model_datatypes import MMUrlType

# DiffusionGemma multimodal prompt contract: the checkpoint chat template
# renders one image_token per image; the engine's multimodal processor
# replaces whatever sits between <boi>/<eoi> with the ViT feature tokens.
# The renderer therefore rewrites each image_token placeholder into a
# (boi, eoi) pair before tokenization.
IMAGE_TOKEN = "<|image|>"


class DiffusionGemmaRenderer(BasicRenderer):
    def _collect_mm_inputs(
        self, request: ChatCompletionRequest
    ) -> Tuple[List[str], List[MMUrlType], List[MMPreprocessConfig]]:
        urls: List[str] = []
        types: List[MMUrlType] = []
        preprocess_configs: List[MMPreprocessConfig] = []
        for message in request.messages:
            if isinstance(message.content, list):
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url is not None
                        urls.append(content_part.image_url.url)
                        types.append(MMUrlType.IMAGE)
                        if content_part.preprocess_config:
                            preprocess_configs.append(
                                get_preprocess_config(content_part.preprocess_config)
                            )
        return urls, types, preprocess_configs

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        template = self._get_template(request)
        request_dict = json.loads(request.model_dump_json(exclude_none=True))
        render_args = {
            "messages": request_dict["messages"],
            "json": json,
            "add_generation_prompt": self.add_generation_prompt,
        }
        render_args.update(self.special_tokens_map)
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

        urls, types, preprocess_configs = self._collect_mm_inputs(request)
        image_placeholders = rendered.count(IMAGE_TOKEN)
        if image_placeholders != len(urls):
            logging.error(
                f"diffusion_gemma renderer: {len(urls)} images but "
                f"{image_placeholders} image tokens in rendered prompt"
            )
        rendered = rendered.replace(IMAGE_TOKEN, "<|image><image|>")
        input_ids = self.tokenizer.encode(rendered)
        return RenderedInputs(
            input_ids=input_ids,
            rendered_prompt=rendered,
            input_urls=urls,
            input_urls_type=types,
            preprocess_configs=preprocess_configs,
        )


register_renderer("diffusion_gemma", DiffusionGemmaRenderer)
