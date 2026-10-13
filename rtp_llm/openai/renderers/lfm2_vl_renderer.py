"""LFM2-VL multimodal-aware renderer.

The d1 chat template's ``parse_content`` maps ``{type: image}`` content
parts to a literal ``<image>`` token (id 124907); raw OpenAI-style
``image_url`` parts would fall through to a JSON dump. This renderer lifts
image URLs out of ``image_url`` parts, rewrites them into the template's
``{type: image}`` shape, renders the prompt, and returns the URLs as
``RenderedInputs.multimodal_inputs`` so the backend multimodal processor
expands each ``<image>`` placeholder into the ViT feature sequence.
"""

import json
import logging
from typing import List, Tuple

from typing_extensions import override

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import (
    BasicRenderer,
    PromptWithMMInput,
)
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Lfm2VlRenderer(BasicRenderer):
    """Multimodal renderer for LFM2-VL (lfm2_vl). Image-only."""

    def _collect_and_rewrite(
        self, messages: List[ChatMessage]
    ) -> Tuple[List[dict], PromptWithMMInput]:
        urls: List[str] = []
        types: List[MMUrlType] = []
        rewritten: List[dict] = []
        for msg in messages:
            if isinstance(msg.content, str) or msg.content is None:
                rewritten.append({"role": msg.role, "content": msg.content})
                continue

            new_parts = []
            for part in msg.content:
                if part.type == ContentPartTypeEnum.text:
                    new_parts.append({"type": "text", "text": part.text})
                elif part.type == ContentPartTypeEnum.image_url:
                    assert part.image_url is not None
                    urls.append(part.image_url.url)
                    types.append(MMUrlType.IMAGE)
                    new_parts.append({"type": "image"})
                elif part.type == ContentPartTypeEnum.video_url:
                    raise ValueError(
                        "LFM2-VL renderer does not support video_url parts: "
                        "the Lfm2VlVisionProcessor is image-only."
                    )
                else:
                    raise ValueError(
                        f"LFM2-VL renderer does not support content part type: {part.type}"
                    )
            rewritten.append({"role": msg.role, "content": new_parts})

        return rewritten, PromptWithMMInput(prompt="", urls=urls, mm_types=types)

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        has_mm = any(
            isinstance(m.content, list) and m.content for m in request.messages
        )
        if not has_mm:
            return super().render_chat(request)

        rewritten_messages, mm_input = self._collect_and_rewrite(list(request.messages))

        template = self._get_template(request)
        render_args = {
            "messages": rewritten_messages,
            "json": json,
            "add_generation_prompt": self.add_generation_prompt,
        }
        render_args.update(self.special_tokens_map)
        if request.chat_template_kwargs is not None:
            render_args.update(request.chat_template_kwargs)
        rendered = template.render(**render_args)
        logging.debug(f"lfm2_vl rendered string: [{rendered}]")
        input_ids = self.tokenizer.encode(rendered)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=mm_input.urls,
            input_urls_type=mm_input.mm_types,
            rendered_prompt=rendered,
        )


register_renderer("lfm2_vl", Lfm2VlRenderer)
