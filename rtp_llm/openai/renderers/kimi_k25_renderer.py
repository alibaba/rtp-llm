"""Kimi-K2.5 multimodal-aware renderer.

Inherits the KimiK2 reasoning / tool-call detector & stop-word handling
and adds an `image_url` content-part path for multimodal chat requests.

The chat template (`chat_template.jinja` in the HF ckpt) already emits one
``<|media_pad|>`` token per image. With ``mm_sep_tokens=[[163605]]`` the
RTP-LLM C++ splicer replaces each placeholder with the full ViT output
sequence, so this renderer does not need to pre-expand placeholders.

Video parts are not supported in this build: the in-tree fallback image
processor is image-only and the ckpt-side processor's video path depends
on ``mecord`` which is not packaged. Requests with ``video_url`` parts
are rejected up front rather than failing deep inside the ViT path.
"""

import copy
from typing import List

from typing_extensions import override

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.kimik2_renderer import KimiK2Renderer
from rtp_llm.utils.multimodal_util import MMUrlType


class KimiK25Renderer(KimiK2Renderer):
    """Multimodal renderer for Kimi-K2.5 (kimi_k25). Image-only."""

    def _collect_and_rewrite(
        self, messages: List[ChatMessage]
    ) -> tuple[List[ChatMessage], PromptWithMMInput]:
        """Lift media URLs out of content parts and rewrite each part into
        the dict shape consumed by the HF chat template (`{type: image,
        image: <url>}` etc.). Returns the rewritten message list and a
        PromptWithMMInput carrying URLs + per-URL types.

        Raises ``ValueError`` if any content part is a ``video_url`` —
        the current image processor cannot consume video frames.
        """

        urls: List[str] = []
        types: List[MMUrlType] = []
        rewritten: List[ChatMessage] = []
        for msg in messages:
            if isinstance(msg.content, str) or msg.content is None:
                rewritten.append(msg)
                continue

            new_parts = []
            for part in msg.content:
                if part.type == ContentPartTypeEnum.text:
                    new_parts.append({"type": "text", "text": part.text})
                elif part.type == ContentPartTypeEnum.image_url:
                    assert part.image_url is not None
                    urls.append(part.image_url.url)
                    types.append(MMUrlType.IMAGE)
                    new_parts.append({"type": "image", "image": part.image_url.url})
                elif part.type == ContentPartTypeEnum.video_url:
                    raise ValueError(
                        "Kimi-K2.5 renderer does not support video_url parts: "
                        "the in-tree image processor is image-only."
                    )
                else:
                    new_parts.append({"type": part.type.value})
            new_msg = msg.model_copy()
            new_msg.content = new_parts
            rewritten.append(new_msg)

        return rewritten, PromptWithMMInput(
            prompt="", urls=urls, mm_types=types
        )

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        has_mm = any(
            isinstance(m.content, list) and m.content for m in request.messages
        )
        if not has_mm:
            # Text-only path: defer to KimiK2Renderer's reasoning + tool flow.
            return super().render_chat(request)

        new_request = copy.copy(request)
        new_messages, mm_input = self._collect_and_rewrite(list(request.messages))
        new_request.messages = new_messages

        prompt = self._build_prompt(new_request)
        input_ids = self.tokenizer.encode(prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=mm_input.urls,
            input_urls_type=mm_input.mm_types,
            rendered_prompt=prompt,
        )


register_renderer("kimi_k25", KimiK25Renderer)
