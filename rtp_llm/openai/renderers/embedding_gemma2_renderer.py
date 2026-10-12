"""Checkpoint embedding template with the existing image URL transport."""

import json

from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.utils.base_model_datatypes import MMUrlType


class EmbeddingGemma2ChatRenderer(BasicRenderer):
    def render_chat(self, request):
        messages = json.loads(request.model_dump_json(exclude_none=True))["messages"]
        urls = []
        types = []
        for message in messages:
            if not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                if part["type"] == "image_url":
                    urls.append(part["image_url"]["url"])
                    part["type"] = "image"
                    types.append(MMUrlType.IMAGE)
                elif part["type"] in ("audio_url", "video_url"):
                    kind = part["type"].removesuffix("_url")
                    urls.append(part[part["type"]]["url"])
                    types.append(
                        MMUrlType.AUDIO if kind == "audio" else MMUrlType.VIDEO
                    )
                    part["type"] = kind
        prompt = self._get_template(request).render(
            messages=messages, add_generation_prompt=False, **self.special_tokens_map
        )
        if sum(
            prompt.count(f"<|{kind}|>") for kind in ("image", "video", "audio")
        ) != len(urls):
            raise ValueError(
                "EmbeddingGemma2 image placeholders must match image inputs"
            )
        # Existing embedding RPC expands the span between these boundaries to
        # the real feature length. The tokenizer supplies BOS/EOS exactly once.
        token_prompt = prompt.replace("<|image|>", "<|image><|image|><image|>")
        token_prompt = token_prompt.replace("<|video|>", "<|image><|video|><image|>")
        token_prompt = token_prompt.replace("<|audio|>", "<|audio><|audio|><audio|>")
        return RenderedInputs(
            input_ids=self.tokenizer.encode(token_prompt),
            rendered_prompt=prompt,
            input_urls=urls,
            input_urls_type=types,
        )


register_renderer("embedding_gemma2", EmbeddingGemma2ChatRenderer)
