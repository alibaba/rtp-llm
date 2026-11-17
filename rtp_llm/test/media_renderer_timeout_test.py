"""Renderer extraction through real Native inputs and timeout resolution.

The tokenizer only supplies prompt text; media config/input bindings are real.
No downloads or model execution are needed.
"""

from types import SimpleNamespace
from unittest import TestCase

from rtp_llm.multimodal.mm_process_engine import MMWorkItem
from rtp_llm.openai.api_datatype import ChatCompletionRequest, RoleEnum
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.deepseek_vl2_renderer import (
    Conversation as DeepSeekConversation,
)
from rtp_llm.openai.renderers.kimi_k25_renderer import KimiK25Renderer
from rtp_llm.openai.renderers.llava_renderer import Conversation, SeparatorStyle
from rtp_llm.openai.renderers.qwen_vl_renderer import Qwen2VLRenderer
from rtp_llm.ops import MMPreprocessConfig


class MediaRendererTimeoutTest(TestCase):
    def render_media(self, route, parts):
        request = ChatCompletionRequest(
            model="test", messages=[{"role": "user", "content": parts}]
        )
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")
        roles = {RoleEnum.user: "user", RoleEnum.assistant: "assistant"}
        if route == "deepseek":
            return DeepSeekConversation(
                name="test", roles=roles, seps=["\n", "\n"], connector=[":", ":"]
            ).render_messages(request.messages, tokenizer)
        if route.startswith("llava"):
            style = SeparatorStyle.LLAMA_3 if route == "llava3" else SeparatorStyle.TWO
            return Conversation(
                system_content="",
                roles=roles,
                sep_style=style,
                seps=["\n", "\n"],
                connector=[":", ":"],
            ).render_messages(request.messages, tokenizer)
        if route == "kimi":
            renderer = object.__new__(KimiK25Renderer)
            return renderer._collect_and_rewrite(request.messages)[1]
        renderer = object.__new__(Qwen2VLRenderer)
        renderer.tokenizer = tokenizer
        return renderer._render_messages(request, False)

    def test_default_explicit_and_mixed_media_inherit_custom_server_budget(self):
        for route, kind in (
            ("deepseek", "image_url"),
            ("kimi", "image_url"),
            ("llava", "image_url"),
            ("llava3", "image_url"),
            ("llava", "video_url"),
            ("qwen", "image_url"),
            ("qwen", "video_url"),
        ):
            for timeouts in ((None,), (4567,), (None, 4567, None)):
                with self.subTest(route=route, kind=kind, timeouts=timeouts):
                    parts = []
                    for index, timeout in enumerate(timeouts):
                        part = {"type": kind, kind: {"url": f"https://example/{index}"}}
                        if timeout is not None:
                            part["preprocess_config"] = {
                                "mm_timeout_ms": timeout,
                                "resized_width": 42,
                            }
                        parts.append(part)
                    media = self.render_media(route, parts)
                    self.assertEqual(len(media.urls), len(timeouts))
                    self.assertEqual(len(media.preprocess_configs), len(timeouts))
                    rendered = RenderedInputs(
                        input_ids=[1],
                        input_urls=media.urls,
                        input_urls_type=media.mm_types,
                        preprocess_configs=media.preprocess_configs,
                    )
                    for server_budget in (123000, 234567):
                        for item, timeout in zip(rendered.multimodal_inputs, timeouts):
                            self.assertIsInstance(
                                item.mm_preprocess_config, MMPreprocessConfig
                            )
                            self.assertEqual(
                                item.mm_preprocess_config.mm_timeout_ms,
                                -1 if timeout is None else timeout,
                            )
                            if timeout is not None:
                                self.assertEqual(item.mm_preprocess_config.width, 42)
                            self.assertEqual(
                                MMWorkItem(
                                    [item], mm_timeout_ms=server_budget
                                ).mm_timeout_ms,
                                server_budget if timeout is None else timeout,
                            )
                        self.assertEqual(
                            MMWorkItem(
                                rendered.multimodal_inputs, mm_timeout_ms=server_budget
                            ).mm_timeout_ms,
                            max(server_budget if t is None else t for t in timeouts),
                        )

    def test_kimi_video_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support video_url"):
            self.render_media(
                "kimi", [{"type": "video_url", "video_url": {"url": "v"}}]
            )
