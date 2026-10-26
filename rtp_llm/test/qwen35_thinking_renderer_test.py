"""Prompt/parser thinking-mode consistency, using the real checkpoint template."""

import os
import unittest

from transformers import AutoTokenizer

from rtp_llm.config.generate_config import ThinkingMode
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.qwen35_renderer import Qwen35Renderer
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Qwen35ThinkingRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ["RTP_MM_TEST_CHECKPOINT"], local_files_only=True
        )

    def render(self, default=ThinkingMode.DISABLED, **kwargs):
        renderer = Qwen35Renderer.__new__(Qwen35Renderer)
        renderer.tokenizer = self.tokenizer
        renderer.default_thinking_mode = default
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "请只回答：测试正常"}], **kwargs
        )
        before = request.model_dump()
        result = renderer.render_chat(request)
        self.assertEqual(request.model_dump(), before)
        expected = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "请只回答：测试正常"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=renderer.in_think_mode(request),
        )
        self.assertEqual(result.rendered_prompt, expected)
        self.assertEqual(result.input_ids, self.tokenizer.encode(expected))
        return result.rendered_prompt

    def test_server_disabled(self):
        self.assertTrue(self.render().endswith("<think>\n\n</think>\n\n"))

    def test_server_enabled(self):
        self.assertTrue(self.render(ThinkingMode.ENABLED).endswith("<think>\n"))

    def test_request_enabled(self):
        self.assertTrue(self.render(enable_thinking=True).endswith("<think>\n"))

    def test_request_disabled(self):
        self.assertTrue(
            self.render(ThinkingMode.ENABLED, enable_thinking=False).endswith(
                "<think>\n\n</think>\n\n"
            )
        )

    def test_template_kwargs_enabled(self):
        self.render(chat_template_kwargs={"enable_thinking": True})

    def test_extra_template_kwargs_precedence(self):
        self.render(
            chat_template_kwargs={"enable_thinking": False},
            extra_configs={"chat_template_kwargs": {"enable_thinking": True}},
        )

    def test_zero_budget_disables_template(self):
        self.render(
            ThinkingMode.ENABLED,
            thinking_budget=0,
            chat_template_kwargs={"enable_thinking": True},
        )

    def test_video_inputs_preserved(self):
        renderer = Qwen35Renderer.__new__(Qwen35Renderer)
        renderer.tokenizer = self.tokenizer
        renderer.default_thinking_mode = ThinkingMode.DISABLED
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "/video.mp4"},
                            "preprocess_config": {
                                "fps": 6,
                                "max_frames": 180,
                                "resized_width": 1280,
                                "resized_height": 704,
                            },
                        },
                        {"type": "text", "text": "描述视频"},
                    ],
                }
            ]
        )
        before = request.model_dump()
        result = renderer.render_chat(request)
        self.assertEqual(request.model_dump(), before)
        self.assertEqual(len(result.multimodal_inputs), 1)
        media = result.multimodal_inputs[0]
        self.assertEqual(media.url, "/video.mp4")
        self.assertEqual(media.mm_type, MMUrlType.VIDEO)
        self.assertEqual(media.mm_preprocess_config.fps, 6)
        self.assertEqual(media.mm_preprocess_config.max_frames, 180)
        self.assertEqual(media.mm_preprocess_config.width, 1280)
        self.assertEqual(media.mm_preprocess_config.height, 704)
        self.assertIn("<|video_pad|>", result.rendered_prompt)
        self.assertTrue(result.rendered_prompt.endswith("<think>\n\n</think>\n\n"))


if __name__ == "__main__":
    unittest.main()
