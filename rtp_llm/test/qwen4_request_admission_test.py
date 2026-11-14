import unittest
from unittest.mock import patch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.qwen35_renderer import Qwen4Renderer, Qwen35Renderer


class Qwen4RequestAdmissionTest(unittest.TestCase):
    def test_image_and_video_use_the_shared_renderer(self):
        renderer = Qwen4Renderer.__new__(Qwen4Renderer)
        image = {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/image.png"},
        }
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": [image]}],
        )
        expected = object()
        with patch.object(Qwen35Renderer, "render_chat", return_value=expected):
            self.assertIs(renderer.render_chat(request), expected)
        request = ChatCompletionRequest(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "https://example.invalid/video.mp4"},
                        }
                    ],
                }
            ],
        )
        with patch.object(Qwen35Renderer, "render_chat", return_value=expected):
            self.assertIs(renderer.render_chat(request), expected)

    def test_text_request_keeps_shared_renderer(self):
        renderer = Qwen4Renderer.__new__(Qwen4Renderer)
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )
        expected = object()
        with patch.object(Qwen35Renderer, "render_chat", return_value=expected):
            self.assertIs(renderer.render_chat(request), expected)


if __name__ == "__main__":
    unittest.main()
