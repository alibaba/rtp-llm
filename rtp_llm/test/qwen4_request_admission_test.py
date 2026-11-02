import unittest
from unittest.mock import patch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.qwen35_renderer import Qwen4Renderer, Qwen35Renderer


class Qwen4RequestAdmissionTest(unittest.TestCase):
    def test_image_is_rejected_before_rendering(self):
        renderer = Qwen4Renderer.__new__(Qwen4Renderer)
        image = {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/image.png"},
        }
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": [image]}],
        )
        with patch.object(
            Qwen35Renderer,
            "render_chat",
            side_effect=AssertionError("must reject before preprocessing"),
        ):
            with self.assertRaisesRegex(FtRuntimeException, "text inputs only"):
                renderer.render_chat(request)

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
