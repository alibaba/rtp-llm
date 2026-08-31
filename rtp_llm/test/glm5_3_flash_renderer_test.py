import unittest

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.chatglm45_renderer import Glm53FlashRenderer
from rtp_llm.utils.base_model_datatypes import MMUrlType


class _Tokenizer:
    @staticmethod
    def encode(prompt):
        return list(range(len(prompt)))


class Glm53FlashRendererTest(unittest.TestCase):
    def _renderer(self):
        renderer = Glm53FlashRenderer.__new__(Glm53FlashRenderer)
        renderer.tokenizer = _Tokenizer()
        renderer._build_prompt = lambda request: "".join(
            part.text or ""
            for message in request.messages
            if isinstance(message.content, list)
            for part in message.content
        )
        return renderer

    def test_image_content_is_forwarded_to_vit(self):
        renderer = self._renderer()
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "/tmp/image.jpg"},
                        },
                        {"type": "text", "text": "describe it"},
                    ],
                }
            ]
        )

        rendered = renderer.render_chat(request)

        self.assertEqual(
            rendered.rendered_prompt,
            "<|begin_of_image|><|image|><|end_of_image|>\ndescribe it",
        )
        self.assertEqual(
            [item.url for item in rendered.multimodal_inputs], ["/tmp/image.jpg"]
        )
        self.assertEqual(
            [item.mm_type for item in rendered.multimodal_inputs], [MMUrlType.IMAGE]
        )
        self.assertEqual(len(rendered.input_ids), len(rendered.rendered_prompt))

    def test_video_is_rejected_at_renderer_boundary(self):
        renderer = self._renderer()
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "/tmp/video.mp4", "fps": 2.0},
                        }
                    ],
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "does not support video input"):
            renderer.render_chat(request)

    def test_checkpoint_forced_think_mode_is_always_enabled(self):
        renderer = self._renderer()
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

        self.assertTrue(renderer.in_think_mode(request))

    def test_explicit_disable_cannot_bypass_checkpoint_forced_parser(self):
        renderer = self._renderer()
        renderer._build_prompt = lambda request: "<think>"
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            chat_template_kwargs={"enable_thinking": False},
        )

        parser = renderer._create_reasoning_parser(request)
        self.assertTrue(request.disable_thinking())
        self.assertTrue(renderer.in_think_mode(request))
        self.assertIsNotNone(parser)
        reasoning, content = parser.parse_non_stream(
            "I should answer briefly.</think>Hello!"
        )
        self.assertEqual(reasoning, "I should answer briefly.")
        self.assertEqual(content, "Hello!")

    def test_default_think_parser_splits_implicit_reasoning(self):
        renderer = self._renderer()
        renderer._build_prompt = lambda request: "<think>"
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}]
        )

        parser = renderer._create_reasoning_parser(request)
        self.assertIsNotNone(parser)
        reasoning, content = parser.parse_non_stream(
            "I should answer briefly.</think>Hello!"
        )
        self.assertEqual(reasoning, "I should answer briefly.")
        self.assertEqual(content, "Hello!")


if __name__ == "__main__":
    unittest.main()
