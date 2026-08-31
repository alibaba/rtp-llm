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

    def _template_renderer(self):
        renderer = Glm53FlashRenderer.__new__(Glm53FlashRenderer)
        renderer.tokenizer = _Tokenizer()
        renderer.chat_template = (
            "{% if add_generation_prompt %}<|assistant|><think>{% endif %}"
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

    def test_video_content_is_forwarded_to_vit(self):
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

        rendered = renderer.render_chat(request)

        self.assertEqual(
            rendered.rendered_prompt,
            "<|begin_of_video|><|video|><|end_of_video|>\n",
        )
        self.assertEqual(
            [item.url for item in rendered.multimodal_inputs], ["/tmp/video.mp4"]
        )
        self.assertEqual(
            [item.mm_type for item in rendered.multimodal_inputs], [MMUrlType.VIDEO]
        )
        self.assertAlmostEqual(
            rendered.multimodal_inputs[0].mm_preprocess_config.fps, 2.0
        )

    def test_checkpoint_forced_think_mode_is_always_enabled(self):
        renderer = self._renderer()
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

        self.assertTrue(renderer.in_think_mode(request))

    def test_explicit_disable_closes_checkpoint_forced_think_prefix(self):
        for request_kwargs in (
            {"enable_thinking": False},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ):
            with self.subTest(request_kwargs=request_kwargs):
                renderer = self._template_renderer()
                request = ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hello"}],
                    **request_kwargs,
                )

                rendered = renderer.render_chat(request)
                parser = renderer._create_reasoning_parser(request)

                self.assertTrue(request.disable_thinking())
                self.assertTrue(renderer.in_think_mode(request))
                self.assertEqual(
                    rendered.rendered_prompt,
                    "<|assistant|><think></think>",
                )
                self.assertIsNotNone(parser)
                reasoning, content = parser.parse_non_stream("Hello!")
                self.assertEqual(reasoning, "")
                self.assertEqual(content, "Hello!")

    def test_default_generation_keeps_checkpoint_forced_think_prefix(self):
        renderer = self._template_renderer()
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}]
        )

        rendered = renderer.render_chat(request)
        parser = renderer._create_reasoning_parser(request)

        self.assertEqual(rendered.rendered_prompt, "<|assistant|><think>")
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
