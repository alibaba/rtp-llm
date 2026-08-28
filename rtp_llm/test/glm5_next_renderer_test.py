import unittest

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.chatglm45_renderer import Glm5NextRenderer
from rtp_llm.utils.base_model_datatypes import MMUrlType


class _Tokenizer:
    @staticmethod
    def encode(prompt):
        return list(range(len(prompt)))


class Glm5NextRendererTest(unittest.TestCase):
    def test_render_multimodal_content(self):
        renderer = Glm5NextRenderer.__new__(Glm5NextRenderer)
        renderer.tokenizer = _Tokenizer()
        renderer._build_prompt = lambda request: "".join(
            part.text or ""
            for message in request.messages
            if isinstance(message.content, list)
            for part in message.content
        )
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
                        {
                            "type": "video_url",
                            "video_url": {"url": "/tmp/video.mp4", "fps": 2.0},
                        },
                    ],
                }
            ]
        )

        rendered = renderer.render_chat(request)

        self.assertEqual(
            rendered.rendered_prompt,
            "<|begin_of_image|><|image|><|end_of_image|>\ndescribe it"
            "<|begin_of_video|><|video|><|end_of_video|>\n",
        )
        self.assertEqual(
            [item.url for item in rendered.multimodal_inputs],
            ["/tmp/image.jpg", "/tmp/video.mp4"],
        )
        self.assertEqual(
            [item.mm_type for item in rendered.multimodal_inputs],
            [MMUrlType.IMAGE, MMUrlType.VIDEO],
        )
        self.assertEqual(rendered.multimodal_inputs[1].mm_preprocess_config.fps, 2.0)
        self.assertEqual(len(rendered.input_ids), len(rendered.rendered_prompt))


if __name__ == "__main__":
    unittest.main()
