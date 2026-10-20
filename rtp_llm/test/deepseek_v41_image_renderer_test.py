import base64
import io
import os
import unittest
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import torch
from PIL import Image
from tokenizers import Tokenizer

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.deepseekv41_renderer import DeepseekV41Renderer
from rtp_llm.utils.base_model_datatypes import GenerateInput, MMUrlType


def data_url(color):
    stream = io.BytesIO()
    Image.new("RGB", (42, 84), color=color).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class V41ImageRendererTest(TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_env = os.environ.get("DSV41_MODEL_PATH")
        if not checkpoint_env:
            raise unittest.SkipTest(
                "DSV41_MODEL_PATH not set; checkpoint-dependent tests run where the "
                "V4.1 model is available"
            )
        checkpoint = Path(checkpoint_env)
        backend = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))

        class Adapter:
            unk_token_id = None

            def encode(self, text):
                return backend.encode(text, add_special_tokens=False).ids

            def convert_tokens_to_ids(self, text):
                return backend.token_to_id(text)

        cls.renderer = DeepseekV41Renderer.__new__(DeepseekV41Renderer)
        cls.renderer.encoding_module = cls.renderer._load_encoding_module(
            str(checkpoint)
        )
        cls.renderer.tokenizer = Adapter()
        cls.renderer.think_mode = False

    def test_mixin_preserves_norm_precision_and_three_delimiters(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_mixin import (
            V41VitWeights,
        )
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            DeepSeekV41VisionEmbedding,
        )

        config = V41Config.from_path(os.environ["DSV41_MODEL_PATH"])
        with torch.device("meta"):
            adapter = DeepSeekV41VisionEmbedding(config)
        state = adapter.state_dict()
        weights = V41VitWeights({"vision_model": adapter})
        self.assertEqual(set(weights.weight_names), set(state))
        self.assertEqual(weights.ckpt_prefix, "")
        fp32_norms = {
            name for name, tensor in state.items() if tensor.dtype == torch.float32
        }
        self.assertEqual(len(fp32_norms), 65)
        self.assertTrue(
            all(
                name.endswith((".norm1.weight", ".norm2.weight"))
                or name == "vision.norm.weight"
                for name in fp32_norms
            )
        )
        self.assertEqual(
            {name for name in state if name.startswith("image_")},
            {"image_start", "image_newline", "image_end"},
        )

    def test_user_and_tool_images_keep_prompt_order(self):
        first, second = data_url((17, 23, 42)), data_url((73, 19, 2))
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Inspect"},
                            {"type": "image_url", "image_url": {"url": first}},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": [
                            {"type": "image_url", "image_url": {"url": second}}
                        ],
                    },
                    {"role": "user", "content": "Compare the two images."},
                ],
                "max_tokens": 256,
            }
        )
        original = request.model_dump()
        prompt, media = self.renderer._encode_request(request)
        self.assertEqual([image["url"] for image in media["images"]], [first, second])
        rendered = self.renderer.render_chat(request)
        self.assertEqual(rendered.rendered_prompt, prompt)
        self.assertEqual(rendered.input_ids, self.renderer.tokenizer.encode(prompt))
        self.assertEqual(rendered.input_ids.count(self.renderer.image_token_id), 2)
        self.assertEqual(
            [
                base64.b64decode(item.url.split(",", 1)[1])
                for item in rendered.multimodal_inputs
            ],
            [base64.b64decode(url.split(",", 1)[1]) for url in (first, second)],
        )
        self.assertEqual(
            [item.mm_type for item in rendered.multimodal_inputs],
            [MMUrlType.IMAGE] * 2,
        )
        self.assertEqual(request.model_dump(), original)
        normal_input = GenerateInput(
            request_id=17,
            token_ids=torch.tensor(rendered.input_ids, dtype=torch.int32),
            mm_inputs=rendered.multimodal_inputs,
            generate_config=GenerateConfig(max_new_tokens=256),
        )
        normal_wire = trans_input(normal_input)
        self.assertEqual(list(normal_wire.token_ids), rendered.input_ids)
        self.assertEqual(len(normal_wire.multimodal_inputs), 2)
        self.assertEqual(
            [item.multimodal_url for item in normal_wire.multimodal_inputs],
            [item.url for item in rendered.multimodal_inputs],
        )

    def test_text_remains_renderable_and_resize_hints_are_ignored(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]}
                ]
            }
        )
        actual = self.renderer.render_chat(request)
        self.assertEqual(
            actual.input_ids, self.renderer.tokenizer.encode(actual.rendered_prompt)
        )
        self.assertEqual(actual.input_ids.count(self.renderer.image_token_id), 0)
        self.assertEqual(actual.multimodal_inputs, [])
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url((0, 0, 0))},
                                "preprocess_config": {"resized_height": 123},
                            }
                        ],
                    }
                ]
            }
        )
        original = request.model_dump()
        actual = self.renderer.render_chat(request)
        baseline = request.model_dump()
        baseline["messages"][0]["content"][0]["preprocess_config"] = None
        expected = self.renderer.render_chat(
            ChatCompletionRequest.model_validate(baseline)
        )
        self.assertEqual(actual.rendered_prompt, expected.rendered_prompt)
        self.assertEqual(actual.input_ids, expected.input_ids)
        self.assertEqual(len(actual.multimodal_inputs), 1)
        self.assertEqual(
            actual.multimodal_inputs[0].url, expected.multimodal_inputs[0].url
        )
        self.assertEqual(actual.multimodal_inputs[0].mm_type, MMUrlType.IMAGE)
        self.assertEqual(request.model_dump(), original)

    def test_normal_render_does_not_decode_pixels(self):
        url = data_url((17, 23, 42))
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
            }
        )
        with patch(
            "PIL.Image.open", side_effect=AssertionError("frontend decoded pixels")
        ):
            result = self.renderer.render_chat(request)
        self.assertEqual(result.input_ids.count(self.renderer.image_token_id), 1)
        self.assertEqual(
            base64.b64decode(result.multimodal_inputs[0].url.split(",", 1)[1]),
            base64.b64decode(url.split(",", 1)[1]),
        )

    def test_normal_render_rejects_placeholder_mismatch_without_loading_image(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url((1, 2, 3))},
                            },
                        ],
                    }
                ],
            }
        )
        with patch.object(self.renderer.tokenizer, "encode", return_value=[17, 18]):
            with self.assertRaisesRegex(FtRuntimeException, "placeholder count"):
                self.renderer.render_chat(request)

    def test_image_bytes_are_forwarded_without_frontend_validation(self):
        png = base64.b64decode(data_url((1, 2, 3)).split(",", 1)[1])
        for payload in (b"not an image", png[:40]):
            with self.subTest(payload=payload):
                url = "data:image/png;base64," + base64.b64encode(payload).decode()
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": url}}
                                ],
                            }
                        ]
                    }
                )
                with patch(
                    "PIL.Image.open",
                    side_effect=AssertionError("frontend decoded pixels"),
                ):
                    rendered = self.renderer.render_chat(request)
                self.assertEqual(
                    base64.b64decode(
                        rendered.multimodal_inputs[0].url.split(",", 1)[1]
                    ),
                    payload,
                )


if __name__ == "__main__":
    main()
