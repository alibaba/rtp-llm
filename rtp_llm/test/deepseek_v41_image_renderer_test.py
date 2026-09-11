import base64
import io
import os
from dataclasses import replace
from pathlib import Path
from unittest import TestCase, main

import torch
from PIL import Image
from tokenizers import Tokenizer

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import GenerateInputPB
from rtp_llm.models.multimodal.deepseek_v41_vision import DeepSeekV41VisionEmbedding
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.deepseekv41_renderer import DeepseekV41Renderer
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.utils.grpc_util import trans_tensor


def data_url(color):
    stream = io.BytesIO()
    Image.new("RGB", (42, 84), color=color).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class V41ImageRendererTest(TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = Path(os.environ["DSV41_MODEL_PATH"])
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

    def test_weight_descriptor_preserves_norm_overrides_and_three_delimiters(self):
        config = V41Config.from_path(os.environ["DSV41_MODEL_PATH"])
        with torch.device("meta"):
            adapter = DeepSeekV41VisionEmbedding(config)
        weights = adapter.create_weight_info()
        self.assertEqual(len(weights.weight_names), 266)
        self.assertEqual(len(weights.weight_dtypes), 65)
        self.assertTrue(
            all(dtype == torch.float32 for dtype in weights.weight_dtypes.values())
        )
        self.assertEqual(
            {name for name in weights.weight_names if name.startswith("image_")},
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
        _, media = self.renderer._encode_request(request)
        self.assertEqual([image["url"] for image in media["images"]], [first, second])
        prepared = self.renderer.prepare_v41_inputs(request)
        self.assertEqual(len(prepared.images), 2)
        self.assertLess(prepared.images[0].start, prepared.images[1].start)
        self.assertEqual(
            prepared.image_mask.sum().item(),
            sum(image.length for image in prepared.images),
        )
        self.assertNotEqual(*prepared.image_content_hashes)
        self.assertEqual(request.model_dump(), original)
        rendered = self.renderer.render_chat(request)
        self.assertEqual(rendered.input_ids, list(prepared.token_ids))
        self.assertEqual(rendered.multimodal_inputs, [])
        self.assertEqual(
            rendered.v41_inputs.image_content_hashes, prepared.image_content_hashes
        )
        generate_input = GenerateInput(
            request_id=17,
            token_ids=torch.tensor(rendered.input_ids, dtype=torch.int32),
            mm_inputs=[],
            generate_config=GenerateConfig(max_new_tokens=256),
            v41_inputs=rendered.v41_inputs,
        )
        wire = trans_input(generate_input)
        restored = GenerateInputPB.FromString(wire.SerializeToString())
        self.assertEqual(restored.v41_inputs.schema_version, 1)
        self.assertEqual(list(restored.token_ids), rendered.input_ids)
        self.assertEqual(tuple(restored.v41_inputs.token_types), prepared.token_types)
        self.assertEqual(
            list(restored.v41_inputs.image_mask), prepared.image_mask.tolist()
        )
        self.assertEqual(len(restored.multimodal_inputs), 0)
        for source, destination in zip(prepared.images, restored.v41_inputs.images):
            self.assertEqual(destination.start, source.start)
            self.assertEqual(destination.processor_identity, source.processor_identity)
            self.assertEqual(destination.content_sha256, source.content_sha256)
            self.assertEqual(list(destination.types), source.types.tolist())
            self.assertTrue(
                torch.equal(trans_tensor(destination.patches), source.patches)
            )
        changed_ids = generate_input.token_ids.clone()
        changed_ids[prepared.images[0].start] = 7
        with self.assertRaisesRegex(ValueError, "canonical"):
            trans_input(replace(generate_input, token_ids=changed_ids))
        with self.assertRaisesRegex(ValueError, "prefixes"):
            generate_input.update_prefix(torch.tensor([7], dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "output budget"):
            trans_input(
                replace(
                    generate_input,
                    generate_config=GenerateConfig(max_new_tokens=1048576),
                )
            )

    def test_text_remains_renderable_and_resize_override_fails(self):
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]}
                ]
            }
        )
        actual = self.renderer.render_chat(request)
        prepared = self.renderer.prepare_v41_inputs(request)
        self.assertEqual(actual.input_ids, list(prepared.token_ids))
        self.assertEqual(prepared.images, ())
        self.assertIsNotNone(actual.v41_inputs)
        self.assertFalse(actual.v41_inputs.image_mask.any().item())
        appended = prepared.append_text(" tail", [17, 18])
        self.assertEqual(appended.token_types[-2:], (-1, -1))
        with self.assertRaisesRegex(ValueError, "placeholders"):
            prepared.append_text(" image", [129264])
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
        with self.assertRaisesRegex(FtRuntimeException, "fixed by the model config"):
            self.renderer.prepare_v41_inputs(request)


if __name__ == "__main__":
    main()
