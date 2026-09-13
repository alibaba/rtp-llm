"""Real pixels through the pinned processor, full ViT and embedding injector."""

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).parent))
import deepseek_v41_processor_test as processor_fixture
import deepseek_v41_vision_test as vision_fixture
from rtp_llm.models.multimodal.deepseek_v41_processor import (
    IMAGE_PLACEHOLDER,
    TEXT,
    prepare_vl_inputs,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.deepseekv41_renderer import DeepseekV41Renderer


class V41PixelPipelineTest(TestCase):
    @classmethod
    def setUpClass(cls):
        vision_fixture.V41VisionEmbeddingTest.setUpClass.__func__(cls)
        processor_fixture.V41ProcessorTest.setUpClass.__func__(cls)
        cls.reference_args = SimpleNamespace(
            **cls.adapter.processor_config.__dict__, vision_enabled=True
        )

    def compare_pipeline(self, name, prompt, records, prepared):
        with patch.dict(sys.modules, {"encoding": self.encoding}):
            token_ids, token_types, images = self.reference.prepare_vl_inputs(
                prompt, records, self.tokenizer, self.reference_args
            )
        self.assertEqual(prepared.token_ids, tuple(token_ids))
        self.assertEqual(prepared.token_types, tuple(token_types))
        self.assertEqual(len(prepared.images), len(images))
        text_rows = torch.tensor(token_types, device="cuda") == TEXT
        initial = (
            (torch.arange(len(token_ids), device="cuda", dtype=torch.float32) % 17)[
                :, None
            ]
            .expand(-1, self.adapter.image_start.numel())
            .to(torch.bfloat16)
        )
        expected = initial.clone()
        cases = []
        for actual, reference in zip(prepared.images, images):
            self.assertEqual(
                (actual.start, actual.n_vit_h, actual.n_vit_w),
                (reference.start, reference.n_vit_h, reference.n_vit_w),
            )
            torch.testing.assert_close(
                actual.patches, reference.patches, rtol=0, atol=0
            )
            self.assertTrue(torch.equal(actual.types, reference.types))
            patches = reference.patches.to(device="cuda", dtype=torch.bfloat16)
            # The pinned reference constructs RoPE tensors on the default device.
            with torch.device(patches.device):
                features = self.reference_vision(
                    patches, reference.n_vit_h, reference.n_vit_w
                )
                aligned = self.reference_aligner(
                    features, reference.n_vit_h, reference.n_vit_w
                )
            height, width = (reference.n_vit_h + 2) // 3, (reference.n_vit_w + 2) // 3
            parts = [self.delimiters["image_start"].unsqueeze(0)]
            for row in aligned.reshape(height, width, -1):
                parts.extend((row, self.delimiters["image_newline"].unsqueeze(0)))
            parts.append(self.delimiters["image_end"].unsqueeze(0))
            span = torch.cat(parts)
            self.assertEqual(span.shape[0], actual.length)
            expected[reference.start : reference.start + span.shape[0]] = span
            cases.append(
                {
                    "start": reference.start,
                    "rows": span.shape[0],
                    "vit_grid": [reference.n_vit_h, reference.n_vit_w],
                }
            )
        calls = []
        hook = self.adapter.vision.register_forward_hook(lambda *args: calls.append(1))
        try:
            actual = self.adapter.inject_embeddings(initial.clone(), prepared)
        finally:
            hook.remove()
        self.assertEqual(len(calls), len(images))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[text_rows], initial[text_rows]))
        self.assertTrue(torch.isfinite(actual).all().item())
        folder = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        tensor_path = folder / (name + ".pt")
        torch.save(
            {
                "actual": actual.cpu(),
                "expected": expected.cpu(),
                "token_ids": token_ids,
                "token_types": token_types,
            },
            tensor_path,
        )
        (folder / (name + ".json")).write_text(
            json.dumps(
                {
                    "case": name,
                    "images": cases,
                    "vision_calls": len(calls),
                    "token_count": len(token_ids),
                    "mismatches": 0,
                    "rtol": 0,
                    "atol": 0,
                    "pixel_hashes": list(prepared.image_content_hashes),
                    "tensor_sha256": hashlib.sha256(
                        tensor_path.read_bytes()
                    ).hexdigest(),
                    "vision_source_sha256": vision_fixture.VISION_SHA256,
                    "processor_source_sha256": processor_fixture.PROCESSOR_SHA256,
                    "encoding_source_sha256": processor_fixture.ENCODING_SHA256,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @torch.inference_mode()
    def test_pixels_to_full_vision_and_injection_match_official(self):
        with sdpa_kernel(SDPBackend.MATH):
            for width, height, mode in (
                (1, 1, "RGB"),
                (17, 31, "RGBA"),
                (545, 546, "L"),
                (1, 8192, "RGB"),
                (8192, 1, "RGB"),
                (1400, 900, "RGB"),
            ):
                with self.subTest(size=(width, height), mode=mode):
                    records = [
                        {"data": processor_fixture.image_data(width, height, mode)}
                    ]
                    prompt = f"prefix {IMAGE_PLACEHOLDER} suffix"
                    prepared = prepare_vl_inputs(
                        prompt,
                        records,
                        self.tokenizer,
                        self.adapter.processor_config,
                        output_budget=256,
                    )
                    self.compare_pipeline(
                        f"pixels-{width}-{height}-{mode}", prompt, records, prepared
                    )

    @torch.inference_mode()
    def test_user_and_tool_pixels_reach_distinct_complete_image_spans(self):
        model = Path(os.environ["DSV41_MODEL_PATH"])
        renderer = DeepseekV41Renderer.__new__(DeepseekV41Renderer)
        renderer.encoding_module = renderer._load_encoding_module(str(model))
        renderer.tokenizer = self.tokenizer
        renderer.think_mode = False
        urls = [
            "data:image/png;base64,"
            + base64.b64encode(
                processor_fixture.image_data(41, 83, offset=offset)
            ).decode()
            for offset in (0, 37)
        ]
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Inspect"},
                            {"type": "image_url", "image_url": {"url": urls[0]}},
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
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": [
                            {"type": "image_url", "image_url": {"url": urls[1]}},
                        ],
                    },
                    {"role": "user", "content": "Compare the two images."},
                ],
                "max_tokens": 256,
            }
        )
        original = request.model_dump()
        prompt, media = renderer._encode_request(request)
        self.assertEqual([record["url"] for record in media["images"]], urls)
        prepared = renderer.prepare_v41_inputs(request)
        self.assertEqual(len(prepared.images), 2)
        self.assertNotEqual(*prepared.image_content_hashes)
        self.assertEqual(request.model_dump(), original)
        with sdpa_kernel(SDPBackend.MATH):
            self.compare_pipeline("user-tool", prompt, media["images"], prepared)


if __name__ == "__main__":
    main()
