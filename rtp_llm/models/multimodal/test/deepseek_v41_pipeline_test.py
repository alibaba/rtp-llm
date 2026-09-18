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
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
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


class V41SyntheticImageRowInjectionTest(TestCase):
    """Row-mapping and injection glue on synthetic features (no weights).

    Exercises ``_image_row_indices``/``_inject_image_rows`` with fabricated CP
    metadata so the rank-local derivation (including image spans straddling
    the CP chunk boundary and decode-prefix rows) stays covered independently
    of the weight-backed pipeline cases above.
    """

    @staticmethod
    def bare_model():
        model = object.__new__(DeepSeekV41Model)
        model._image_plan = None
        return model

    @staticmethod
    def fake_inputs(features, locs, cp=None):
        return SimpleNamespace(
            multimodal_features=features,
            mm_features_locs=torch.tensor(locs, dtype=torch.int32),
            attention_inputs=SimpleNamespace(context_parallel_info=cp),
        )

    def test_global_rows_and_injection_without_cp(self):
        model = self.bare_model()
        first = torch.arange(6, dtype=torch.float32)[:, None].expand(-1, 8).contiguous() + 100.0
        second = torch.arange(4, dtype=torch.float32)[:, None].expand(-1, 8).contiguous() + 200.0
        model._prepare_image_features(
            self.fake_inputs([first, second], [3, 10])
        )
        rows = model._image_row_indices(16, torch.device("cpu"))
        expected = torch.full((16,), -1, dtype=torch.int64)
        expected[3:9] = torch.arange(6)
        expected[10:14] = torch.arange(6, 10)
        self.assertTrue(torch.equal(rows, expected))

        hidden = torch.arange(16, dtype=torch.float32)[:, None].expand(-1, 8).contiguous()
        injected = model._inject_image_rows(hidden.clone(), torch.zeros(16, dtype=torch.long))
        self.assertTrue(torch.equal(injected[3:9], first))
        self.assertTrue(torch.equal(injected[10:14], second))
        self.assertTrue(torch.equal(injected[:3], hidden[:3]))
        self.assertTrue(torch.equal(injected[9:10], hidden[9:10]))
        self.assertTrue(torch.equal(injected[14:], hidden[14:]))

    def test_cp_shuffle_maps_rank_local_rows_across_straddling_image(self):
        model = self.bare_model()
        features = torch.arange(4, dtype=torch.float32)[:, None].expand(-1, 8).contiguous() + 50.0
        # One prefill request of 12 tokens; the image occupies [4, 8), which
        # straddles the CP chunk boundary at 7. This rank keeps local tokens
        # {0, 5, pad, 6, 7, 11} of the request.
        cp = SimpleNamespace(
            prefill_shuffle_indices=torch.tensor([0, 5, -1, 6, 7, 11], dtype=torch.int64),
            prefill_actual_input_lengths_cpu=torch.tensor([12]),
            prefill_cp_chunk_lengths=torch.tensor([7, 5], dtype=torch.int64),
        )
        model._prepare_image_features(self.fake_inputs([features], [4], cp=cp))
        rows = model._image_row_indices(6, torch.device("cpu"))
        self.assertEqual(rows.tolist(), [-1, 1, -1, 2, 3, -1])

        hidden = torch.zeros(6, 8)
        injected = model._inject_image_rows(hidden, torch.zeros(6, dtype=torch.long))
        self.assertTrue(torch.equal(injected[1], features[1]))
        self.assertTrue(torch.equal(injected[3], features[2]))
        self.assertTrue(torch.equal(injected[4], features[3]))
        self.assertTrue(torch.equal(injected[:1], hidden[:1]))
        self.assertTrue(torch.equal(injected[2:3], hidden[2:3]))
        self.assertTrue(torch.equal(injected[5], hidden[5]))

    def test_cp_zigzag_padding_indices_stay_text_rows(self):
        # Zigzag shuffle indices are in the per-request PADDED coordinate
        # space: the odd pair reads from the padded tail, so entries can
        # reach padded_length - 1 >= actual_length. Regression: the gather
        # used to index rows out of bounds (device assert on the last
        # request) or silently cross into the next request's span.
        model = self.bare_model()
        features = torch.arange(4, dtype=torch.float32)[:, None].expand(-1, 8).contiguous() + 50.0
        # One request of 12 actual tokens padded to 16; this rank's odd pair
        # reads the padded tail: local tokens {0, 5, 12..15} where 12..15
        # are padding (>= actual length 12). The image occupies [4, 8).
        cp = SimpleNamespace(
            prefill_shuffle_indices=torch.tensor([0, 5, 12, 13, 14, 15], dtype=torch.int64),
            prefill_actual_input_lengths_cpu=torch.tensor([12]),
            prefill_cp_chunk_lengths=torch.tensor([6], dtype=torch.int64),
        )
        model._prepare_image_features(self.fake_inputs([features], [4], cp=cp))
        rows = model._image_row_indices(6, torch.device("cpu"))
        self.assertEqual(rows.tolist(), [-1, 1, -1, -1, -1, -1])

        # Two requests: the first one's zigzag padding indices must not
        # leak into the second request's image span. Request 0 spans global
        # [0, 6) with its image at [2, 6); request 1 spans [6, 12) with its
        # image at [7, 10). This rank keeps 2 tokens per request; request
        # 0's second entry is a padding index (7 >= actual 6).
        second = torch.arange(3, dtype=torch.float32)[:, None].expand(-1, 8).contiguous() + 150.0
        cp = SimpleNamespace(
            prefill_shuffle_indices=torch.tensor([1, 7, 0, 2], dtype=torch.int64),
            prefill_actual_input_lengths_cpu=torch.tensor([6, 6]),
            prefill_cp_chunk_lengths=torch.tensor([2, 2], dtype=torch.int64),
        )
        model._prepare_image_features(self.fake_inputs([features, second], [2, 7], cp=cp))
        rows = model._image_row_indices(4, torch.device("cpu"))
        # Local token 0 -> global 1 (text); local 1 -> padding (masked);
        # local 2 -> global 6 (text); local 3 -> global 8 (image row 5).
        self.assertEqual(rows.tolist(), [-1, -1, -1, 5])

    def test_decode_prefix_rows_stay_text(self):
        model = self.bare_model()
        features = torch.ones(2, 8)
        # Two decode rows precede one prefill request of 4 tokens (global
        # positions 2..5); the image occupies global [4, 6). This rank's
        # prefill tokens are request positions 2 and 3 (global 4 and 5).
        cp = SimpleNamespace(
            prefill_shuffle_indices=torch.tensor([2, 3], dtype=torch.int64),
            prefill_actual_input_lengths_cpu=torch.tensor([1, 1, 4]),
            prefill_cp_chunk_lengths=torch.tensor([2, 2], dtype=torch.int64),
        )
        model._prepare_image_features(self.fake_inputs([features], [4], cp=cp))
        rows = model._image_row_indices(4, torch.device("cpu"))
        self.assertEqual(rows.tolist(), [-1, -1, 0, 1])


if __name__ == "__main__":
    main()
