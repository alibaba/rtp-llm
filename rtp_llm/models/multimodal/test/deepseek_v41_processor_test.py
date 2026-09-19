import base64
import binascii
import hashlib
import importlib.util
import io
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

from rtp_llm.models.multimodal.deepseek_v41_processor import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    TEXT,
    V41ImageProcessorConfig,
    image_token_types,
    load_image_bytes,
    plan_image_grid,
    prepare_vl_inputs,
    preprocess_image,
)

PROCESSOR_SHA256 = "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272"
ENCODING_SHA256 = "f64a67e5680a5621b9320585a9684967cd5c75a9b82e19d914cbc02845d72cab"


def load_reference(name, path, expected_sha256):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(
            f"reference source does not match the fixed HF revision: {path.name}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def image_data(width, height, mode="RGB", offset=0):
    values = (
        (
            np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3)
            + offset
        )
        % 256
    ).astype(np.uint8)
    image = Image.fromarray(values, "RGB").convert(mode)
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


class V41ProcessorTest(TestCase):
    @classmethod
    def setUpClass(cls):
        model = Path(os.environ["DSV41_MODEL_PATH"])
        cls.reference = load_reference(
            "dsv41_official_image_processor",
            model / "inference/image_processor.py",
            PROCESSOR_SHA256,
        )
        cls.encoding = load_reference(
            "dsv41_processor_official_encoding",
            model / "encoding/encoding.py",
            ENCODING_SHA256,
        )
        backend = Tokenizer.from_file(str(model / "tokenizer.json"))

        class Adapter:
            unk_token_id = None

            def encode(self, text):
                return backend.encode(text, add_special_tokens=False).ids

            def convert_tokens_to_ids(self, text):
                return backend.token_to_id(text)

        cls.tokenizer = Adapter()
        cls.config = V41ImageProcessorConfig()
        cls.reference_args = type(
            "Args", (), {**cls.config.__dict__, "vision_enabled": True}
        )()
        print(
            json.dumps(
                {
                    "processor_source_sha256": PROCESSOR_SHA256,
                    "encoding_source_sha256": ENCODING_SHA256,
                    "model_path": str(model),
                }
            ),
            flush=True,
        )

    def test_geometry_and_types_match_official(self):
        for width, height in (
            (1, 1),
            (1, 8192),
            (8192, 1),
            (17, 31),
            (544, 544),
            (1000, 999),
            (3072, 4096),
            (16, 65535),
            (65535, 16),
        ):
            with self.subTest(size=(width, height)):
                actual = plan_image_grid(width, height, self.config)
                expected = self.reference.plan_image_grid(
                    width, height, self.reference_args
                )
                self.assertEqual(actual, expected)
                types = image_token_types(*actual[:2])
                self.assertTrue(
                    torch.equal(types, self.reference.image_token_types(*expected[:2]))
                )
                self.assertLessEqual(types.numel(), 1024)
        self.assertEqual(
            image_token_types(2, 3).tolist(),
            [
                IMAGE_START,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE,
                IMAGE,
                IMAGE,
                IMAGE_NEW_LINE,
                IMAGE_END,
            ],
        )

    def test_pixels_and_patch_order_match_official(self):
        for width, height, mode in (
            (1, 1, "RGB"),
            (17, 31, "RGBA"),
            (545, 546, "L"),
            (1, 8192, "RGB"),
            (8192, 1, "RGB"),
            (1400, 900, "RGB"),
        ):
            with self.subTest(size=(width, height), mode=mode):
                data = image_data(width, height, mode)
                with Image.open(io.BytesIO(data)) as image:
                    actual = preprocess_image(image, self.config)
                expected = self.reference.load_image(
                    {"data": data}, self.reference_args
                )
                self.assertEqual(actual[1:], expected[1:])
                self.assertEqual(actual[0].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(actual[0], expected[0]))

    def test_interleaved_images_keep_ids_mask_and_order(self):
        records = [
            {"data": image_data(41, 83)},
            {"data": image_data(73, 39, offset=13)},
        ]
        prompt = f"first {IMAGE_PLACEHOLDER} middle {IMAGE_PLACEHOLDER} final"
        actual = prepare_vl_inputs(
            prompt, records, self.tokenizer, self.config, output_budget=256
        )
        with patch.dict(sys.modules, {"encoding": self.encoding}):
            expected = self.reference.prepare_vl_inputs(
                prompt, records, self.tokenizer, self.reference_args
            )
        self.assertEqual(actual.token_ids, tuple(expected[0]))
        self.assertEqual(actual.token_types, tuple(expected[1]))
        self.assertTrue(
            torch.equal(actual.image_mask, torch.tensor(expected[1]) != TEXT)
        )
        self.assertEqual(
            actual.image_content_hashes,
            tuple(hashlib.sha256(record["data"]).hexdigest() for record in records),
        )
        for image, other in zip(actual.images, expected[2]):
            self.assertEqual(
                (image.start, image.n_vit_h, image.n_vit_w),
                (other.start, other.n_vit_h, other.n_vit_w),
            )
            self.assertTrue(torch.equal(image.patches, other.patches))
            self.assertTrue(torch.equal(image.types, other.types))
            self.assertEqual(image.processor_identity, self.config.identity)
            self.assertEqual(
                set(actual.token_ids[image.start : image.start + image.length]),
                {129264},
            )
        changed = prepare_vl_inputs(
            prompt, list(reversed(records)), self.tokenizer, self.config
        )
        self.assertEqual(
            actual.image_content_hashes, tuple(reversed(changed.image_content_hashes))
        )
        different_pixels = prepare_vl_inputs(
            prompt,
            [{"data": image_data(41, 83, offset=37)}, records[1]],
            self.tokenizer,
            self.config,
        )
        self.assertEqual(actual.token_ids, different_pixels.token_ids)
        self.assertEqual(actual.token_types, different_pixels.token_types)
        self.assertNotEqual(
            actual.image_content_hashes, different_pixels.image_content_hashes
        )

    def test_append_text_preserves_image_spans_and_identity(self):
        prompt = f"prefix {IMAGE_PLACEHOLDER} suffix"
        records = [{"data": image_data(41, 83)}]
        prepared = prepare_vl_inputs(prompt, records, self.tokenizer, self.config)
        suffix = " added text"
        suffix_tokens = tuple(self.tokenizer.encode(suffix))
        appended = prepared.append_text(suffix, iter(suffix_tokens))
        self.assertEqual(appended.prompt, prompt + suffix)
        self.assertEqual(appended.token_ids, prepared.token_ids + suffix_tokens)
        self.assertEqual(
            appended.token_types, prepared.token_types + (TEXT,) * len(suffix_tokens)
        )
        self.assertIs(appended.images, prepared.images)
        self.assertEqual(appended.image_content_hashes, prepared.image_content_hashes)
        self.assertTrue(
            torch.equal(
                appended.image_mask[: len(prepared.token_ids)], prepared.image_mask
            )
        )
        self.assertFalse(appended.image_mask[len(prepared.token_ids) :].any())
        self.assertEqual(prepared.prompt, prompt)

    def test_preparation_leaves_context_budget_to_generation(self):
        config = replace(self.config, max_seq_len=1)
        for prompt, records in (
            ("text request", []),
            (f"prefix {IMAGE_PLACEHOLDER} suffix", [{"data": image_data(41, 83)}]),
        ):
            with self.subTest(prompt=prompt):
                expected = prepare_vl_inputs(prompt, records, self.tokenizer, config)
                actual = prepare_vl_inputs(
                    prompt,
                    records,
                    self.tokenizer,
                    config,
                    output_budget=1048576,
                )
                self.assertGreater(len(actual.token_ids), config.max_seq_len)
                self.assertEqual(actual.token_ids, expected.token_ids)
                self.assertEqual(actual.token_types, expected.token_types)
                self.assertEqual(
                    actual.image_content_hashes, expected.image_content_hashes
                )
                self.assertEqual(
                    tuple(image.processor_identity for image in actual.images),
                    tuple(image.processor_identity for image in expected.images),
                )

    def test_invalid_image_records_still_report_parse_errors(self):
        prompt = f"prefix {IMAGE_PLACEHOLDER} suffix"
        record = {"data": image_data(3, 5)}
        for text, records in (
            (prompt, []),
            (prompt, [record, record]),
            ("no image placeholder", [record]),
        ):
            with self.subTest(text=text, count=len(records)):
                with self.assertRaisesRegex(ValueError, "placeholder count"):
                    prepare_vl_inputs(text, records, self.tokenizer, self.config)
        with self.assertRaisesRegex(ValueError, "invalid or damaged"):
            prepare_vl_inputs(
                prompt, [{"data": b"not an image"}], self.tokenizer, self.config
            )
        with self.assertRaisesRegex(ValueError, "no supported data or URL"):
            load_image_bytes({})
        with self.assertRaisesRegex(ValueError, "require base64"):
            load_image_bytes({"url": "data:image/png,plain-text"})
        with self.assertRaises(binascii.Error):
            load_image_bytes({"data": "!invalid-base64!"})

    def test_data_records_are_equivalent(self):
        data = image_data(3, 5)
        encoded = base64.b64encode(data).decode()
        for record in (
            {"data": data},
            {"data": encoded},
            {"source": {"data": encoded}},
            {"url": "data:image/png;base64," + encoded},
        ):
            self.assertEqual(load_image_bytes(record), data)


if __name__ == "__main__":
    main()
