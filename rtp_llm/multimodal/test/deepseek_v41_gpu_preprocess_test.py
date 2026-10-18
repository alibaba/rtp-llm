"""Byte-exact DeepSeek-V4.1 backend GPU preprocessing parity tests.

Run only through the repository Bazel GPU lock.  The unchanged CPU path is the
reference; no tolerance or regenerated golden is used.
"""

import io
import unittest

import numpy as np
import torch
from PIL import Image

from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
    deepseek_v41_processor as processor,
)


def pattern(width, height, channels=3, seed=0):
    y, x = np.indices((height, width), dtype=np.uint32)
    return np.stack(
        [
            ((x * 17 + y * 31 + channel * 79 + seed) % 256).astype(np.uint8)
            for channel in range(channels)
        ],
        axis=-1,
    )


def encoded(image, image_format="PNG", **options):
    output = io.BytesIO()
    image.save(output, format=image_format, **options)
    return output.getvalue()


def media_cases():
    cases = []

    def add(name, image, image_format="PNG", **options):
        cases.append((name, encoded(image, image_format, **options)))

    for width, height, name in (
        (1, 1, "one_pixel"),
        (1024, 1, "single_row"),
        (1, 1024, "single_column"),
        (4096, 16, "extreme_wide"),
        (16, 4096, "extreme_tall"),
        (17, 31, "small_upscale"),
        (333, 221, "odd_landscape"),
        (221, 333, "odd_portrait"),
        (511, 513, "near_square_odd"),
        (1920, 1080, "hd_downsample"),
    ):
        add(name, Image.fromarray(pattern(width, height)))
    rgb = pattern(257, 259)
    rgba = np.concatenate([rgb, np.full((259, 257, 1), 128, np.uint8)], axis=2)
    add("rgba", Image.fromarray(rgba, "RGBA"))
    add("gray", Image.fromarray(pattern(333, 221)[..., 0], "L"))
    palette = Image.fromarray(pattern(127, 129)[..., 0], "P")
    palette.putpalette([v for i in range(256) for v in (i, 255 - i, (i * 13) % 256)])
    palette.info["transparency"] = bytes(range(256))
    add("palette_alpha", palette)
    add("jpeg", Image.fromarray(pattern(333, 221)), "JPEG", quality=91)
    return cases


class DeepSeekV41GpuPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; launch this target through gpu_lock")
        cls.device = torch.device("cuda", torch.cuda.current_device())
        cls.config = processor.V41ImageProcessorConfig()

    def assert_bytes_equal(self, expected, actual, label):
        self.assertEqual(expected.dtype, actual.dtype, f"{label}: dtype")
        self.assertEqual(expected.shape, actual.shape, f"{label}: shape")
        expected_bits = expected.detach().contiguous().view(torch.uint8).cpu()
        actual_bits = actual.detach().contiguous().view(torch.uint8).cpu()
        difference = (expected_bits != actual_bits).flatten()
        if difference.any().item():
            first = difference.nonzero()[0].item()
            self.fail(
                f"{label}: byte mismatch at {first}: "
                f"expected={expected_bits.flatten()[first].item()} "
                f"actual={actual_bits.flatten()[first].item()}"
            )

    def test_generated_modes_and_geometry_are_byte_exact(self):
        retained = []
        for name, payload in media_cases():
            with self.subTest(name=name), Image.open(io.BytesIO(payload)) as image:
                expected, vit_h, vit_w, _, _ = processor.preprocess_image(
                    image, self.config
                )
                rgb = processor._rgb_tensor(image)
                snapshot = rgb.clone()
                actual = processor._preprocess_rgb_on_device(
                    rgb,
                    (
                        vit_h * self.config.vision_patch_size,
                        vit_w * self.config.vision_patch_size,
                    ),
                    self.config.vision_patch_size,
                    self.device,
                )
                self.assertTrue(torch.equal(rgb, snapshot), f"{name}: input mutated")
                self.assert_bytes_equal(expected, actual, name)
                retained.append((expected.clone(), actual))

        # Later launches and allocator reuse must not mutate prior outputs.
        for index, (expected, actual) in enumerate(retained):
            self.assert_bytes_equal(expected, actual, f"retained-{index}")

    def test_repeated_and_reordered_requests_preserve_values(self):
        selected = media_cases()[5:9]
        snapshots = {}
        for name, payload in selected + list(reversed(selected)) + selected:
            with Image.open(io.BytesIO(payload)) as image:
                expected, vit_h, vit_w, _, _ = processor.preprocess_image(
                    image, self.config
                )
                actual = processor._preprocess_rgb_on_device(
                    processor._rgb_tensor(image),
                    (
                        vit_h * self.config.vision_patch_size,
                        vit_w * self.config.vision_patch_size,
                    ),
                    self.config.vision_patch_size,
                    self.device,
                )
            self.assert_bytes_equal(expected, actual, name)
            current = actual.detach().contiguous().view(torch.uint8).cpu().clone()
            if name in snapshots:
                self.assertTrue(torch.equal(snapshots[name], current))
            else:
                snapshots[name] = current

    def test_gpu_helper_rejects_non_cuda_without_copy(self):
        rgb = torch.zeros((31, 43, 3), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "explicit CUDA device"):
            processor._preprocess_rgb_on_device(rgb, (42, 56), 14, "cpu")


if __name__ == "__main__":
    unittest.main()
