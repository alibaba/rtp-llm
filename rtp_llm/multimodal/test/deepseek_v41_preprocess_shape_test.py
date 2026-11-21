"""CPU-only support-gate contracts; no CUDA or numerical tolerance changes."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import vision_preprocess_tables
from rtp_llm.models_py.modules.dsv41 import vision_preprocess_triton


class PreprocessShapeTest(unittest.TestCase):
    def test_native_coefficients_are_bit_exact_and_cached(self):
        import rtp_llm.ops  # Load the native config module as the worker does.
        from libth_transformer_config import vit_resize_coefficients

        cache = vision_preprocess_tables._coefficients
        cache.cache_clear()
        self.addCleanup(cache.cache_clear)
        with (
            patch.dict(os.environ, {"MM_VIT_NATIVE_RESIZE_TABLES": "1"}),
            patch(
                "libth_transformer_config.vit_resize_coefficients",
                wraps=vit_resize_coefficients,
            ) as native,
        ):
            for shape in ((1, 42), (31, 43), (33, 21), (1080, 756), (65536, 256)):
                with self.subTest(shape=shape):
                    actual = cache(*shape)
                    expected = vision_preprocess_tables._python_coefficients(*shape)
                    for lhs, rhs in zip(actual, expected):
                        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
                    self.assertIs(cache(*shape), actual)
            self.assertEqual(native.call_count, 5)

    def load_with_pillow_version(self, version):
        name = "v41_preprocess_version_probe"
        spec = importlib.util.spec_from_file_location(
            name, vision_preprocess_triton.__file__
        )
        module = importlib.util.module_from_spec(spec)
        with patch("PIL.__version__", version), patch.dict(sys.modules, {name: module}):
            spec.loader.exec_module(module)
        return module

    def test_geometry_is_independent_of_installed_pillow_version(self):
        for version in ("11.2.1", "12.3.0", "future-version"):
            with self.subTest(version=version):
                module = self.load_with_pillow_version(version)
                for source, target in (
                    ((1, 1), (546, 546)),
                    ((31, 43), (42, 56)),
                    ((1080, 1920), (756, 1344)),
                    ((65535, 16), (21462, 42)),
                ):
                    self.assertTrue(module.is_supported_shape(source, target))

    def test_shape_and_memory_safety_limits_are_retained(self):
        valid = (42, 56)
        for source, target, patch_size in (
            ((0, 43), valid, 14),
            ((-1, 43), valid, 14),
            ((65537, 43), valid, 14),
            ((31.0, 43), valid, 14),
            ((True, 43), valid, 14),
            ((31,), valid, 14),
            ((31, 43), (43, 56), 14),
            ((31, 43), valid, 0),
            ((31, 43), valid, 65),
            ((31, 43), valid, True),
            ((65536, 65536), (65536, 65536), 1),
            ((65536, 65536), (14, 14), 14),
            ((1, 65536), (14, 14), 14),
        ):
            with self.subTest(source=source, target=target, patch_size=patch_size):
                self.assertFalse(
                    vision_preprocess_triton.is_supported_shape(
                        source, target, patch_size=patch_size
                    )
                )


if __name__ == "__main__":
    unittest.main()
