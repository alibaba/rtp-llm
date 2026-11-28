"""CPU-only support-gate contracts; no CUDA or numerical tolerance changes."""

import importlib.util
import sys
import unittest
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41 import vision_preprocess_triton


class PreprocessShapeTest(unittest.TestCase):
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
