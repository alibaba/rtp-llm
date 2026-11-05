import importlib.metadata
import sys
import types
import unittest
from unittest.mock import patch

from rtp_llm.models_py.kernels.ppu import deepgemm_wrapper


class DeepGemmMaskedDispatchTest(unittest.TestCase):
    def setUp(self):
        self.cpp_masked = lambda *args: None
        self.python_masked = lambda *args: None
        self.wheel = types.ModuleType("deep_gemm")
        self.wheel.m_grouped_gemm_bf16_bf16_bf16_nt_masked = self.cpp_masked
        legacy = types.ModuleType("deep_gemm.jit_kernels.m_grouped_gemm")
        legacy.m_grouped_gemm_bf16_bf16_bf16_nt_masked = self.python_masked
        self.modules = patch.dict(
            sys.modules, {"deep_gemm.jit_kernels.m_grouped_gemm": legacy}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_affected_wheel_uses_current_stream_implementation(self):
        with patch.object(
            importlib.metadata, "version", return_value="1.0.0+v0.4.0.ppu2.2.0"
        ):
            self.assertIs(
                deepgemm_wrapper._resolve_bf16_masked(self.wheel),
                self.python_masked,
            )

    def test_other_wheels_keep_their_exported_implementation(self):
        for version in (
            "1.0.0+ppu2.1.0.post1.dev000.g164e851bd",
            "1.0.0+v0.4.1.ppu2.2.0",
            "2.8.0",
        ):
            with self.subTest(version=version), patch.object(
                importlib.metadata, "version", return_value=version
            ):
                self.assertIs(
                    deepgemm_wrapper._resolve_bf16_masked(self.wheel), self.cpp_masked
                )

    def test_missing_distribution_metadata_keeps_exported_implementation(self):
        with patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            self.assertIs(
                deepgemm_wrapper._resolve_bf16_masked(self.wheel), self.cpp_masked
            )

    def test_modern_symbol_keeps_precedence(self):
        modern = lambda *args: None
        self.wheel.m_grouped_bf16_gemm_nt_masked = modern
        with patch.object(importlib.metadata, "version") as version:
            self.assertIs(deepgemm_wrapper._resolve_bf16_masked(self.wheel), modern)
            version.assert_not_called()

    def test_missing_symbol_stays_unavailable(self):
        del self.wheel.m_grouped_gemm_bf16_bf16_bf16_nt_masked
        with patch.object(importlib.metadata, "version") as version:
            self.assertIsNone(deepgemm_wrapper._resolve_bf16_masked(self.wheel))
            version.assert_not_called()


if __name__ == "__main__":
    unittest.main()
