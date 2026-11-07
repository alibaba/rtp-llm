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


class DeepGemmFp8DispatchTest(unittest.TestCase):
    def test_grouped_layouts_use_current_stream_only_for_affected_wheel(self):
        python_module = types.ModuleType("deep_gemm.jit_kernels.m_grouped_gemm_fp8")
        for layout in ("contiguous", "masked", "nopad"):
            symbol = "m_grouped_gemm_fp8_fp8_bf16_nt_" + layout
            exported, python_impl = lambda *args: None, lambda *args: None
            wheel = types.SimpleNamespace(**{symbol: exported})
            setattr(python_module, symbol, python_impl)
            for version in ("1.0.0+v0.4.0.ppu2.2.0", "1.0.0+v0.4.1.ppu2.2.0"):
                with self.subTest(layout=layout, version=version), patch.dict(
                    sys.modules, {python_module.__name__: python_module}
                ), patch.object(importlib.metadata, "version", return_value=version):
                    self.assertIs(
                        deepgemm_wrapper._resolve_fp8_grouped(wheel, layout),
                        python_impl if version == "1.0.0+v0.4.0.ppu2.2.0" else exported,
                    )

    def test_missing_metadata_and_symbol_preserve_exported_contract(self):
        exported = lambda *args: None
        wheel = types.SimpleNamespace(m_grouped_gemm_fp8_fp8_bf16_nt_nopad=exported)
        with patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            self.assertIs(deepgemm_wrapper._resolve_fp8_grouped(wheel, "nopad"), exported)
        self.assertIsNone(deepgemm_wrapper._resolve_fp8_grouped(wheel, "masked"))


if __name__ == "__main__":
    unittest.main()
