import importlib
import importlib.metadata
import platform
import unittest


class ArmWheelImportTest(unittest.TestCase):
    @unittest.skipUnless(platform.machine() == "aarch64", "requires aarch64 wheels")
    def test_custom_kernel_packages_import_with_expected_versions(self):
        packages = {
            "deep-ep": ("deep_ep", "1.2.1.11+unknown.pai"),
            "deep-gemm": ("deep_gemm", "2.1.1+local"),
            "fast-safetensors": (
                "fast_safetensors",
                "0.7.3+torch2.1.2.cu121",
            ),
            "flash-attn": ("flash_attn", "2.8.3"),
            "flashinfer-python": ("flashinfer", "0.6.6"),
            "rtp-kernel": ("rtp_kernel", "0.1.0+125c29e5.20260422155252"),
        }
        for distribution, (module, expected_version) in packages.items():
            with self.subTest(distribution=distribution):
                importlib.import_module(module)
                self.assertEqual(
                    importlib.metadata.version(distribution).lower(),
                    expected_version.lower(),
                )


if __name__ == "__main__":
    unittest.main()
