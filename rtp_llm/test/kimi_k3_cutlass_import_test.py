import os
import subprocess
import sys
import unittest


class KimiK3CutlassImportTest(unittest.TestCase):
    def test_cutlass_is_importable_before_runtime_path_setup(self):
        environment = os.environ.copy()
        environment.pop("KIMI_K3_FLASHINFER_ROOT", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.metadata, cutlass, cutlass.cute; "
                "assert cutlass.__version__ == "
                "importlib.metadata.version('nvidia-cutlass-dsl')",
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cuda_factories_use_the_declared_cutlass_distribution(self):
        environment = os.environ.copy()
        environment.pop("KIMI_K3_FLASHINFER_ROOT", None)
        factories = (
            "rtp_llm.models_py.modules.factory.attention",
            "rtp_llm.models_py.modules.factory.fused_moe",
        )
        fp8 = "rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel"
        for order in (factories, factories[::-1], (fp8, *factories)):
            with self.subTest(order=order):
                code = "\n".join(
                    [
                        "import importlib, importlib.metadata",
                        *[f"importlib.import_module({name!r})" for name in order],
                        "import cutlass",
                        "assert cutlass.__version__ == importlib.metadata.version('nvidia-cutlass-dsl')",
                    ]
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
