"""Verify importing the attention package has no CUDA-only side effects on CPU."""

import subprocess
import sys
import textwrap
import unittest


class TestAttentionNonCudaImport(unittest.TestCase):
    def test_cpu_import_does_not_load_cuda_attention_modules(self) -> None:
        script = textwrap.dedent(
            """
            import importlib
            import importlib.abc
            import sys

            from rtp_llm.device import device_type


            class RejectCudaAttentionImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    is_cuda_attention = (
                        fullname.startswith(
                            "rtp_llm.models_py.modules.factory.attention.cuda_"
                        )
                        or ".cuda_impl" in fullname
                    )
                    if fullname == "rtp_kernel" or fullname.startswith("rtp_kernel."):
                        raise ImportError(f"unexpected CUDA dependency: {fullname}")
                    if is_cuda_attention:
                        raise ImportError(f"unexpected CUDA attention import: {fullname}")
                    return None


            device_type.get_device_type = lambda: device_type.DeviceType.Cpu
            # Importing the top-level rtp_llm package initializes its shared
            # extension surface before this test can patch device detection.
            # Remove any target-package entries loaded by that prelude, then
            # prove a cold attention-package import does not load CUDA code.
            attention_package = "rtp_llm.models_py.modules.factory.attention"
            for name in list(sys.modules):
                if (
                    name == attention_package
                    or name.startswith(f"{attention_package}.")
                    or name == "rtp_kernel"
                    or name.startswith("rtp_kernel.")
                ):
                    del sys.modules[name]
            sys.meta_path.insert(0, RejectCudaAttentionImports())
            importlib.import_module(attention_package)
            assert not any(
                name == "rtp_kernel" or name.startswith("rtp_kernel.")
                for name in sys.modules
            )
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_ppu_import_registers_flashinfer_attention_modules(self) -> None:
        script = textwrap.dedent(
            """
            import importlib
            import sys

            from rtp_llm.device import device_type


            device_type.get_device_type = lambda: device_type.DeviceType.Ppu
            attention_package = "rtp_llm.models_py.modules.factory.attention"
            for name in list(sys.modules):
                if name == attention_package or name.startswith(
                    f"{attention_package}."
                ):
                    del sys.modules[name]

            importlib.import_module(attention_package)
            from rtp_llm.models_py.modules.factory.attention.attn_factory import (
                DECODE_MHA_IMPS,
                PREFILL_MHA_IMPS,
            )

            prefill_impls = {impl.__name__ for impl in PREFILL_MHA_IMPS}
            decode_impls = {impl.__name__ for impl in DECODE_MHA_IMPS}
            assert {
                "PyFlashinferPrefillImpl",
                "PyFlashinferPagedPrefillImpl",
                "CPFlashInferImpl",
            }.issubset(prefill_impls), prefill_impls
            assert "PyFlashinferDecodeImpl" in decode_impls, decode_impls
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
