import subprocess
import sys
import unittest


class QuantizedLinearImportTest(unittest.TestCase):
    def test_module_import_does_not_require_deep_gemm(self):
        code = """
import sys

class DeepGemmImportBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "deep_gemm" or fullname.startswith("deep_gemm."):
            raise ImportError("deep_gemm intentionally unavailable")
        return None

sys.meta_path.insert(0, DeepGemmImportBlocker())
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import quantized_linear
assert quantized_linear is not None
assert not any(
    name == "deep_gemm" or name.startswith("deep_gemm.") for name in sys.modules
)
"""
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
