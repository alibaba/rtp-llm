import builtins
import importlib
import unittest
from unittest import mock

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import quantized_linear


class QuantizedLinearImportTest(unittest.TestCase):
    def test_module_import_does_not_require_deep_gemm(self):
        original_import = builtins.__import__

        def reject_deep_gemm(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "deep_gemm" or name.startswith("deep_gemm."):
                raise ModuleNotFoundError("deep_gemm intentionally unavailable")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=reject_deep_gemm):
            importlib.reload(quantized_linear)


if __name__ == "__main__":
    unittest.main()
