"""PPU JIT must use the selected device without probing nvidia-smi."""

import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from rtp_llm.platforms.ppu.kernels.cuda import sglang_jit


class SglangJitTest(unittest.TestCase):
    def test_architecture_and_environment_restoration(self):
        for previous in (None, "9.0"):
            for fail in (False, True):
                with self.subTest(previous=previous, fail=fail):
                    ffi = ModuleType("tvm_ffi")
                    cpp = ModuleType("tvm_ffi.cpp")
                    result = object()

                    def build(*args, **kwargs):
                        self.assertEqual(os.environ["TVM_FFI_CUDA_ARCH_LIST"], "8.9")
                        if fail:
                            raise RuntimeError("compile failed")
                        return result

                    cpp.load_inline = build
                    ffi.cpp = cpp
                    sglang_jit.load_sglang_kernel.cache_clear()
                    with patch.dict(sys.modules, {"tvm_ffi": ffi, "tvm_ffi.cpp": cpp}), patch.dict(os.environ):
                        if previous is None:
                            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
                        else:
                            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous
                        args = ("test", "test.cuh", "test", (8, 9))
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                                sglang_jit.load_sglang_kernel(*args)
                        else:
                            self.assertIs(sglang_jit.load_sglang_kernel(*args), result)
                        self.assertEqual(os.environ.get("TVM_FFI_CUDA_ARCH_LIST"), previous)
        sglang_jit.load_sglang_kernel.cache_clear()


if __name__ == "__main__":
    unittest.main()
