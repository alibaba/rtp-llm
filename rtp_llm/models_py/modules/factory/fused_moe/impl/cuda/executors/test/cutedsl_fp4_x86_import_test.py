# SPDX-License-Identifier: Apache-2.0
"""x86 CUDA 13 import-level guard for the CuteDSL FP4 path.

The CuteDSL FP4 *numerical* regression runs only on the ARM SM100 pool
(see the sibling BUILD). This test gives x86 CUDA 13 a hardware-free guard:
it verifies the FP4 export/import path resolves to a well-defined state
(either the real CuteDSL impl imports cleanly, or it degrades to the
_fp4_unavailable failure stub) and never leaves a bare None / NameError or an
ABI-level import crash. It needs no SM100 device.

It also logs which branch the current x86 build takes, so the observed
qualification state is visible in the test output.
"""

import unittest


_FP4_EXPORTS = [
    "flashinfer_cutedsl_moe_masked",
    "scaled_fp4_grouped_quant",
    "silu_and_mul_scaled_fp4_grouped_quant",
    "cutlass_scaled_fp4_mm_wrapper",
    "scaled_fp4_quant_wrapper",
]


class CutedslFp4X86ImportTest(unittest.TestCase):
    def test_fp4_kernel_exports_resolve(self):
        from rtp_llm.models_py.kernels.cuda import fp4_kernel

        for name in _FP4_EXPORTS:
            obj = getattr(fp4_kernel, name)
            branch = (
                "unavailable-stub"
                if getattr(obj, "__name__", "") == "_fp4_unavailable"
                else "real-impl"
            )
            print(f"FP4_X86_PROBE export={name} branch={branch} obj={obj!r}")
            # On a CUDA build these are callable in both branches (real impl or
            # the _fp4_unavailable stub). None is the non-CUDA contract and must
            # not appear here; a missing attribute would be a NameError-class bug.
            self.assertIsNotNone(obj, f"{name} resolved to None on a CUDA build")
            self.assertTrue(callable(obj), f"{name} is not callable: {obj!r}")

    def test_cutedsl_executor_imports(self):
        # Importing the executor must not ABI-crash. Reaching an unavailable
        # kernel is fine — that surfaces as a clean RuntimeError at call time,
        # not at import time.
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )

        print(f"FP4_X86_PROBE executor_import=ok cls={CutedslFp4Executor!r}")
        self.assertTrue(hasattr(CutedslFp4Executor, "executor_type"))


if __name__ == "__main__":
    unittest.main()
