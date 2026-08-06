"""Cross-architecture launch guard for the SM120 CUTLASS binding.

This target is intentionally built only by the cuda12_9 x86 H20 job. A
missing binding is therefore a BUILD select/local_defines contract failure,
not a reason to skip.
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.test.sm120_test_utils import (
    make_blockwise_op_inputs,
)
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.ops import compute_ops


@unittest.skipIf(is_sm120(), "Non-SM120 direct binding guard requires non-sm120")
class CudaFp8VllmBlockwiseNonSM120GuardTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("H20 guard target requires a working CUDA runtime")
        try:
            backend_compiled = compute_ops.has_cutlass_scaled_mm_blockwise_sm120_fp8()
        except AttributeError as error:
            raise RuntimeError("SM120 capability probe is missing") from error
        if not backend_compiled:
            raise RuntimeError(
                "SM120 backend is missing from the cuda12_9 x86 build; keep "
                "BUILD local_defines and dependency using_cuda12_9_x86 paired. "
                "If this target ran under another build config, check the "
                "cuda12_9 CI tag filter."
            )

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.M = 8
        self.K = 128
        self.N = 128

    def _make_op_inputs(self):
        return make_blockwise_op_inputs(self.M, self.K, self.N)

    def test_direct_binding_rejects_non_sm120_before_launch(self):
        D, A, B, A_sf, B_sf = self._make_op_inputs()
        with self.assertRaisesRegex(RuntimeError, "was compiled for sm_120"):
            compute_ops.cutlass_scaled_mm_blockwise_sm120_fp8(D, A, B, A_sf, B_sf)


if __name__ == "__main__":
    unittest.main()
