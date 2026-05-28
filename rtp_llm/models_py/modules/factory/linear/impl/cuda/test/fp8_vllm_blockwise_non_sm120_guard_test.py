"""Cross-architecture launch guard for the SM120 CUTLASS binding.

This target is intentionally built only by the cuda12_9 x86 H20 job.  A
missing binding is therefore a build-contract failure, not a reason to skip.
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.ops.compute_ops import cutlass_scaled_mm_blockwise_sm120_fp8
from rtp_llm.test.utils.numeric_util import per_block_cast_to_fp8


@unittest.skipUnless(
    torch.cuda.is_available() and not is_sm12x(),
    "Non-SM120 direct binding guard requires a non-sm12x CUDA device",
)
class CudaFp8VllmBlockwiseNonSM120GuardTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.M = 8
        self.K = 128
        self.N = 128

    def _make_op_inputs(self):
        input_tensor = torch.randn(
            self.M, self.K, dtype=torch.bfloat16, device="cuda"
        ).contiguous()
        weight_bf16 = (
            torch.randn((self.N, self.K), dtype=torch.bfloat16, device="cuda") * 0.1
        ).contiguous()
        A, A_sf = sgl_per_token_group_quant_fp8(
            input_tensor,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=False,
            scale_ue8m0=False,
        )
        B, B_sf = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)
        D = torch.empty(self.M, self.N, dtype=torch.bfloat16, device="cuda")
        return D, A, B, A_sf, B_sf

    def test_direct_binding_rejects_non_sm120_before_launch(self):
        D, A, B, A_sf, B_sf = self._make_op_inputs()
        with self.assertRaisesRegex(RuntimeError, "requires sm_120 family"):
            cutlass_scaled_mm_blockwise_sm120_fp8(D, A, B, A_sf, B_sf)


if __name__ == "__main__":
    unittest.main()
