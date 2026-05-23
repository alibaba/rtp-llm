"""Test that CudaFp8DeepGEMMLinear.forward(input_fp8, input_scales=...) gives
the same numerical result as forward(input_bf16) when (input_fp8, input_scales)
were obtained from the same `sgl_per_token_group_quant_fp8` quantizer.

This validates the new external-scale path required by PR-C/PR-D's fused
norm/silu+quant kernels.
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.test.utils.numeric_util import calc_diff, per_block_cast_to_fp8


class TestFp8DeepGEMMLinearExternalScale(unittest.TestCase):
    K = 2048
    N = 4096

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)
        cls.device = "cuda"

        # Build weights — go through UE8M0 requant only when this build/HW
        # uses it; otherwise the linear treats scales as float32 per-block.
        weight_bf16 = torch.randn(
            (cls.N, cls.K), dtype=torch.bfloat16, device=cls.device
        )
        w_fp8, w_scales = per_block_cast_to_fp8(weight_bf16, use_ue8m0=False)
        cls.use_ue8m0 = is_deep_gemm_e8m0_used()
        if cls.use_ue8m0:
            w_fp8, w_scales = requant_weight_ue8m0(w_fp8, w_scales)
        else:
            # Match the (K, N) layout that the non-UE8M0 path expects
            scale_K = (cls.K + 127) // 128
            scale_N = (cls.N + 127) // 128
            w_fp8 = w_fp8.reshape(cls.K, cls.N)
            w_scales = w_scales.reshape(scale_K, scale_N)
        cls.weight_bf16 = weight_bf16
        cls.linear = CudaFp8DeepGEMMLinear(w_fp8, w_scales)

    def _run(self, M: int):
        x = torch.randn(M, self.K, dtype=torch.bfloat16, device=self.device)
        # Path 1: linear quantizes internally
        out_internal = self.linear(x.clone())
        # Path 2: external quant (same quantizer), pass (fp8, scale) to linear
        x_fp8, x_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=self.use_ue8m0,
        )
        out_external = self.linear(x_fp8, input_scales=x_scale)
        # Both forwards should produce identical output (same fp8 values, same scale).
        # Allow a tiny numerical slack for non-deterministic deepgemm scheduling
        # but expect bitwise equality on most elements.
        diff = calc_diff(out_external, out_internal)
        self.assertLess(
            diff,
            1e-5,
            f"M={M}: external-scale forward diverges from internal-quant forward "
            f"(diff={diff:.6e})",
        )

    def test_grid(self):
        for M in (1, 8, 32, 256, 1024):
            with self.subTest(M=M):
                self._run(M)


if __name__ == "__main__":
    unittest.main(verbosity=2)
