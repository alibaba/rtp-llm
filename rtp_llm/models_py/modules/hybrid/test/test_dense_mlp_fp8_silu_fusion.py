"""End-to-end test for DenseMLP's fused silu_and_mul + fp8 quant path.

Compares the fused-on `DenseMLP` against the same `DenseMLP` with the fusion
forced off. Both use CudaFp8DeepGEMMLinear for down_proj; the only difference
is whether silu_and_mul + per-token-group quant is one fused kernel or two.

Run:
    python -m pytest rtp_llm/models_py/modules/hybrid/test/test_dense_mlp_fp8_silu_fusion.py -v -s
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.test.utils.numeric_util import calc_diff, per_block_cast_to_fp8


def _build_fp8_linear(N: int, K: int, device: str) -> CudaFp8DeepGEMMLinear:
    """Build a CudaFp8DeepGEMMLinear with random weights."""
    bf16_w = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    w_fp8, w_scales = per_block_cast_to_fp8(bf16_w, use_ue8m0=False)
    if is_deep_gemm_e8m0_used():
        w_fp8, w_scales = requant_weight_ue8m0(w_fp8, w_scales)
    else:
        scale_K = (K + 127) // 128
        scale_N = (N + 127) // 128
        w_fp8 = w_fp8.reshape(K, N)
        w_scales = w_scales.reshape(scale_K, scale_N)
    return CudaFp8DeepGEMMLinear(w_fp8, w_scales)


class TestDenseMLPFp8SiluFusion(unittest.TestCase):
    HIDDEN = 1024
    INTER = 1024

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if not has_deep_gemm():
            raise unittest.SkipTest("DeepGEMM not available")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        cls.device = "cuda"

    def _build_mlp_pair(self):
        """Build two DenseMLP instances sharing the same weights:
        one with fusion enabled, one with fusion forced off."""
        up_linear = _build_fp8_linear(
            N=self.INTER * 2, K=self.HIDDEN, device=self.device
        )
        down_linear = _build_fp8_linear(N=self.HIDDEN, K=self.INTER, device=self.device)

        from rtp_llm.ops import ActivationType

        # Build fused MLP by manually assembling parts
        mlp_fused = DenseMLP.__new__(DenseMLP)
        torch.nn.Module.__init__(mlp_fused)
        mlp_fused.activation_type = ActivationType.Swiglu
        mlp_fused.parallelism_config = type(
            "P", (), {"get_ffn_tp_size": lambda self: 1}
        )()
        mlp_fused.act_fn = FusedSiluAndMul()
        mlp_fused.is_gated = True
        mlp_fused.up_proj = up_linear
        mlp_fused.down_proj = down_linear
        mlp_fused._fuse_silu_quant = down_linear.K % 128 == 0
        if mlp_fused._fuse_silu_quant and down_linear.scale_ue8m0:
            mlp_fused._fuse_silu_quant = down_linear.K % 512 == 0

        # Build unfused MLP (same weights, fusion disabled)
        mlp_unfused = DenseMLP.__new__(DenseMLP)
        torch.nn.Module.__init__(mlp_unfused)
        mlp_unfused.activation_type = ActivationType.Swiglu
        mlp_unfused.parallelism_config = mlp_fused.parallelism_config
        mlp_unfused.act_fn = FusedSiluAndMul()
        mlp_unfused.is_gated = True
        mlp_unfused.up_proj = up_linear
        mlp_unfused.down_proj = down_linear
        mlp_unfused._fuse_silu_quant = False

        return mlp_fused, mlp_unfused

    def test_fused_matches_unfused(self):
        mlp_fused, mlp_unfused = self._build_mlp_pair()

        if not mlp_fused._fuse_silu_quant:
            self.skipTest(
                "DenseMLP._fuse_silu_quant did not activate "
                "(K not aligned or build doesn't support it)"
            )

        for T in (1, 8, 32, 256):
            with self.subTest(T=T):
                x = torch.randn(
                    T, self.HIDDEN, dtype=torch.bfloat16, device=self.device
                )
                out_fused = mlp_fused(x.clone())
                out_unfused = mlp_unfused(x.clone())
                diff = calc_diff(out_fused, out_unfused)
                self.assertLess(
                    diff,
                    5e-3,
                    f"T={T}: fused vs unfused DenseMLP diff = {diff:.6e}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
