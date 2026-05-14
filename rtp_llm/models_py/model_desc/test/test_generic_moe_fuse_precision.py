"""Precision verification for GenericMoeDecoderLayer fused norm paths.

Verifies that the fused paths produce equivalent results to the unfused paths:
1. Input norm fusion: fused_add_rmsnorm_fp8_quant vs (residual_add + rmsnorm + quant)
2. Post norm fusion (Dense): same as above for DenseMLP path
3. Post norm fusion (MoE dual): fused dual-output vs (residual_add + rmsnorm + quant + bf16)

Run:
    PYTHONPATH=.:bazel-bin /opt/conda310/bin/python3 -m pytest \
        rtp_llm/models_py/model_desc/test/test_generic_moe_fuse_precision.py -v -s
"""

import unittest

import flashinfer.norm
import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    fused_add_rmsnorm_fp8_quant,
    fused_add_rmsnorm_fp8_quant_with_bf16_output,
)


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _dequantize_fp8(
    fp8: torch.Tensor, scale: torch.Tensor, group_size: int = 128
) -> torch.Tensor:
    T, H = fp8.shape
    n_groups = H // group_size
    return (
        fp8.float().view(T, n_groups, group_size) * scale.float().unsqueeze(-1)
    ).view(T, H)


class TestGenericMoeFusePrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _run_input_norm_fuse(self, T: int, H: int):
        """Simulate fused input_norm path vs unfused."""
        hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(H, dtype=torch.bfloat16, device="cuda")
        eps = 1e-6

        # Unfused path: residual += hidden_states; normed = rmsnorm(residual); quant
        res_unfused = residual.clone()
        res_unfused.add_(hidden_states)
        normed_unfused = flashinfer.norm.rmsnorm(res_unfused, weight, eps=eps)
        fp8_unfused, scale_unfused = sgl_per_token_group_quant_fp8(
            normed_unfused.contiguous(),
            group_size=128,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

        # Fused path
        res_fused = residual.clone()
        fp8_fused, scale_fused = fused_add_rmsnorm_fp8_quant(
            hidden_states.clone(),
            res_fused,
            weight,
            eps,
            group_size=128,
            scale_ue8m0=False,
        )

        # Compare dequantized outputs
        deq_unfused = _dequantize_fp8(fp8_unfused, scale_unfused)
        deq_fused = _dequantize_fp8(fp8_fused, scale_fused)

        rel_err = (
            ((deq_fused - deq_unfused).abs() / (deq_unfused.abs() + 1e-6)).mean().item()
        )

        # Also verify residual is updated identically
        res_diff = (res_fused - res_unfused).abs().max().item()

        return rel_err, res_diff

    def _run_post_norm_moe_dual_fuse(self, T: int, H: int):
        """Simulate fused post_norm MoE dual-output path vs unfused."""
        hidden_states = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(H, dtype=torch.bfloat16, device="cuda")
        eps = 1e-6

        # Unfused path: residual += hidden_states; normed = rmsnorm(residual)
        # bf16_out = normed; fp8_out, scale = quant(normed)
        res_unfused = residual.clone()
        res_unfused.add_(hidden_states)
        normed_unfused = flashinfer.norm.rmsnorm(res_unfused, weight, eps=eps)
        bf16_unfused = normed_unfused.clone()
        fp8_unfused, scale_unfused = sgl_per_token_group_quant_fp8(
            normed_unfused.contiguous(),
            group_size=128,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )

        # Fused path
        res_fused = residual.clone()
        bf16_fused, fp8_fused, scale_fused = (
            fused_add_rmsnorm_fp8_quant_with_bf16_output(
                hidden_states.clone(),
                res_fused,
                weight,
                eps,
                group_size=128,
                scale_ue8m0=False,
            )
        )

        # Compare bf16 outputs
        bf16_max_diff = (bf16_fused - bf16_unfused).abs().max().item()

        # Compare dequantized fp8 outputs
        deq_unfused = _dequantize_fp8(fp8_unfused, scale_unfused)
        deq_fused = _dequantize_fp8(fp8_fused, scale_fused)
        fp8_rel_err = (
            ((deq_fused - deq_unfused).abs() / (deq_unfused.abs() + 1e-6)).mean().item()
        )

        # Verify residual
        res_diff = (res_fused - res_unfused).abs().max().item()

        return bf16_max_diff, fp8_rel_err, res_diff

    def test_input_norm_fuse_precision(self):
        """F3/4: input_layernorm + fp8_quant for CausalAttention qkv_proj."""
        cases = [
            (1, 2048),
            (1, 4096),
            (4, 4096),
            (16, 4096),
            (32, 4096),
            (128, 4096),
            (1024, 4096),
        ]
        for T, H in cases:
            with self.subTest(T=T, H=H):
                rel_err, res_diff = self._run_input_norm_fuse(T, H)
                self.assertLess(
                    rel_err,
                    0.05,
                    f"T={T} H={H}: fp8 mean_rel_err={rel_err:.4e} exceeds 5%",
                )
                self.assertAlmostEqual(
                    res_diff,
                    0.0,
                    places=5,
                    msg=f"T={T} H={H}: residual mismatch={res_diff:.4e}",
                )
                print(
                    f"  input_norm_fuse T={T:5d} H={H:5d}  rel_err={rel_err:.3e}  res_diff={res_diff:.3e}  PASS"
                )

    def test_post_norm_dense_fuse_precision(self):
        """F3: post_attention_layernorm + fp8_quant for DenseMLP."""
        cases = [
            (1, 2048),
            (1, 4096),
            (8, 4096),
            (32, 4096),
            (256, 4096),
        ]
        for T, H in cases:
            with self.subTest(T=T, H=H):
                rel_err, res_diff = self._run_input_norm_fuse(T, H)
                self.assertLess(rel_err, 0.05)
                self.assertAlmostEqual(res_diff, 0.0, places=5)
                print(
                    f"  post_norm_dense T={T:5d} H={H:5d}  rel_err={rel_err:.3e}  PASS"
                )

    def test_post_norm_moe_dual_fuse_precision(self):
        """F7: post_attention_layernorm + dual output (bf16+fp8) for MoE."""
        cases = [
            (1, 2048),
            (1, 4096),
            (4, 4096),
            (16, 4096),
            (32, 4096),
            (128, 4096),
            (1024, 4096),
        ]
        for T, H in cases:
            with self.subTest(T=T, H=H):
                bf16_diff, fp8_rel, res_diff = self._run_post_norm_moe_dual_fuse(T, H)
                self.assertLess(
                    bf16_diff,
                    0.1,
                    f"T={T} H={H}: bf16 max_diff={bf16_diff:.4e}",
                )
                self.assertLess(
                    fp8_rel,
                    0.05,
                    f"T={T} H={H}: fp8 mean_rel_err={fp8_rel:.4e}",
                )
                self.assertAlmostEqual(res_diff, 0.0, places=5)
                print(
                    f"  post_norm_moe  T={T:5d} H={H:5d}  "
                    f"bf16_diff={bf16_diff:.3e}  fp8_rel={fp8_rel:.3e}  PASS"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
