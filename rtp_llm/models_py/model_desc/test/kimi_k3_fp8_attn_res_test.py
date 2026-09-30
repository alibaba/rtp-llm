import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual


class KimiK3Fp8AttnResTest(unittest.TestCase):
    @staticmethod
    def dequantize(values, scale_wire):
        m, k = values.shape
        packed = scale_wire.T[:m].contiguous()
        exponents = packed.view(torch.uint8).reshape(m, k // 128).float()
        scales = torch.exp2(exponents - 127)
        return (values.float().reshape(m, k // 128, 128) * scales[:, :, None]).reshape(m, k)

    def test_fused_producer_matches_bf16_residual_and_preserves_anchor(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        torch.manual_seed(311)
        device = "cuda:0"
        m, k = 64, 7168
        norm = torch.randn(k, device=device, dtype=torch.bfloat16) * 0.02
        projection = torch.randn(k, device=device, dtype=torch.bfloat16) * 0.02
        output_norm = torch.ones(k, device=device, dtype=torch.bfloat16)
        residual = KimiK3AttentionResidual(norm, projection, 1e-6)

        for active_blocks, write_idx in ((0, 0), (1, -1)):
            with self.subTest(active_blocks=active_blocks):
                hidden = torch.randn((m, k), device=device, dtype=torch.bfloat16)
                anchors = torch.randn((m, 2, k), device=device, dtype=torch.bfloat16)
                reference_anchors = anchors.clone()
                reference = residual(
                    hidden.clone(), reference_anchors,
                    output_norm_weight=output_norm,
                    output_norm_eps=1e-6,
                    num_blocks=active_blocks,
                    block_write_idx=write_idx,
                )
                expected_values, expected_scales = sgl_per_token_group_quant_fp8(
                    reference,
                    group_size=128,
                    eps=1e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                actual = residual.forward_fp8(
                    hidden.clone(), anchors,
                    output_norm_weight=output_norm,
                    output_norm_eps=1e-6,
                    num_blocks=active_blocks,
                    block_write_idx=write_idx,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(anchors, reference_anchors, rtol=0, atol=0)
                actual_dequant = self.dequantize(actual.values, actual.scale_wire)
                expected_wire = expected_scales.as_strided(
                    ((k + 511) // 512, (m + 3) // 4 * 4),
                    ((m + 3) // 4 * 4, 1),
                )
                expected_dequant = self.dequantize(expected_values, expected_wire)
                diff = (actual_dequant - expected_dequant).abs()
                print(
                    f"active_blocks={active_blocks} max_abs={diff.max().item()} "
                    f"mean_abs={diff.mean().item()} "
                    f"scale_mismatches={(actual.scale_wire != expected_wire).sum().item()}",
                    flush=True,
                )
                # The native BF16 AttnRes and Triton producer reduce the
                # multi-block softmax in different orders near FP8 boundaries.
                self.assertLess(diff.max().item(), 0.3)
                self.assertLess(diff.mean().item(), 0.002)
                self.assertLess((diff > 0.1).float().mean().item(), 0.01)


if __name__ == "__main__":
    unittest.main()
