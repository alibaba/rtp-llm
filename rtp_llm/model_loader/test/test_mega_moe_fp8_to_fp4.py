"""Unit test for OnlineMegaMoEFp8ToFp4Weight conversion.

Tests that _convert_fp8_moe_to_fp4 produces the same result as the
reference implementation in GLM5MegaMoE.setup_weights_from_fp8.

Can run on any GPU with deep_gemm installed (no SM100/distributed required).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _check_prerequisites():
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        from deep_gemm.utils import per_token_cast_to_fp4

        return True, ""
    except ImportError:
        return False, "deep_gemm not available"


CAN_RUN, SKIP_REASON = _check_prerequisites()

FP4_BLOCK = 32
FP8_BLOCK = 128


def _reference_fp8_to_fp4(w1_fp8, w1_scale, w2_fp8, w2_scale, w3_fp8, w3_scale):
    """Reference: matches GLM5MegaMoE.setup_weights_from_fp8 logic."""
    from deep_gemm.utils import per_token_cast_to_fp4

    fp8_block = 128

    def _dequant_fp8(w_fp8, w_scale):
        E_, N_, K_ = w_fp8.shape
        n_blocks_k = K_ // fp8_block
        w_f = w_fp8.float().view(E_, N_, n_blocks_k, fp8_block)
        s_exp = w_scale.view(E_, N_, n_blocks_k, 1).expand_as(w_f)
        return (w_f * s_exp).reshape(E_, N_, K_).to(torch.bfloat16)

    w1_bf16 = _dequant_fp8(w1_fp8, w1_scale)
    w3_bf16 = _dequant_fp8(w3_fp8, w3_scale)
    w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=1)
    del w1_bf16, w3_bf16

    E = w13_bf16.shape[0]
    D = w13_bf16.shape[2]
    inter_2 = w13_bf16.shape[1]
    device = w13_bf16.device

    w13_packed = torch.empty((E, inter_2, D // 2), dtype=torch.int8, device=device)
    s13_raw = torch.empty(
        (E, inter_2, D // FP4_BLOCK), dtype=torch.float, device=device
    )
    for i in range(E):
        w13_packed[i], s13_raw[i] = per_token_cast_to_fp4(
            w13_bf16[i], use_ue8m0=True, gran_k=FP4_BLOCK
        )
    del w13_bf16

    w2_bf16 = _dequant_fp8(w2_fp8, w2_scale)
    inter = w2_bf16.shape[2]
    w2_packed = torch.empty(
        (E, w2_bf16.shape[1], inter // 2), dtype=torch.int8, device=device
    )
    s2_raw = torch.empty(
        (E, w2_bf16.shape[1], inter // FP4_BLOCK), dtype=torch.float, device=device
    )
    for i in range(E):
        w2_packed[i], s2_raw[i] = per_token_cast_to_fp4(
            w2_bf16[i], use_ue8m0=True, gran_k=FP4_BLOCK
        )
    del w2_bf16

    return w13_packed, s13_raw, w2_packed, s2_raw


def _generate_fp8_per_block_weight(E, N, K, device="cuda"):
    """Generate synthetic FP8 per-block quantized weight."""
    w_bf16 = torch.randn((E, N, K), dtype=torch.bfloat16, device=device)
    fp8_block = 128
    n_blocks = K // fp8_block
    w_view = w_bf16.float().view(E, N, n_blocks, fp8_block)
    amax = w_view.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / 448.0
    w_scaled = (w_view / scale).clamp(-448, 448)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn).view(E, N, K).contiguous()
    scale_out = scale.squeeze(-1).to(torch.float32).contiguous()
    return w_fp8, scale_out


@unittest.skipUnless(CAN_RUN, SKIP_REASON)
class TestFp8ToFp4Conversion(unittest.TestCase):
    """Test that load-time FP8→FP4 matches the reference implementation."""

    def test_small_conversion_matches_reference(self):
        """Test with small dimensions: E=4, inter=128, dim=256."""
        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            convert_fp8_moe_to_fp4_ue8m0 as _convert_fp8_moe_to_fp4,
        )

        E, inter, D = 4, 128, 256
        device = "cuda"

        # Generate FP8 weights matching GLM-5 layout:
        # w1 (gate): [E, inter, D], w3 (up): [E, inter, D], w2 (down): [E, D, inter]
        w1_fp8, w1_scale = _generate_fp8_per_block_weight(E, inter, D, device)
        w3_fp8, w3_scale = _generate_fp8_per_block_weight(E, inter, D, device)
        w2_fp8, w2_scale = _generate_fp8_per_block_weight(E, D, inter, device)

        # Reference: separate gate/up → concat → FP4
        ref_w13_packed, ref_s13, ref_w2_packed, ref_s2 = _reference_fp8_to_fp4(
            w1_fp8, w1_scale, w2_fp8, w2_scale, w3_fp8, w3_scale
        )

        # New method: stacked w1 = [gate || up] already concatenated, as in the
        # actual model loader (stack_moe_w1 produces [E, 2*inter, D])
        w13_fp8 = torch.cat([w1_fp8, w3_fp8], dim=1).contiguous()
        s13 = torch.cat([w1_scale, w3_scale], dim=1).contiguous()

        # Load-time conversion
        new_w13_packed, new_s13 = _convert_fp8_moe_to_fp4(w13_fp8, s13)
        new_w2_packed, new_s2 = _convert_fp8_moe_to_fp4(w2_fp8, w2_scale)

        # Verify shapes
        self.assertEqual(new_w13_packed.shape, ref_w13_packed.shape)
        self.assertEqual(new_s13.shape, ref_s13.shape)
        self.assertEqual(new_w2_packed.shape, ref_w2_packed.shape)
        self.assertEqual(new_s2.shape, ref_s2.shape)

        # Verify exact match (deterministic quantization)
        self.assertTrue(
            torch.equal(new_w13_packed, ref_w13_packed),
            "w13 packed weights don't match reference",
        )
        self.assertTrue(
            torch.allclose(new_s13, ref_s13),
            "w13 scales don't match reference",
        )
        self.assertTrue(
            torch.equal(new_w2_packed, ref_w2_packed),
            "w2 packed weights don't match reference",
        )
        self.assertTrue(
            torch.allclose(new_s2, ref_s2),
            "w2 scales don't match reference",
        )

    def test_glm5_shapes(self):
        """Test with GLM-5 actual shapes (scaled down experts): E=8, inter=2048, dim=6144."""
        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            convert_fp8_moe_to_fp4_ue8m0 as _convert_fp8_moe_to_fp4,
        )

        # Use smaller E to keep test fast, but real inter/dim
        E, inter, D = 2, 256, 384  # divisible by 128 and 32
        device = "cuda"

        w1_fp8, w1_scale = _generate_fp8_per_block_weight(E, inter, D, device)
        w3_fp8, w3_scale = _generate_fp8_per_block_weight(E, inter, D, device)
        w2_fp8, w2_scale = _generate_fp8_per_block_weight(E, D, inter, device)

        ref_w13_packed, ref_s13, ref_w2_packed, ref_s2 = _reference_fp8_to_fp4(
            w1_fp8, w1_scale, w2_fp8, w2_scale, w3_fp8, w3_scale
        )

        w13_fp8 = torch.cat([w1_fp8, w3_fp8], dim=1).contiguous()
        s13 = torch.cat([w1_scale, w3_scale], dim=1).contiguous()

        new_w13_packed, new_s13 = _convert_fp8_moe_to_fp4(w13_fp8, s13)
        new_w2_packed, new_s2 = _convert_fp8_moe_to_fp4(w2_fp8, w2_scale)

        self.assertTrue(torch.equal(new_w13_packed, ref_w13_packed))
        self.assertTrue(torch.allclose(new_s13, ref_s13))
        self.assertTrue(torch.equal(new_w2_packed, ref_w2_packed))
        self.assertTrue(torch.allclose(new_s2, ref_s2))

    def test_output_dtypes(self):
        """Verify output dtypes are correct for downstream consumption."""
        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            convert_fp8_moe_to_fp4_ue8m0 as _convert_fp8_moe_to_fp4,
        )

        E, N, K = 2, 128, 256
        w_fp8, w_scale = _generate_fp8_per_block_weight(E, N, K, "cuda")
        packed, sf = _convert_fp8_moe_to_fp4(w_fp8, w_scale)

        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(sf.dtype, torch.float32)
        self.assertEqual(packed.shape, (E, N, K // 2))
        self.assertEqual(sf.shape, (E, N, K // 32))


if __name__ == "__main__":
    unittest.main()
