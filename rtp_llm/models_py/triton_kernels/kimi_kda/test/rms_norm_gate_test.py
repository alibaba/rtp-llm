import unittest

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda.rms_norm_gate import (
    kimi_kda_rms_norm_sigmoid_gate,
)


class KimiKDARmsNormGateTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(20260806)

    def test_accepts_qkvg_fa_beta_leading_stride(self) -> None:
        token_count = 257
        local_heads = 12
        head_dim = 128
        projection_size = local_heads * head_dim
        fused_width = 4 * projection_size + 128 + 96
        gate_begin = 3 * projection_size

        projected = torch.randn(
            1,
            token_count,
            fused_width,
            dtype=torch.bfloat16,
            device="cuda",
        )
        gate = projected[:, :, gate_begin : gate_begin + projection_size].reshape(
            1, token_count, local_heads, head_dim
        )
        self.assertEqual(
            gate.stride(),
            (token_count * fused_width, fused_width, head_dim, 1),
        )
        self.assertEqual(
            gate.untyped_storage().data_ptr(), projected.untyped_storage().data_ptr()
        )

        x = torch.randn_like(gate.contiguous())
        weight = torch.randn(head_dim, dtype=torch.float32, device="cuda")
        actual = kimi_kda_rms_norm_sigmoid_gate(x, gate, weight, 1e-6)
        expected = kimi_kda_rms_norm_sigmoid_gate(
            x,
            gate.contiguous(),
            weight,
            1e-6,
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
