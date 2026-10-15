"""Compare K3 fused RoPE against the independent complex-number definition."""

import unittest

import torch

from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
    maybe_fused_apply_rope,
)


def _reference_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = pairs * freqs[:, None, :]
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "K3 fused RoPE requires CUDA")
class KimiK3RopeTest(unittest.TestCase):
    def test_fused_qk_matches_complex_rotation(self) -> None:
        torch.manual_seed(11)
        for length, heads, width in ((1, 1, 8), (64, 12, 128), (257, 5, 64)):
            with self.subTest(shape=(length, heads, width)):
                q = torch.randn(length, heads, width, device="cuda", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                angles = torch.randn(length, width // 2, device="cuda")
                freqs = torch.polar(torch.ones_like(angles), angles)
                fused = maybe_fused_apply_rope(q, k, freqs)
                self.assertIsNotNone(fused)
                q_out, k_out = fused
                torch.testing.assert_close(
                    q_out, _reference_rope(q, freqs), rtol=0.02, atol=0.02
                )
                torch.testing.assert_close(
                    k_out, _reference_rope(k, freqs), rtol=0.02, atol=0.02
                )

    def test_unsupported_cpu_input_falls_back(self) -> None:
        q = torch.zeros(2, 1, 8)
        freqs = torch.ones(2, 4, dtype=torch.complex64)
        self.assertIsNone(maybe_fused_apply_rope(q, q, freqs))


if __name__ == "__main__":
    unittest.main()
