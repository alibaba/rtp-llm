"""fused_ba_gdn_gating vs bf16 F.linear (in_proj_ba) followed by fused_gdn_gating."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.gdn_ba_gating import fused_ba_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating


class FusedBaGdnGatingTest(unittest.TestCase):
    def test_matches_linear_then_gating(self):
        # Qwen3.5-27B, TP2: hidden 5120, 48 / 2 = 24 local value heads.
        hidden, heads = 5120, 24
        torch.manual_seed(0)
        w_ba = (torch.randn(2 * heads, hidden, device="cuda") * 0.02).bfloat16()
        A_log = torch.randn(heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(heads, device="cuda").bfloat16()
        for tokens in (1, 4):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, hidden, device="cuda").bfloat16()
                g, beta = fused_ba_gdn_gating(x, w_ba, A_log, dt_bias, heads)

                ba = F.linear(x, w_ba)
                g_ref, beta_ref = fused_gdn_gating(
                    A_log, ba[:, heads:], ba[:, :heads], dt_bias
                )
                self.assertEqual(g.dtype, torch.float32)
                self.assertEqual(beta.dtype, torch.bfloat16)
                # b / a are rounded to bf16 like the linear output; only the K
                # summation order differs, so allow about one bf16 ulp.
                torch.testing.assert_close(
                    g.view(tokens, heads),
                    g_ref.view(tokens, heads),
                    rtol=1e-2,
                    atol=1e-3,
                )
                torch.testing.assert_close(
                    beta.view(tokens, heads).float(),
                    beta_ref.view(tokens, heads).float(),
                    rtol=0,
                    atol=8e-3,
                )


if __name__ == "__main__":
    unittest.main()
