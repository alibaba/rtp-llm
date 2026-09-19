"""A reusable checkpoint must equal a separately computed prefix final state."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_cublas,
)


class CheckpointPrecisionTest(unittest.TestCase):
    def test_intermediate_checkpoint_preserves_recurrent_state(self):
        torch.manual_seed(20260918)
        heads, dim, tokens = 2, 128, 256
        k = torch.randn(1, tokens, heads, dim, device="cuda").bfloat16() * 0.03
        w = torch.randn_like(k) * 0.03
        u = torch.randn_like(k) * 0.1
        gk = -torch.rand(1, tokens, heads, dim, device="cuda") * 0.3
        initial = torch.randn(1, heads, dim, dim, device="cuda") * 0.05
        common = dict(
            initial_state=initial,
            output_final_state=True,
            chunk_size=64,
            intermediate_state_dtype=torch.float32,
        )
        h, _, _ = chunk_gated_delta_rule_fwd_h_cublas(k=k, w=w, u=u, gk=gk, **common)
        for boundary in (64, 128, 192):
            _, _, final = chunk_gated_delta_rule_fwd_h_cublas(
                k=k[:, :boundary],
                w=w[:, :boundary],
                u=u[:, :boundary],
                gk=gk[:, :boundary],
                **common
            )
            self.assertEqual(h.dtype, torch.float32)
            torch.testing.assert_close(h[:, boundary // 64], final, atol=0, rtol=0)
            self.assertFalse(torch.equal(final, final.bfloat16().float()))

    def test_chunk_prefill_resumes_from_intermediate_state(self):
        from rtp_llm.models_py.triton_kernels.kimi_kda.chunk import chunk_kda

        torch.manual_seed(20260919)
        tokens, heads, dim = 344, 2, 128
        q, k, v = [
            torch.randn(1, tokens, heads, dim, device="cuda").bfloat16()
            for _ in range(3)
        ]
        g = -torch.rand(1, tokens, heads, dim, device="cuda") * 0.03
        beta = torch.rand(1, tokens, heads, device="cuda")
        options = dict(
            output_final_state=True,
            return_intermediate_states=True,
            use_qk_l2norm_in_kernel=True,
            chunk_size=64,
        )
        full, final, states = chunk_kda(q, k, v, g, beta, **options)
        for boundary in (128, 256):
            resumed, restored_final, _ = chunk_kda(
                q[:, boundary:].contiguous(),
                k[:, boundary:].contiguous(),
                v[:, boundary:].contiguous(),
                g[:, boundary:].contiguous(),
                beta[:, boundary:].contiguous(),
                initial_state=states[:, boundary // 64].contiguous(),
                **options
            )
            torch.testing.assert_close(resumed, full[:, boundary:], atol=0, rtol=0)
            torch.testing.assert_close(restored_final, final, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
