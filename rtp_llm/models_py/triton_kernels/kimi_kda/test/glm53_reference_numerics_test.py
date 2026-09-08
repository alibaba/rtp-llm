"""Independent GLM arithmetic regressions for the K3-derived decode kernels.

References, audited 2026-09-09:
* zai-org/GLM-5.3-Flash@eb9eb208 (BF16 KDA weights, FP32 A_log/dt_bias).
* SGLang@30e7a307, glm5_next.py and the KDA backend (raw FP32 beta gate).
* Transformers@0a959de1, modeling_glm5_next.py (FP32 K-major recurrence
  and FP32 sigmoid-gated RMSNorm with a single final output cast).

These tests check the recurrence and gated norm independently of RTP's previous
kernel. They do not certify full-model quality or the FP8-to-FP4 MoE path.
"""

import unittest

import torch
from rtp_llm.models_py.triton_kernels.kimi_kda import fused_recurrent_kda
from rtp_llm.models_py.triton_kernels.kimi_kda.rms_norm_gate import (
    kimi_kda_rms_norm_sigmoid_gate,
)


def reference_step(q, k, v, raw_g, raw_beta, a_log, dt_bias, state):
    """GLM lower-bound decay and delta update in canonical [K,V] layout."""
    q, k, v = (x.float() for x in (q, k, v))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    log_decay = -5.0 * torch.sigmoid(
        a_log.float().exp()[None, :, None] * (raw_g.float() + dt_bias.float())
    )
    state = state.float() * log_decay.exp()[..., None]
    delta = (v - (state * k[..., None]).sum(-2)) * raw_beta.float().sigmoid()[..., None]
    state = state + k[..., None] * delta[..., None, :]
    return (state * q[..., None]).sum(-2), state


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53ReferenceNumericsTest(unittest.TestCase):
    def test_fp32_gated_norm_single_output_rounding(self):
        torch.manual_seed(530901)
        for batch in (1, 7, 48, 64):
            for weight_dtype in (torch.bfloat16, torch.float32):
                with self.subTest(batch=batch, weight_dtype=weight_dtype):
                    x = torch.randn(batch, 1, 64, 128, device="cuda").bfloat16()
                    gate = (torch.randn_like(x) * 8).contiguous()
                    weight = torch.randn(128, device="cuda").to(weight_dtype)
                    xf = x.float()
                    reference = (
                        xf
                        * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
                        * weight.float()
                        * gate.float().sigmoid()
                    ).bfloat16()
                    output = kimi_kda_rms_norm_sigmoid_gate(x, gate, weight, 1e-5)
                    self.assertEqual(output.dtype, torch.bfloat16)
                    torch.testing.assert_close(
                        output, reference, rtol=1 / 128, atol=1e-6
                    )
                    # Extra intermediate BF16 rounds are not the reference.
                    relative_l2 = (
                        (output.float() - reference.float()).norm()
                        / reference.float().norm()
                    )
                    self.assertLess(relative_l2.item(), 1e-4)

    def test_fp32_state_and_gate_with_reordered_pages(self):
        torch.manual_seed(530902)
        batch, heads, dim = 48, 64, 128
        table = torch.randperm(batch, device="cuda", dtype=torch.int32).add_(1)[:, None]
        lengths = torch.full((batch,), 64, dtype=torch.int32, device="cuda")
        a_log = torch.linspace(-3, 2, heads, device="cuda")
        bias = torch.linspace(-3, 3, heads * dim, device="cuda").view(heads, dim)
        initial = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
        state = initial.clone()
        reference_state = initial[table[:, 0].long()].clone()
        for step in range(32):
            q, k, v, g = [
                torch.randn(batch, 1, heads, dim, device="cuda").bfloat16() * 0.3
                for _ in range(4)
            ]
            beta = (torch.randn(batch, 1, heads, device="cuda") * 8).bfloat16()
            # Zero Q/K and saturated gates are meaningful numerical boundaries.
            q[0].zero_()
            k[1].zero_()
            output, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=a_log,
                dt_bias=bias.flatten(),
                initial_state=state,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=-5.0,
                block_map=table,
                seq_size_per_block=128,
                sequence_lengths=lengths,
                decode_low_warps=True,
            )
            reference, reference_state = reference_step(
                q[:, 0],
                k[:, 0],
                v[:, 0],
                g[:, 0],
                beta[:, 0],
                a_log,
                bias,
                reference_state,
            )
            with self.subTest(step=step):
                self.assertEqual(state.dtype, torch.float32)
                torch.testing.assert_close(
                    output[:, 0], reference.bfloat16(), rtol=1 / 128, atol=2e-5
                )
                torch.testing.assert_close(
                    state[table[:, 0].long()], reference_state, rtol=2e-4, atol=2e-5
                )
        torch.testing.assert_close(state[0], initial[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
