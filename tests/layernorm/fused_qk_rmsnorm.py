import os
import unittest

import torch
from torch import nn

from rtp_llm.ops.compute_ops import rtp_llm_ops


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class TestLayerNorm(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_fused_qk_rmsnorm(self):
        torch.manual_seed(0)
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )

        dtype = torch.bfloat16

        eps = 1e-6
        norm_size = 128
        # for batch_size in [1]:
        #     for q_head, kvhead in [(2, 1)]:

        for batch_size in [2, 32, 4096, 512]:
            for q_head, kvhead, hidden_units in [
                (1, 1, 384),
                (64, 4, 9216),
                (32, 2, 4608),
                (16, 1, 2304),
            ]:
                hidden_units = (q_head + kvhead * 2) * norm_size
                q_gamma = torch.rand(norm_size, dtype=dtype).cuda()
                # q_bias = torch.rand(q_head, norm_size, dtype=dtype).cuda()
                k_gamma = torch.rand(norm_size, dtype=dtype).cuda()
                # k_bias = torch.rand(kvhead, norm_size, dtype=dtype).cuda()

                q_rms = Qwen3MoeRMSNorm(norm_size, eps)
                q_rms.weight = torch.nn.Parameter(q_gamma)

                k_rms = Qwen3MoeRMSNorm(norm_size, eps)
                k_rms.weight = torch.nn.Parameter(k_gamma)

                input = torch.rand(batch_size, hidden_units, dtype=dtype).cuda()

                input_ref = input.detach().clone()

                normed_q = q_rms(
                    input_ref.reshape(batch_size, -1, norm_size)[
                        :, :q_head, :
                    ].contiguous()
                )

                normed_k = k_rms(
                    input_ref.reshape(batch_size, -1, norm_size)[
                        :, q_head : q_head + kvhead, :
                    ].contiguous()
                )

                fused_qk_rmsnorm = torch.classes.unittest.FusedQkRmsNormOp(eps)
                fused_qk_rmsnorm.forward(
                    input, q_gamma, None, k_gamma, None, q_head, kvhead, norm_size
                )

                fused_q_norm = input.reshape(batch_size, -1, norm_size)[:, :q_head, :]
                fused_k_norm = input.reshape(batch_size, -1, norm_size)[
                    :, q_head : q_head + kvhead, :
                ]

                torch.testing.assert_close(normed_q, fused_q_norm, rtol=0.01, atol=0.01)
                torch.testing.assert_close(normed_k, fused_k_norm, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
