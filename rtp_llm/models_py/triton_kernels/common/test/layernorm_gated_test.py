import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated


class Qwen3NextRMSNormGatedTorch(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


class TestLayerNormGated(unittest.TestCase):
    def test_rms_norm_gated(self):
        weight = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
        bias = None
        eps = 1e-6
        rms_norm_gated = RmsNormGated(weight, bias, eps)
        rms_norm_gated_torch = Qwen3NextRMSNormGatedTorch(weight, eps)
        for batch_size in [1, 2, 32, 128, 512, 1024]:
            x = torch.randn(batch_size, 1024, dtype=torch.bfloat16, device="cuda")
            gate = torch.randn(batch_size, 1024, dtype=torch.bfloat16, device="cuda")
            torch.testing.assert_close(
                rms_norm_gated(x, gate),
                rms_norm_gated_torch(x, gate),
                atol=1e-2,
                rtol=1e-2,
            )


if __name__ == "__main__":
    unittest.main()
