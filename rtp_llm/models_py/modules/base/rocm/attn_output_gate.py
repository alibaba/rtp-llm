"""ROCm-specific attention output gate module (plain PyTorch fallback).

ROCm fallback for Qwen3.5 attention output gate. Uses native PyTorch ops;
no Triton dependency.
"""

import torch
import torch.nn as nn


class SigmoidMulInplace(nn.Module):
    """ROCm fallback: attn_output[:] = attn_output * sigmoid(gate)."""

    def forward(
        self,
        attn_output: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        attn_output.mul_(torch.sigmoid(gate))
        return attn_output
