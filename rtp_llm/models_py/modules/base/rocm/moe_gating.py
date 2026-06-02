"""ROCm-specific MoE gating implementation (plain PyTorch fallback).

Uses native PyTorch ops; no Triton dependency for ROCm compatibility.
"""

import torch
import torch.nn as nn


class SigmoidGateScaleAdd(nn.Module):
    """Sigmoid-gate-scale-add module for MoE shared-expert gating (ROCm/PyTorch).

    ROCm fallback implementation using plain PyTorch ops.
    """

    def forward(
        self,
        gate: torch.Tensor,
        shared: torch.Tensor,
        experts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute in-place on *experts*:
            experts[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + experts[t, :]

        Args:
            gate:    [T, 1]  — scalar gate from shared_expert_gate linear layer.
            shared:  [T, H]  — shared expert MLP output.
            experts: [T, H]  — routed experts output; modified in-place and returned.

        Returns:
            experts tensor (same object, modified in-place).
        """
        T, H = shared.shape
        if T == 0 or H == 0:
            return experts

        experts.add_(torch.sigmoid(gate) * shared)
        return experts
