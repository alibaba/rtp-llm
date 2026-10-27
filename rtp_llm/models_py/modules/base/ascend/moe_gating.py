import torch
import torch.nn as nn


class SigmoidGateScaleAdd(nn.Module):
    def forward(
        self,
        gate: torch.Tensor,
        shared: torch.Tensor,
        experts: torch.Tensor,
    ) -> torch.Tensor:
        T, H = shared.shape
        if T == 0 or H == 0:
            return experts
        experts.add_(torch.sigmoid(gate) * shared)
        return experts
