"""ROCm-specific MoE gating implementation (plain PyTorch fallback).

Uses native PyTorch ops; no Triton dependency for ROCm compatibility.
"""

from typing import Optional

import torch
import torch.nn as nn


class SigmoidGateScaleAdd(nn.Module):
    """Sigmoid-gate-scale-add module for MoE shared-expert gating (ROCm/PyTorch).

    ROCm fallback implementation using plain PyTorch ops.  Signature matches
    the CUDA/Triton module: ``topk_sum`` is accepted for cross-platform call
    sites, but the ROCm executors never defer the top-k sum, so it only
    reduces an already-materialized [T, topk_sum, H] view.
    """

    def forward(
        self,
        gate: torch.Tensor,
        shared: torch.Tensor,
        experts: torch.Tensor,
        topk_sum: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute in-place on *experts*:
            experts[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + experts[t, :]

        Args:
            gate:    [T, 1]  — scalar gate from shared_expert_gate linear layer.
            shared:  [T, H]  — shared expert MLP output.
            experts: [T, H]  — routed experts output; modified in-place and returned.
            topk_sum: deferred top-k; when set, *experts* is [T, topk_sum, H]
                               and is reduced with torch.sum first.

        Returns:
            experts tensor (classic path: same object, modified in-place;
            topk_sum path: the summed [T, H] tensor).
        """
        if topk_sum is not None and topk_sum > 1:
            experts = experts.sum(dim=1)
        T, H = shared.shape
        if T == 0 or H == 0:
            return experts

        experts.add_(torch.sigmoid(gate) * shared)
        return experts
