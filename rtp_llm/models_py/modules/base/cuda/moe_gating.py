"""CUDA-specific MoE gating module.

nn.Module wrapper around the Triton kernel in
rtp_llm.models_py.triton_kernels.common.moe_gating.
"""

import torch
import torch.nn as nn

from rtp_llm.models_py.triton_kernels.common.moe_gating import (
    _MAX_BLOCK_H,
    _MIN_BLOCK_H,
    _select_block_h,
    sigmoid_gate_scale_add_triton,
)


class SigmoidGateScaleAdd(nn.Module):
    """Fused sigmoid-gate-scale-add module for MoE shared-expert gating (CUDA/Triton).

    Fuses three PyTorch ops (sigmoid, mul, add) into a single Triton kernel,
    avoiding one intermediate [T, H] tensor allocation and two extra memory
    round-trips.
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
                               dtype: fp16 / bf16 / fp32.
            shared:  [T, H]  — shared expert MLP output.
            experts: [T, H]  — routed experts output; **modified in-place** and
                               returned as the result.

        Returns:
            experts tensor (same object, modified in-place).
        """
        assert (
            gate.ndim == 2 and gate.shape[1] == 1
        ), f"gate must be [T, 1], got {gate.shape}"
        assert (
            shared.shape == experts.shape
        ), f"shared and experts must have the same shape, got {shared.shape} vs {experts.shape}"
        assert (
            shared.is_cuda and experts.is_cuda and gate.is_cuda
        ), "All tensors must be on CUDA"

        T, H = shared.shape
        if T == 0 or H == 0:
            return experts

        return sigmoid_gate_scale_add_triton(gate, shared, experts)
