"""CUDA-specific MoE gating module.

nn.Module wrapper around the Triton kernel in
rtp_llm.models_py.triton_kernels.common.moe_gating.
"""

from typing import Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.triton_kernels.common.moe_gating import (
    sigmoid_gate_scale_add_triton,
)


class SigmoidGateScaleAdd(nn.Module):
    """Fused sigmoid-gate-scale-add module for MoE shared-expert gating (CUDA/Triton).

    Fuses three PyTorch ops (sigmoid, mul, add) into a single Triton kernel,
    avoiding one intermediate [T, H] tensor allocation and two extra memory
    round-trips.  With ``topk_sum`` the routed experts' top-k reduction is
    folded in as well, replacing the separate ``sum(dim=1)`` aten launch and
    its [T, H] intermediate round-trip.
    """

    def forward(
        self,
        gate: torch.Tensor,
        shared: torch.Tensor,
        experts: torch.Tensor,
        topk_sum: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute in-place on *experts* (topk_sum is None or 1):
            experts[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + experts[t, :]

        With topk_sum > 1, *experts* is the unsummed routed output
        [T, topk_sum, H] and the top-k sum is fused into the kernel:
            out[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + sum_k experts[t, k, :]
        writing a fresh [T, H] tensor (returned).

        Args:
            gate:    [T, 1]  — scalar gate from shared_expert_gate linear layer.
                               dtype: fp16 / bf16 / fp32.
            shared:  [T, H]  — shared expert MLP output.
            experts: [T, H]  — routed experts output; **modified in-place** and
                               returned as the result (classic path).
            topk_sum: deferred top-k to fold into the sum, None for the
                               classic in-place path.

        Returns:
            experts tensor (classic, same object modified in-place) or the
            summed [T, H] result (topk_sum path).
        """
        assert (
            gate.ndim == 2 and gate.shape[1] == 1
        ), f"gate must be [T, 1], got {gate.shape}"
        if topk_sum is not None and topk_sum > 1:
            assert (
                experts.ndim == 3 and experts.size(0) == shared.size(0)
            ), f"experts must be [T, topk_sum, H], got {experts.shape}"
            assert (
                shared.is_cuda and experts.is_cuda and gate.is_cuda
            ), "All tensors must be on CUDA"
        else:
            assert (
                shared.shape == experts.shape
            ), f"shared and experts must have the same shape, got {shared.shape} vs {experts.shape}"
            assert (
                shared.is_cuda and experts.is_cuda and gate.is_cuda
            ), "All tensors must be on CUDA"

        T, H = shared.shape
        if T == 0 or H == 0:
            return experts

        return sigmoid_gate_scale_add_triton(gate, shared, experts, topk_sum)
