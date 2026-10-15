"""CUDA-specific MoE gating module.

Bit-exact decomposition of the aten shared-expert gating chain using the
verified Triton kernels from
rtp_llm.models_py.triton_kernels.common.moe_combine.
"""

from typing import Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.triton_kernels.common.moe_combine import (
    sigmoid_mul_supported,
    sigmoid_mul_triton,
    topk_sum_supported,
    topk_sum_triton,
)


class SigmoidGateScaleAdd(nn.Module):
    """Sigmoid-gate-scale-add module for MoE shared-expert gating (CUDA/Triton).

    Computes the aten chain ``sigmoid(gate) * shared + experts`` (plus the
    ``sum(dim=1)`` fold for a deferred [T, topk, H] routed output) with
    bit-identical results:

    - ``sigmoid_mul_triton`` replaces ``torch.sigmoid(gate) * shared`` — a
      single kernel, bit-exact vs the two-op aten chain.
    - ``topk_sum_triton`` replaces ``experts.sum(dim=1)`` — replicates the
      aten reduce_kernel's fp32 ILP-4 accumulation order, bit-exact.
    - the trailing ``add_`` stays on aten (its fp32-opmath round matches).

    A single fully-fused kernel was tried and retired: keeping the rounded
    multiply result as an add operand requires an in-register
    fp32 -> bf16 -> fp32 materialization that the Triton/LLVM pipeline folds
    away, so the fused kernel computed the whole chain in fp32 with one
    final rounding — not bit-identical to the per-op aten chain (measured
    element-level diffs; see
    runs/minicpmv47_h20_20261013/build_logs/ab_moe_gating.py in the RUN
    notes).
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
        [T, topk_sum, H]; the top-k sum is folded in first and a fresh
        [T, H] tensor is returned:
            out[t, :] = sigmoid(gate[t, 0]) * shared[t, :] + sum_k experts[t, k, :]

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
            experts = (
                topk_sum_triton(experts)
                if topk_sum_supported(experts)
                else experts.sum(dim=1)
            )
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

        if sigmoid_mul_supported(gate, shared):
            experts.add_(sigmoid_mul_triton(gate, shared))
        else:
            experts.add_(torch.sigmoid(gate) * shared)
        return experts
