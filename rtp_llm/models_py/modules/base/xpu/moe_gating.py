"""XPU MoE gating - PyTorch fallback."""
import torch
import torch.nn as nn


class SigmoidGateScaleAdd(nn.Module):
    def forward(self, gate: torch.Tensor, shared: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
        assert gate.ndim == 2 and gate.shape[0] == experts.shape[0] and gate.shape[1] == 1, (
            f"SigmoidGateScaleAdd: gate must be [T, 1], got {gate.shape} "
            f"vs experts {experts.shape}"
        )
        assert shared.shape == experts.shape, (
            f"SigmoidGateScaleAdd: shared/experts shape mismatch: "
            f"shared={shared.shape}, experts={experts.shape}"
        )
        # Accumulate the gate scaling in fp32 to match the CUDA Triton kernel's
        # precision, then cast back to the experts dtype before the in-place add.
        scaled = (torch.sigmoid(gate.float()) * shared.float()).to(experts.dtype)
        experts.add_(scaled)
        return experts
