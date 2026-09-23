"""K3 router projection, with fixed accumulation for BF16 checkpoint weights."""

import torch
from torch import nn


class KimiK3RouterProjection(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("K3 router weight must have shape [hidden, experts]")
        self.register_buffer("weight", weight.float(), persistent=False)
        # Converted BF16 weights are exact FP32 values. Never round genuinely
        # higher-precision checkpoint weights merely to use the BF16 kernel.
        self._bf16_exact = weight.dtype == torch.bfloat16 or torch.equal(
            self.weight, self.weight.bfloat16().float()
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.is_cuda and hidden.dtype == torch.bfloat16 and self._bf16_exact:
            from .router_gemm import router_gemm

            return router_gemm(hidden, self.weight)
        return hidden.float() @ self.weight
