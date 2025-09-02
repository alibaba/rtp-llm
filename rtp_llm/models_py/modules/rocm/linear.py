from typing import Optional

import torch
from libth_transformer import rtp_llm_ops
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            *input.shape[:-1], self.weight.shape[1], dtype=input.dtype
        ).to(input.device)
        rtp_llm_ops.gemm(output, input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output
