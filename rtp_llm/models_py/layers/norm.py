from typing import Dict, Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name:
                self.weight.data.copy_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class LayerNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name or "gamma" in name:
                self.weight.data.copy_(tensor)
            elif "bias" in name or "beta" in name:
                self.bias.data.copy_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, [self.hidden_size], self.weight, self.bias, self.eps
        )
