"""GLM-5.3 router projection in FP32, with bounded activation storage."""

import torch
import torch.nn.functional as F
from torch import nn


class Glm53FP32Router(nn.Module):
    def __init__(self, weight: torch.Tensor, chunk_rows: int = 8192):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.float32:
            raise ValueError(
                "GLM-5.3 router requires a loaded FP32 [hidden, experts] weight"
            )
        if chunk_rows <= 0:
            raise ValueError("router chunk_rows must be positive")
        self.register_buffer("weight", weight.T.contiguous())
        self.chunk_rows = chunk_rows

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("GLM-5.3 FP32 router requires matmul.allow_tf32=False")
        if hidden_states.shape[0] <= self.chunk_rows:
            return F.linear(hidden_states.float(), self.weight)
        return torch.cat(
            [
                F.linear(x.float(), self.weight)
                for x in hidden_states.split(self.chunk_rows)
            ],
            dim=0,
        )
