import torch
from torch import nn
from torch import dtype as _dtype
from typing import Optional
from torch.nn import functional as F

class EmbeddingTorch(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input,
            self.weight)

class Embedding(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tokens = input.size(0)
        hidden_size = self.weight.size(-1)
        output = torch.empty((tokens, hidden_size), dtype=self.weight.dtype, device=input.device)
        torch.ops.libth_transformer.embedding(output, input, self.weight.data, 0)
        return output

