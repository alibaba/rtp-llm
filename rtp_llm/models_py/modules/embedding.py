import torch
from torch import nn
from torch import dtype as _dtype
from typing import Optional
from torch.nn import functional as F

class Embedding(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input,
            self.weight)

