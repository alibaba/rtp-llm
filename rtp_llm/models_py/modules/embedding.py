import torch
from libth_transformer import rtp_llm_ops
from torch import nn
from torch.nn import functional as F


class EmbeddingTorch(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight)


class Embedding(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tokens = input.size(0)
        hidden_size = self.weight.size(-1)
        output = torch.empty(
            (tokens, hidden_size), dtype=self.weight.dtype, device=input.device
        )
        rtp_llm_ops.embedding(output, input, self.weight.data)
        return output
