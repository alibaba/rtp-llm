from typing import Optional
import torch
from torch import nn

from aiter import hipb_mm, hipb_create_extension
from functools import lru_cache

class Linear(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, bpreshuffle: bool = False
    ) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.bpreshuffle = bpreshuffle
     
    @staticmethod    
    @lru_cache(maxsize=1)
    def init_hipblas():
        hipb_create_extension()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.init_hipblas()
        return hipb_mm(
            input,
            self.weight,
            solution_index=-1,
            bias=self.bias,
            out_dtype=input.dtype,
            scaleA=None,
            scaleB=None,
            scaleOut=None,
            bpreshuffle=self.bpreshuffle,
        )
