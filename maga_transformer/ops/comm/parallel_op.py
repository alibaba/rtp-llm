import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict

from maga_transformer.utils.nccl_util import all_gather_tp

class ParallelEmbedding(object):
    def __init__(self, all_gather: bool):
        self._emb: Optional[torch.Tensor] = None
        self._scalar: Optional[float] = None
        self._all_gather = all_gather

    def set_weight(self, emb: torch.Tensor):
        self._emb = emb

    def set_scalar(self, scalar: float):
        self._scalar = scalar

    def __call__(self, input: torch.Tensor):
        assert self._emb is not None
        output = F.embedding(input, self._emb)
        if self._scalar:
            output = output * self._scalar
        if self._all_gather:
            return all_gather_tp(output)
        return output
    
    @property
    def weight(self):
        return self._emb

class ParallelLinear(object):
    def __init__(self, all_gather: bool):        
        self._w: Optional[torch.Tensor] = None
        self._b: Optional[torch.Tensor] = None
        self._all_gather = all_gather

    def set_weight(self, w: torch.Tensor, b: Optional[torch.Tensor]):
        self._w = w
        self._b = b
        
    @property
    def weight(self):
        return self._w
    
    @property
    def bias(self):
        return self._b

    def __call__(self, input: torch.Tensor):
        assert self._w is not None
        output = F.linear(input, self._w, self._b)
        if self._all_gather:
            return all_gather_tp(output)
        return output