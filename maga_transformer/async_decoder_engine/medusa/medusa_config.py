import torch
from typing import NamedTuple, List
from dataclasses import dataclass

# 这个是不会变的
class MedusaBuffer(NamedTuple):
    medusa_attn_mask: torch.Tensor
    tree_indices: torch.Tensor
    medusa_position_ids: List[int]
    retrieve_indices: torch.Tensor    

@dataclass
class MedusaState:
    candidates: torch.Tensor
    tree_candidates: torch.Tensor