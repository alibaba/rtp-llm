import time
import torch
from typing import List

def get_first_token_from_combo_tokens(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    start_indices = torch.cumsum(torch.cat((torch.tensor([0]), lengths[:-1])), dim=0)    
    return tensor[start_indices]

def get_last_token_from_combo_tokens(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    end_indices = torch.cumsum(lengths, dim=0) - 1
    return tensor[end_indices]