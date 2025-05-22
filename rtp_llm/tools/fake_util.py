import torch
from typing import Dict, List

def generate_fake_model(shape_map: Dict[str, List[int]]):
    fake_model: Dict[str, torch.Tensor] = {}
    for key, shape in shape_map.items():
        print(f"generate tensor: {key}, shape: {shape}")
        fake_model[key] = torch.rand(shape, dtype=torch.half)
    return fake_model

def copy_from_model(shape_map: Dict[str, List[int]], model: Dict[str, torch.Tensor]):
    copy_model: Dict[str, torch.Tensor] = {}
    for key, shape in shape_map.items():
        print("key = ", key)
        print(f"copy tensor {key}, origin shape: {model[key].shape}, copy shape: {shape}")
        copy_model[key] = copy_tensor(model[key], shape)
        copy_model[key].contiguous()

    return copy_model

def copy_tensor(x: torch.Tensor, shape: List[int]) -> torch.Tensor:
    for i, dim_size in enumerate(shape):
        x = x.narrow(i, 0, dim_size)
    return torch.clone(x)