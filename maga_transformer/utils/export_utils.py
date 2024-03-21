import os
import torch
import numpy as np
from typing import Dict

# This file helps exporting tensor to file so that they can be loaded from cpp world.
# DO NOT REMOVE THIS FILE

# method 1: torch.jit.save
def export_tensors_to_jit_module(tensor_map: Dict[str, torch.Tensor], file_name: str) -> None:
    class Container(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            for key in values:
                setattr(self, key, values[key].cpu())

    container = torch.jit.script(Container(tensor_map).cpu())
    container.save(file_name)

# method 2: export to numpy dir
def export_tensors_to_numpy_dir(tensor_map: Dict[str, torch.Tensor], dir_name: str) -> None:
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass
    for key in tensor_map:
        np.save(dir_name + "/" + key + ".npy", tensor_map[key].cpu().numpy())

# use example:
#
# dump_dir = "maga_transformer/test/model_test/fake_test/testdata/qwen_0.5b"
# export_tensors_to_jit_module(self.weight._pytorch_weights, os.path.join(dump_dir, "pytorch_tensors.pt"))
# export_tensors_to_numpy_dir(self.weight._pytorch_weights, os.path.join(dump_dir, "pytorch_tensors"))
# for i in range(len(self.weight.weights)):
#     export_tensors_to_jit_module(self.weight.weights[i], os.path.join(dump_dir, f"layer_{i}.pt"))
#     export_tensors_to_numpy_dir(self.weight.weights[i], os.path.join(dump_dir, f"layer_{i}"))

# use exapmle: export from transformers model

# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
# params = {}
# for key, p in model.named_parameters():
#     params.update({key: p.data})
# export_tensors_to_jit_module(params, "model.pt")

