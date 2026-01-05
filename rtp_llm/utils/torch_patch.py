import logging
from typing import List

import torch

# 1. 保存原始 concat
original_concat = torch.concat


# 2. 定义自定义 concat
def custom_concat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    try:
        return original_concat(tensors, dim=dim).contiguous()
    except RuntimeError as e:
        if "cat_cuda" in str(e) and "Float8_e4m3fn" in str(e):
            logging.info("Caught Float8_e4m3fn concat error. Falling back to CPU.")
            original_device = tensors[0].device
            tensors_cpu = [t.cpu() for t in tensors]
            result_cpu = original_concat(tensors_cpu, dim=dim).contiguous()
            return result_cpu.to(original_device)
        else:
            raise


# 3. 劫持 torch.concat
torch.concat = custom_concat
