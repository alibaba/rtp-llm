import torch
import torch.distributed as dist
from maga_transformer.distribute.worker_info import g_parallel_info

def all_gather_tp(output: torch.Tensor) -> torch.Tensor:
    tensor_list = [torch.empty_like(output) for _ in range(g_parallel_info.tp_size)]
    tensor_list[g_parallel_info.tp_rank] = output
    dist.all_gather(tensor_list, output)
    output = torch.cat(tensor_list, dim=output.dim() - 1).contiguous()
    return output
