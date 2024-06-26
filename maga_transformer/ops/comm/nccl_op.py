from typing import List
import torch
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.distribute.worker_info import g_parallel_info, g_master_info

def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

@singleton
class NcclOp(FTOPBase):
    def __init__(self):
        super().__init__()
        self.ft_op_ = torch.classes.FasterTransformer.NcclOp( # type: ignore
            g_parallel_info.tp_size,
            g_parallel_info.pp_size,
            g_master_info.ip,
            g_master_info.nccl_op_port)

    def broadcast_tp(self, tensors: List[torch.Tensor], root: int = 0):
        self.ft_op_.broadcast_tp(tensors, root)

    def barrier(self):
        dummy_tensor = torch.zeros(1)
        self.ft_op_.broadcast_tp([dummy_tensor], 0)
        torch.cuda.current_stream().synchronize()
