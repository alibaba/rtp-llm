import torch
import torch.distributed as dist
from typing import Optional
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.distribute.worker_info import g_parallel_info, g_master_info

class EmbeddingOp(object):
    def __init__(self, embedding_weight: torch.Tensor, position_weight: Optional[torch.Tensor], token_type_weight: Optional[torch.Tensor], all_gather: bool):
        self.embedding_weight_ = embedding_weight
        self.position_weight_ = position_weight
        self.token_type_weight_ = token_type_weight
        self.do_gather = all_gather
        self.embedding_op_ = \
            torch.classes.FasterTransformer.FusedEmbeddingOp(self.embedding_weight_,
                                                             self.position_weight_,
                                                             self.token_type_weight_)

    def forward(self, token_ids: torch.Tensor, position_ids: Optional[torch.Tensor], token_type_ids: Optional[torch.Tensor]):
        embed = self.embedding_op_.forward(token_ids, position_ids, token_type_ids)
        return self._maybe_all_gather(embed)
    
    def _maybe_all_gather(self, embed: torch.Tensor):
        if not self.do_gather or g_parallel_info.world_size == 0:
            return embed
        tensor_list = [torch.empty_like(embed) for _ in range(g_parallel_info.tp_size)]
        tensor_list[g_parallel_info.tp_rank] = embed
        dist.all_gather(tensor_list, embed)
        output = torch.cat(tensor_list, -1).contiguous()
        return output

        