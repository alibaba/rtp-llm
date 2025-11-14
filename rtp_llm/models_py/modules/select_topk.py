import torch
from torch import nn
from rtp_llm.config.model_config import ModelConfig
from librtp_compute_ops.rtp_llm_ops import SelectTopkOp
    
class SelectTopk(nn.Module):
    def __init__(self, config: ModelConfig, fake_balance_expert: bool, dp_rank: int):
        super().__init__()
        self.config = config
        self.select_topk_op = SelectTopkOp(self.config, fake_balance_expert, dp_rank)

    def forward(self, router_logits_fp32: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        self.select_topk_op.forward(router_logits_fp32, topk_ids, topk_weights)