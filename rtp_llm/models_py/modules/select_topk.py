import torch
from torch import nn
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from libth_transformer.rtp_llm_ops import SelectTopkOp
    
class SelectTopk(nn.Module):
    def __init__(self, config: GptInitModelParameters):
        super().__init__()
        self.config = config
        self.select_topk_op = SelectTopkOp(self.config)

    def forward(self, router_logits_fp32: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        self.select_topk_op.forward(router_logits_fp32, topk_ids, topk_weights)