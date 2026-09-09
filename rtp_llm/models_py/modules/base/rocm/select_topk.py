import aiter
import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig


class SelectTopk(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.moe_k
        self._token_expert_indicies = torch.empty(0, dtype=torch.int32)

    def _scratch(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            return torch.empty(num_tokens, self.top_k, dtype=torch.int32, device=device)
        buf = self._token_expert_indicies
        if buf.numel() < num_tokens * self.top_k or buf.device != device:
            buf = torch.empty(num_tokens, self.top_k, dtype=torch.int32, device=device)
            self._token_expert_indicies = buf
        return buf[:num_tokens]

    def forward(
        self,
        router_logits: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        # aiter requires int32 and writes topk_ids in place.
        if topk_ids.dtype != torch.int32:
            raise TypeError(f"expect int32 topk_ids, got {topk_ids.dtype}")
        aiter.topk_softmax(
            topk_weights,
            topk_ids,
            self._scratch(topk_ids.shape[0], topk_ids.device),
            router_logits,
            True,
        )
