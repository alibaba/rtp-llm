from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.ops import ParallelismConfig


class Embedding(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        self.config = config
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()

    def forward(
        self,
        input: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        text_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Qwen-family models have no absolute position/type embedding tables
        # (positions are applied by RoPE in the attention layers), so
        # position_ids/token_types are accepted for signature parity and
        # ignored. text_tokens_mask carries semantics: mask==0 positions hold
        # feature hash ids (not real token ids, possibly out-of-vocab), so we
        # clamp them to token 0 for a safe lookup and zero the result rows —
        # mirroring embedding_kernels.cu's USE_MASK path. The real multimodal
        # features are inserted by MultimodalEmbeddingInjector afterwards.
        if text_tokens_mask is not None and text_tokens_mask.numel() == input.numel():
            mask = text_tokens_mask.to(input.device, dtype=torch.bool)
            safe_ids = torch.where(mask, input, torch.zeros_like(input))
            output = F.embedding(safe_ids, self.weight)
            output = output * mask.unsqueeze(-1).to(output.dtype)
        else:
            output = F.embedding(input, self.weight)
        if self.tp_size > 1:
            m, n = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.tp_size, m, n)
                .transpose(0, 1)
                .contiguous()
                .reshape(m, -1)
            )
        return output


class EmbeddingBert(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        self.config = config
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()

    def forward(
        self,
        input: torch.Tensor,
        combo_position_ids: torch.Tensor,
        position_encoding: torch.Tensor,
        combo_tokens_type_ids: torch.Tensor,
        token_type_embedding: torch.Tensor,
        input_embedding_scalar: float,
    ) -> torch.Tensor:
        token_emb = F.embedding(input, self.weight)
        pos_emb = F.embedding(combo_position_ids, position_encoding)
        type_emb = F.embedding(combo_tokens_type_ids, token_type_embedding)
        output = token_emb * input_embedding_scalar + pos_emb + type_emb

        if self.tp_size > 1:
            m, n = output.shape
            output = all_gather(output, group=Group.TP)
            output = (
                output.reshape(self.tp_size, m, n)
                .transpose(0, 1)
                .contiguous()
                .reshape(m, -1)
            )
        return output
