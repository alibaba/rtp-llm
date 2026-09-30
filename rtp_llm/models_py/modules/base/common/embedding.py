import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import rtp_llm_ops


class EmbeddingTorch(nn.Module):
    def __init__(self, weight: torch.Tensor, *, tp_size: int = 1):
        super().__init__()
        self.weight = weight
        self.tp_size = tp_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.embedding(input, self.weight)
        if self.tp_size > 1:
            # Embedding weights are sharded along hidden width. The collective
            # concatenates ranks along dimension zero, so restore token order
            # before joining each token's hidden slices.
            local_width = output.shape[-1]
            gathered = all_gather(output.reshape(-1, local_width), group=Group.TP)
            output = (
                gathered.reshape(self.tp_size, input.numel(), local_width)
                .transpose(0, 1)
                .contiguous()
                .reshape(*input.shape, self.tp_size * local_width)
            )
        return output


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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tokens = input.size(0)
        hidden_size = self.weight.size(-1)
        output = torch.empty(
            (tokens, hidden_size), dtype=self.weight.dtype, device=input.device
        )
        rtp_llm_ops.embedding(output, input, self.weight.data)
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
        tokens = input.size(0)
        hidden_size = self.weight.size(-1)
        output = torch.empty(
            (tokens, hidden_size), dtype=self.weight.dtype, device=input.device
        )

        rtp_llm_ops.embedding_bert(
            output,
            input,
            self.weight.data,
            combo_position_ids,
            position_encoding,
            combo_tokens_type_ids,
            token_type_embedding,
            input_embedding_scalar,
        )

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
