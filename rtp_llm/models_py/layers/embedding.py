from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.ops.compute_ops import rtp_llm_ops


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        # Floor-division shards below would silently lose the tail vocab rows
        # if vocab_size doesn't divide evenly. Caller should pad vocab_size to
        # a multiple of tp_size (vLLM does this in its config) — fail fast
        # rather than silently corrupt the embedding shard layout.
        assert vocab_size % tp_size == 0, (
            f"vocab_size ({vocab_size}) must be divisible by tp_size ({tp_size}); "
            f"pad vocab_size up to a multiple of tp_size before constructing "
            f"VocabParallelEmbedding."
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.vocab_size_per_partition = vocab_size // tp_size
        self.vocab_start_idx = tp_rank * self.vocab_size_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.vocab_size_per_partition

        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition, embedding_dim, dtype=params_dtype
            ),
            requires_grad=False,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name:
                if self.tp_size > 1:
                    tensor = tensor[self.vocab_start_idx : self.vocab_end_idx, :]
                self.weight.data.copy_(tensor)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        text_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.tp_size > 1:
            mask = (input_ids >= self.vocab_start_idx) & (
                input_ids < self.vocab_end_idx
            )
            masked_ids = (input_ids - self.vocab_start_idx) * mask
            output = torch.nn.functional.embedding(masked_ids, self.weight)
            output = output * mask.unsqueeze(-1)
            output = all_reduce(output, group=Group.TP)
        else:
            output = torch.nn.functional.embedding(input_ids, self.weight)
        return output


class HiddenParallelEmbedding(nn.Module):
    """Embedding whose hidden dimension is tensor-parallel sharded.

    This mirrors the legacy PyModel Embedding path: each rank stores
    [vocab_size, hidden_size / tp_size], looks up its local hidden shard, then
    all-gathers hidden shards back into the full hidden state.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        assert embedding_dim % tp_size == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by tp_size "
            f"({tp_size}) for HiddenParallelEmbedding."
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.embedding_dim_per_partition = embedding_dim // tp_size
        self.hidden_start_idx = tp_rank * self.embedding_dim_per_partition
        self.hidden_end_idx = self.hidden_start_idx + self.embedding_dim_per_partition

        self.weight = nn.Parameter(
            torch.empty(
                vocab_size, self.embedding_dim_per_partition, dtype=params_dtype
            ),
            requires_grad=False,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name:
                if self.tp_size > 1:
                    tensor = tensor[:, self.hidden_start_idx : self.hidden_end_idx]
                self.weight.data.copy_(tensor)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        text_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = input_ids.size(0)
        local_hidden_size = self.weight.size(-1)
        output = torch.empty(
            (tokens, local_hidden_size),
            dtype=self.weight.dtype,
            device=input_ids.device,
        )
        rtp_llm_ops.embedding(
            output,
            input_ids,
            self.weight.data,
            position_ids,
            token_types,
            text_tokens_mask,
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


class ParallelLMHead(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        assert vocab_size % tp_size == 0, (
            f"vocab_size ({vocab_size}) must be divisible by tp_size ({tp_size}); "
            f"pad vocab_size up to a multiple of tp_size before constructing "
            f"ParallelLMHead."
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.vocab_size_per_partition = vocab_size // tp_size
        self.weight = nn.Parameter(
            torch.empty(self.vocab_size_per_partition, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name:
                if self.tp_size > 1:
                    start = self.tp_rank * self.vocab_size_per_partition
                    tensor = tensor[start : start + self.vocab_size_per_partition, :]
                self.weight.data.copy_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)
