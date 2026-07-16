from typing import Dict

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.module_base import RtpModule


def _validate_vocab_partition(vocab_size: int, tp_size: int, tp_rank: int) -> None:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"Invalid TP partition: rank={tp_rank}, size={tp_size}")
    if vocab_size % tp_size != 0:
        raise ValueError(
            f"vocab_size={vocab_size} must be divisible by tp_size={tp_size}"
        )


class VocabParallelEmbedding(RtpModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        _validate_vocab_partition(vocab_size, tp_size, tp_rank)
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.vocab_size_per_partition = vocab_size // tp_size
        self.vocab_start_idx = tp_rank * self.vocab_size_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.vocab_size_per_partition
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition,
                embedding_dim,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            if name != "weight":
                raise RuntimeError(f"Unsupported embedding tensor {name!r}")
            if tensor.shape[0] != self.vocab_size:
                raise ValueError(
                    f"Embedding checkpoint rows must be {self.vocab_size}, "
                    f"got {tensor.shape[0]}"
                )
            local = tensor[self.vocab_start_idx : self.vocab_end_idx]
            if not self._assign_weight(self, "weight", local.contiguous()):
                raise RuntimeError("Failed to assign embedding weight")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return torch.nn.functional.embedding(input_ids, self.weight)

        mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
        local_ids = torch.where(
            mask,
            input_ids - self.vocab_start_idx,
            torch.zeros_like(input_ids),
        )
        output = torch.nn.functional.embedding(local_ids, self.weight)
        output = output * mask.unsqueeze(-1)
        return all_reduce(output, group=Group.TP)


class ParallelLMHead(RtpModule):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        _validate_vocab_partition(vocab_size, tp_size, tp_rank)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.vocab_size_per_partition = vocab_size // tp_size
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

    def _copy_weight(self, tensor: torch.Tensor) -> None:
        if tensor.shape[0] == self.vocab_size:
            start = self.tp_rank * self.vocab_size_per_partition
            tensor = tensor[start : start + self.vocab_size_per_partition]
        elif tensor.shape[0] != self.vocab_size_per_partition:
            raise ValueError(
                f"LM head rows must be {self.vocab_size} or "
                f"{self.vocab_size_per_partition}, got {tensor.shape[0]}"
            )
        if not self._assign_weight(self, "weight", tensor.contiguous()):
            raise RuntimeError("Failed to assign LM head weight")

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            if name != "weight":
                raise RuntimeError(f"Unsupported LM head tensor {name!r}")
            self._copy_weight(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)
