from typing import Dict

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.module_base import RtpModule


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")


def _validate_vocab_partition(vocab_size: int, tp_size: int, tp_rank: int) -> None:
    _positive_int(vocab_size, "vocab_size")
    _positive_int(tp_size, "tp_size")
    if isinstance(tp_rank, bool) or not isinstance(tp_rank, int):
        raise ValueError(f"tp_rank must be an integer, got {tp_rank!r}")
    if not 0 <= tp_rank < tp_size:
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
        _positive_int(embedding_dim, "embedding_dim")

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
        _positive_int(hidden_size, "hidden_size")

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
        if tensor.shape[0] != self.vocab_size:
            raise ValueError(
                f"LM head checkpoint rows must be {self.vocab_size}, "
                f"got {tensor.shape[0]}"
            )
        start = self.tp_rank * self.vocab_size_per_partition
        tensor = tensor[start : start + self.vocab_size_per_partition]
        if not self._assign_weight(self, "weight", tensor.contiguous()):
            raise RuntimeError("Failed to assign LM head weight")

    def _copy_local_tied_weight(self, tensor: torch.Tensor) -> None:
        if tuple(tensor.shape) != tuple(self.weight.shape):
            raise ValueError(
                f"Tied LM head shard must have shape {tuple(self.weight.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if not self._assign_weight(self, "weight", tensor.contiguous()):
            raise RuntimeError("Failed to assign tied LM head weight")

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            if name != "weight":
                raise RuntimeError(f"Unsupported LM head tensor {name!r}")
            self._copy_weight(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)
