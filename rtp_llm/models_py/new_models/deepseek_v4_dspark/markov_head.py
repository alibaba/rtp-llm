"""DSpARK sampler parameters owned by the NewLoader module tree.

Execution stays in the existing C++ MtpExecutor sampler. These tables are full
vocabulary tensors on every sampler rank, not tensor-parallel linear shards.
This leaf alone does not register or implement a full DSpARK draft model.
"""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule, _mark_loaded
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _MarkovTable(RtpModule):
    def __init__(self, vocabulary, padded_vocabulary, rank):
        super().__init__()
        self.vocabulary = vocabulary
        self.weight = nn.Parameter(
            torch.empty(padded_vocabulary, rank, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def load_weights(self, weights):
        for name, value in weights.items():
            if name != "weight":
                raise ValueError(f"Unknown Markov tensor {name}")
            tensor, _ = unwrap_checkpoint_tensor(value)
            if (
                tensor.ndim != 2
                or tensor.shape[1] != self.weight.shape[1]
                or not self.vocabulary <= tensor.shape[0] <= self.weight.shape[0]
            ):
                raise ValueError("Shape mismatch for DSpARK Markov table")
            if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise TypeError(
                    "DSpARK Markov checkpoint must use floating inference weights"
                )
            if "weight" in getattr(self, "_rtp_loaded_weight_names", ()):
                raise RuntimeError("Duplicate DSpARK Markov table")
            with torch.no_grad():
                self.weight[: tensor.shape[0]].copy_(tensor)
                self.weight[tensor.shape[0] :].zero_()
            _mark_loaded(self, "weight")


class DSparkMarkovHead(RtpModule):
    def __init__(
        self,
        target_vocab_size,
        draft_vocab_size,
        markov_rank,
        *,
        padded_draft_vocab_size=None,
        head_stage=2,
    ):
        super().__init__()
        padded = (
            draft_vocab_size
            if padded_draft_vocab_size is None
            else padded_draft_vocab_size
        )
        for name, value in (
            ("target_vocab_size", target_vocab_size),
            ("draft_vocab_size", draft_vocab_size),
            ("markov_rank", markov_rank),
            ("padded_draft_vocab_size", padded),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if padded < draft_vocab_size:
            raise ValueError(
                "padded draft vocabulary cannot be smaller than draft vocabulary"
            )
        if (
            isinstance(head_stage, bool)
            or not isinstance(head_stage, int)
            or head_stage < 0
        ):
            raise ValueError("head_stage must be a non-negative integer")
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.markov_rank = markov_rank
        self.checkpoint_prefix = f"mtp.{head_stage}.markov_head."
        self.WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={self.checkpoint_prefix: ""})
        self.markov_w1 = _MarkovTable(target_vocab_size, target_vocab_size, markov_rank)
        self.markov_w2 = _MarkovTable(draft_vocab_size, padded, markov_rank)
        self.requires_grad_(False)

    def checkpoint_weight_name_filter(self):
        names = {
            self.checkpoint_prefix + name
            for name in ("markov_w1.weight", "markov_w2.weight")
        }
        return names.__contains__

    def load_weights(self, weights):
        weights = weights.items() if isinstance(weights, dict) else weights
        super().load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def runtime_weight_view(self):
        # Public sampler export keys, not a second checkpoint naming scheme.
        return {
            "dspark_markov_w1.weight": self.markov_w1.weight,
            "dspark_markov_w2.weight": self.markov_w2.weight,
        }
