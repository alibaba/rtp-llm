"""V4 PPU vocab lookup: every rank owns all token IDs and a hidden slice."""

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.read_spec import make_sharded_read_spec, unwrap_checkpoint_tensor
from rtp_llm.models_py.runtime_weight_update import prepare_leaf_weight_updates


class PpuHiddenShardEmbedding(RtpModule):
    supports_independent_weight_updates = True

    def prepare_weight_updates(self, weights):
        return prepare_leaf_weight_updates(self, weights)

    def __init__(self, vocab_size, dim, tp_size, tp_rank, *, dtype=torch.bfloat16):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (vocab_size, dim, tp_size)):
            raise ValueError("V4 embedding requires positive integer dimensions")
        if type(tp_rank) is not int or not 0 <= tp_rank < tp_size or dim % tp_size:
            raise ValueError("Invalid V4 embedding hidden partition")
        self.global_shape = (vocab_size, dim)
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self.weight = nn.Parameter(
            torch.empty((vocab_size, dim // tp_size), dtype=dtype), requires_grad=False
        )

    def checkpoint_read_spec(self, name, shape):
        if name != "weight":
            return None
        if tuple(shape) != self.global_shape:
            raise ValueError(f"V4 embedding expected {self.global_shape}, got {shape}")
        return make_sharded_read_spec(
            name, self.global_shape, 1, self.tp_size, self.tp_rank
        )

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        for name, value in items:
            if name != "weight":
                raise ValueError(f"Unexpected PPU embedding tensor {name}")
            tensor, spec = unwrap_checkpoint_tensor(value)
            expected = self.checkpoint_read_spec(
                name, spec.source_shape if spec else tensor.shape
            )
            if spec is None:
                tensor, _ = unwrap_checkpoint_tensor(expected.materialize(tensor))
            elif (spec.slices, spec.shard_dim, spec.shard_count, spec.shard_rank) != (
                expected.slices,
                expected.shard_dim,
                expected.shard_count,
                expected.shard_rank,
            ):
                raise ValueError("PPU embedding received a different TP shard")
            copy_weight_(self.weight, tensor, "PpuHiddenShardEmbedding.weight")
            self._mark_weight_loaded(name)

    def forward(self, input_ids):
        # PpuV4Transformer._embed owns the cross-rank hidden gather.
        return F.embedding(input_ids, self.weight)
