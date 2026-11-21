"""V4.1 global matrix owners matching main's hidden-TP embedding layout."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.read_spec import make_sharded_read_spec, unwrap_checkpoint_tensor


class OwnedV41Matrix(RtpModule):
    def __init__(self, rows, columns, *, shard_dim, tp_size, tp_rank, dtype):
        super().__init__()
        if (
            any(
                type(value) is not int or value < 1
                for value in (rows, columns, tp_size)
            )
            or type(tp_rank) is not int
            or not 0 <= tp_rank < tp_size
        ):
            raise ValueError("V4.1 global matrix dimensions/TP partition are invalid")
        if shard_dim not in (0, 1) or (rows, columns)[shard_dim] % tp_size:
            raise ValueError("V4.1 global matrix must divide its TP partition")
        self.source_shape, self.shard_dim = (rows, columns), shard_dim
        self.tp_size, self.tp_rank = tp_size, tp_rank
        shape = [rows, columns]
        shape[shard_dim] //= tp_size
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        object.__setattr__(self, "_runtime", None)

    def checkpoint_read_spec(self, name, shape):
        if name != "weight" or tuple(shape) != self.source_shape:
            raise ValueError(f"Invalid V4.1 global {name} source shape {tuple(shape)}")
        return make_sharded_read_spec(
            name, self.source_shape, self.shard_dim, self.tp_size, self.tp_rank
        )

    def load_weights(self, weights):
        for name, value in weights.items() if isinstance(weights, dict) else weights:
            if name in getattr(self, "_rtp_loaded_weight_names", ()):
                raise RuntimeError("Duplicate V4.1 global checkpoint weight")
            tensor, applied = unwrap_checkpoint_tensor(value)
            if applied is None:
                tensor, _ = unwrap_checkpoint_tensor(
                    self.checkpoint_read_spec(name, tensor.shape).materialize(tensor)
                )
            else:
                expected = self.checkpoint_read_spec(name, applied.source_shape)
                if (
                    applied.slices != expected.slices
                    or applied.shard_rank != expected.shard_rank
                ):
                    raise ValueError("V4.1 global ReadSpec belongs to another TP shard")
            copy_weight_(self.weight, tensor, "V4.1 global weight")
            self._mark_weight_loaded(name)

    def bind_embedding_runtime(self, config, parallelism_config):
        if self.shard_dim != 1:
            raise ValueError(
                "Only the hidden-sharded embedding owns a V41Embedding executor"
            )
        from rtp_llm.models_py.modules.dsv41.utils import V41Embedding

        object.__setattr__(
            self, "_runtime", V41Embedding(config, parallelism_config, self.weight)
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        object.__setattr__(self, "_runtime", None)
        return result

    def forward(self, input_ids):
        if self._runtime is None:
            raise RuntimeError(
                "V4.1 embedding must bind its main runtime before execution"
            )
        return self._runtime(input_ids)
