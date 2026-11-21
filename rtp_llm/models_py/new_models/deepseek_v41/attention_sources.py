"""V4.1 shared KV/index source weights; consumer layers own no duplicate."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v41.linear_owner import Dsv41MXFP8Linear


class _BF16Weight(RtpModule):
    def __init__(self, *shape):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(shape, dtype=torch.bfloat16), requires_grad=False
        )


class OwnedV41AttentionSources(RtpModule):
    def __init__(
        self,
        *,
        layer_id,
        dim,
        head_dim,
        q_lora_rank,
        index_n_heads,
        index_head_dim,
        compress_ratio,
        v41_config,
    ):
        super().__init__()
        if compress_ratio not in (0, 1, 2):
            raise ValueError("V4.1 compression ratio must be 0, 1 or 2")
        sources = {}
        for key in ("kv_source_layer_ids", "index_source_layer_ids"):
            values = tuple(v41_config.get(key, ()))
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{key} must contain nonnegative layer IDs")
            sources[key] = values
        self.layer_id, self.compress_ratio = layer_id, compress_ratio
        self.is_kv_source = layer_id in sources["kv_source_layer_ids"]
        self.is_index_source = layer_id in sources["index_source_layer_ids"]
        self.kv_source_layer_id = self.index_source_layer_id = None
        if compress_ratio:
            for key, attr in (
                ("kv_source_layer_ids", "kv_source_layer_id"),
                ("index_source_layer_ids", "index_source_layer_id"),
            ):
                preceding = [value for value in sources[key] if value <= layer_id]
                if not preceding:
                    raise ValueError(
                        f"Compressed V4.1 layer {layer_id} has no preceding {key}"
                    )
                setattr(self, attr, max(preceding))
        if self.is_kv_source and not compress_ratio:
            raise ValueError("A V4.1 global KV producer must be compressed")
        self.compressor = RtpModule()
        self.indexer = RtpModule()
        if self.is_kv_source:
            self.compressor.wkv = _BF16Weight(head_dim, dim)
            self.compressor.norm = _BF16Weight(head_dim)
            if compress_ratio == 2:
                self.compressor.wgate = _BF16Weight(head_dim, dim)
            self.indexer.wk = _BF16Weight(index_head_dim, head_dim)
            self.indexer.k_norm = _BF16Weight(index_head_dim)
        if self.is_index_source:
            self.indexer.wq_b = Dsv41MXFP8Linear(
                q_lora_rank, index_n_heads * index_head_dim
            )
            self.indexer.weights_proj = _BF16Weight(index_n_heads, dim)

    def runtime_parameter_view(self):
        view = {}
        if self.is_kv_source:
            view.update(
                global_wkv=self.compressor.wkv.weight,
                global_norm=self.compressor.norm.weight,
                global_wgate=(
                    self.compressor.wgate.weight if self.compress_ratio == 2 else None
                ),
                index_wk=self.indexer.wk.weight,
                index_k_norm=self.indexer.k_norm.weight,
            )
        if self.is_index_source:
            view.update(
                index_wq=self.indexer.wq_b,
                index_weights=self.indexer.weights_proj.weight,
            )
        return view
