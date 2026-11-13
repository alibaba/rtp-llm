"""Model-specific DSA parameters, independent of the attention implementation."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DsaConfig:
    # Runtime layer IDs are zero-based; checkpoint schedules are one-based.
    layers: tuple[int, ...]
    index_num_heads: int
    index_head_dim: int
    index_topk: int
    index_rope_dim: int
    index_k_norm_type: str
    indexer_use_fp8: bool
    index_topk_freq: int
    index_skip_topk_offset: int

    @classmethod
    def from_kimi_config(
        cls, text: Mapping[str, Any], *, allow_mtp: bool = False
    ) -> "DsaConfig":
        raw = text.get("dsa_config")
        if not isinstance(raw, Mapping):
            raise ValueError("Kimi K3.1 requires an object-valued dsa_config")
        count = text.get("num_hidden_layers")
        if type(count) is not int or count < 1:
            raise ValueError("num_hidden_layers must be a positive integer")
        if allow_mtp:
            if text.get("num_nextn_predict_layers") != 1:
                raise ValueError(
                    "DSA MTP requires one explicitly configured nextn layer"
                )
            count += 1
        layers = raw.get("dsa_layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("dsa_layers must be a nonempty one-based layer list")
        if any(type(i) is not int or not 1 <= i <= count for i in layers):
            raise ValueError("dsa_layers contains an invalid one-based layer ID")
        if len(set(layers)) != len(layers):
            raise ValueError("dsa_layers contains duplicate layer IDs")
        full = text.get("linear_attn_config", {}).get("full_attn_layers", [])
        if set(layers) != set(full):
            raise ValueError("K3.1 dsa_layers must match full_attn_layers")

        def integer(name, minimum):
            value = raw.get(name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"dsa_config.{name} must be an integer >= {minimum}")
            return value

        result = cls(
            layers=tuple(sorted(i - 1 for i in layers)),
            index_num_heads=integer("index_num_heads", 1),
            index_head_dim=integer("index_head_dim", 1),
            index_topk=integer("index_topk", 1),
            index_rope_dim=integer("index_rope_dim", 0),
            index_k_norm_type=raw.get("index_k_norm_type"),
            indexer_use_fp8=raw.get("indexer_use_fp8", False),
            index_topk_freq=integer("index_topk_freq", 1),
            index_skip_topk_offset=integer("index_skip_topk_offset", 0),
        )
        if type(result.indexer_use_fp8) is not bool:
            raise ValueError("indexer_use_fp8 must be a boolean")
        if result.index_k_norm_type != "layernorm":
            raise ValueError("K3.1 currently requires Indexer LayerNorm")
        if result.index_rope_dim != 0:
            raise ValueError("K3.1 currently requires index_rope_dim=0")
        if result.index_topk_freq != 1 or result.index_skip_topk_offset != 1:
            raise ValueError("K3.1 currently supports TopK frequency/offset 1/1")
        if result.index_head_dim != 128:
            raise ValueError("K3.1 Indexer currently requires head dimension 128")
        return result

    def has_layer(self, layer: int) -> bool:
        return layer in self.layers

    @property
    def index_cache_entry_bytes(self) -> int:
        # BF16 keys have no quantization scale. FP8 uses one FP32 scale per 128.
        if self.indexer_use_fp8:
            return self.index_head_dim + 4 * (self.index_head_dim // 128)
        return 2 * self.index_head_dim
